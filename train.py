import os
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import timm
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super().default(obj)
            
# set seed to make results reproducible
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# define command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Image classification from CSV (train/val; optional test folder).')
    parser.add_argument('--data_dir', type=str, default='', help='Dataset root directory')
    parser.add_argument('--csv_file', type=str, default='dataset.csv', help='CSV file with image paths, labels, and splits')
    parser.add_argument('--image_dir', type=str, default='images', help='Directory containing images')
    parser.add_argument('--model_type', type=str, default='resnet50', 
                        choices=['efficientnet', 'resnet50', 'resnet18', 'swin_tiny','mobilenet','inception','vit','convnext','vit_small',"maxvit"], help='Model type')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--randomrotation', type=int, default=15, help='randomrotation')
    parser.add_argument('--color_jitter', type=float, default=0.2, help='color_jitter')
    parser.add_argument('--shear', type=float, default=0.0,
                        help='RandomAffine shear in degrees (0 disables; e.g. 5–10)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='Training device')
    parser.add_argument('--output_dir', type=str, default='outputs_lr0001/', help='Output directory')
    parser.add_argument('--save_interval', type=int, default=5, 
                        help='Epoch interval to save model checkpoints (default: 5)')
    parser.add_argument('--freeze_backbone', action='store_true', 
                        help='Freeze backbone and only train classification head (especially for ViT with few classes)')
    parser.add_argument('--unfreeze_layers', type=int, default=0,
                        help='Number of ViT encoder layers to unfreeze for fine-tuning (default: 0, only train head)')
    parser.add_argument('--split_column', type=str, default='split', help='Column name for split (train/val/test) in CSV')
    parser.add_argument('--image_path_column', type=str, default='image_path', help='Column name for image paths in CSV')
    parser.add_argument('--label_column', type=str, default='label', help='Column name for labels in CSV')
    parser.add_argument('--test_dir', type=str, default=None, help='Test set directory path (folder-based, if CSV does not contain test split)')
    parser.add_argument(
        '--vit_small_weights',
        type=str,
        default=None,
        help='Optional .safetensors path for ViT-Small backbone (head ignored). If unset, timm ImageNet pretrained weights are used.',
    )
    return parser.parse_args()

# Dataset: single CSV with split column (train / val / test)
class SingleCSVDataset(Dataset):
    def __init__(self, csv_path, image_dir, split, transform=None, 
                 split_column='split', image_path_column='image_path', 
                 label_column='label', class_to_idx=None):
        """
        Rows filtered by split_column == split. If class_to_idx is None, class order
        is derived from all labels in the CSV (sorted unique).
        """
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df[split_column] == split].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No data found for split '{split}' in CSV file {csv_path}")
        
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        self.image_path_column = image_path_column
        self.label_column = label_column
        
        if class_to_idx is None:
            all_df = pd.read_csv(csv_path)
            self.classes = sorted(all_df[label_column].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
        
        self.labels = [self.class_to_idx[label] for label in self.df[self.label_column]]
        
        print(f"Loaded {len(self.df)} images for {split} split")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx][self.image_path_column]
        label = self.labels[idx]
        full_img_path = os.path.join(self.image_dir, img_path)
        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders_from_single_csv(data_dir, csv_file, image_dir, batch_size, 
                                     model_type='resnet', split_column='split',
                                     image_path_column='image_path', label_column='label',
                                     test_dir=None, randomrotation=15, color_jitter=0.2, shear=0.0):
    """Load train/val from CSV; test from CSV split 'test' or from --test_dir ImageFolder."""
    if model_type == 'inception':
        input_size = 299
    else:
        input_size = 224
        
    # train data augmentation
    train_aug_list = [
        #transforms.RandomResizedCrop(input_size),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(randomrotation),
    ]
    if shear > 0:
        train_aug_list.append(transforms.RandomAffine(degrees=0, shear=shear))
    train_aug_list.extend([
        transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_transform = transforms.Compose(train_aug_list)
    
    val_test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_image_dir = os.path.join(data_dir, image_dir)
    csv_path = os.path.join(data_dir, csv_file)
    
    print(f"Loading training data from {csv_path}")
    train_dataset = SingleCSVDataset(
        csv_path, full_image_dir, split='train', 
        transform=train_transform, split_column=split_column,
        image_path_column=image_path_column, label_column=label_column
    )
    
    class_to_idx = train_dataset.class_to_idx
    class_names = train_dataset.classes
    
    print(f"Loading validation data from {csv_path}")
    val_dataset = SingleCSVDataset(
        csv_path, full_image_dir, split='val', 
        transform=val_test_transform, split_column=split_column,
        image_path_column=image_path_column, label_column=label_column,
        class_to_idx=class_to_idx
    )
    
    if test_dir is not None:
        print(f"Loading test data from folder: {test_dir}")
        if not os.path.exists(test_dir):
            raise ValueError(f"Test directory {test_dir} does not exist!")
        
        test_dataset_folder = datasets.ImageFolder(test_dir, transform=val_test_transform)
        
        if len(test_dataset_folder) == 0:
            raise ValueError(f"Test directory {test_dir} is empty!")
        
        test_classes = test_dataset_folder.classes
        
        print(f"Found {len(test_dataset_folder)} test images in {len(test_classes)} classes")
        print(f"Test classes (from folder): {test_classes}")
        print(f"Training classes (from CSV): {class_names}")
        
        train_class_str_map = {str(cls): cls for cls in class_names}
        
        test_classes_set = set(test_classes)
        class_names_set = set(class_names)
        test_classes_str_set = set(str(cls) for cls in test_classes)
        class_names_str_set = set(str(cls) for cls in class_names)
        
        classes_match = (test_classes_set == class_names_set) or (test_classes_str_set == class_names_str_set)
        
        if not classes_match:
            print(f"Warning: Test set classes {test_classes_set} do not exactly match training classes {class_names_set}")
            print(f"  Trying string comparison: test={test_classes_str_set} vs train={class_names_str_set}")
            
            valid_indices = []
            valid_labels = []
            for idx, (img_path, label_idx) in enumerate(test_dataset_folder.samples):
                test_class_name = test_classes[label_idx]
                test_class_str = str(test_class_name)
                
                if test_class_name in class_to_idx:
                    valid_indices.append(idx)
                    valid_labels.append(class_to_idx[test_class_name])
                elif test_class_str in train_class_str_map:
                    train_class_original = train_class_str_map[test_class_str]
                    valid_indices.append(idx)
                    valid_labels.append(class_to_idx[train_class_original])
                else:
                    try:
                        test_class_int = int(test_class_name)
                        if test_class_int in class_to_idx:
                            valid_indices.append(idx)
                            valid_labels.append(class_to_idx[test_class_int])
                    except (ValueError, TypeError):
                        pass
            
            if len(valid_indices) == 0:
                raise ValueError(f"No valid test samples found! All test classes are not in training classes.")
            
            print(f"Filtered to {len(valid_indices)} valid test samples (out of {len(test_dataset_folder)})")
            
            class FilteredDataset(Dataset):
                def __init__(self, original_dataset, valid_indices, valid_labels):
                    self.original_dataset = original_dataset
                    self.valid_indices = valid_indices
                    self.valid_labels = valid_labels
                
                def __len__(self):
                    return len(self.valid_indices)
                
                def __getitem__(self, idx):
                    original_idx = self.valid_indices[idx]
                    img, _ = self.original_dataset[original_idx]
                    label = self.valid_labels[idx]
                    return img, label
            
            test_dataset = FilteredDataset(test_dataset_folder, valid_indices, valid_labels)
        else:
            class_name_mapping = {}
            for test_class in test_classes:
                test_class_str = str(test_class)
                if test_class in class_to_idx:
                    class_name_mapping[test_class] = test_class
                elif test_class_str in train_class_str_map:
                    class_name_mapping[test_class] = train_class_str_map[test_class_str]
                else:
                    try:
                        test_class_int = int(test_class)
                        if test_class_int in class_to_idx:
                            class_name_mapping[test_class] = test_class_int
                    except (ValueError, TypeError):
                        pass
            
            class ReindexedDataset(Dataset):
                def __init__(self, original_dataset, class_to_idx, test_classes_list, class_name_mapping):
                    self.original_dataset = original_dataset
                    self.class_to_idx = class_to_idx
                    self.test_classes_list = test_classes_list
                    self.class_name_mapping = class_name_mapping
                
                def __len__(self):
                    return len(self.original_dataset)
                
                def __getitem__(self, idx):
                    img, label_idx = self.original_dataset[idx]
                    test_class_name = self.test_classes_list[label_idx]
                    train_class_name = self.class_name_mapping.get(test_class_name, test_class_name)
                    new_label_idx = self.class_to_idx[train_class_name]
                    return img, new_label_idx
            
            test_dataset = ReindexedDataset(test_dataset_folder, class_to_idx, test_classes, class_name_mapping)
    else:
        print(f"Loading test data from {csv_path}")
        try:
            test_dataset = SingleCSVDataset(
                csv_path, full_image_dir, split='test', 
                transform=val_test_transform, split_column=split_column,
                image_path_column=image_path_column, label_column=label_column,
                class_to_idx=class_to_idx
            )
        except ValueError as e:
            print(f"Warning: {e}")
            print("Test dataset will be empty. Please provide --test_dir to load test set from folder.")
            class EmptyDataset(Dataset):
                def __len__(self):
                    return 0
                def __getitem__(self, idx):
                    raise IndexError("Empty dataset")
            test_dataset = EmptyDataset()
    
    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    if test_dir is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) if len(test_dataset) > 0 else None
    
    num_classes = len(class_names)
    
    print(f"\nDataset Statistics:")
    print(f"  Classes: {num_classes}")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")
    print(f"  Test images: {len(test_dataset) if len(test_dataset) > 0 else 0}")
    
    if test_loader is not None:
        print(f"  Test loader created successfully with {len(test_dataset)} samples")
    else:
        if test_dir is not None:
            print(f"  Warning: test_dir was provided ({test_dir}) but test_loader is None!")
        else:
            print(f"  Test loader is None (no test data found in CSV and no test_dir provided)")
    
    return train_loader, val_loader, test_loader, class_names, num_classes

def get_data_loaders(data_dir, batch_size, model_type='resnet', 
                     csv_file=None, image_dir=None, split_column='split',
                     image_path_column='image_path', label_column='label',
                     test_dir=None, randomrotation=15, color_jitter=0.2, shear=0.0):
    """CSV mode if csv_file is set; otherwise ImageFolder under data_dir/train|val|test."""
    if csv_file is not None:
        print("Using single CSV data loading mode")
        return get_data_loaders_from_single_csv(
            data_dir, csv_file, image_dir or 'images', 
            batch_size, model_type, split_column,
            image_path_column, label_column, test_dir,
            randomrotation=randomrotation, color_jitter=color_jitter, shear=shear
        )
    else:
        print("Using folder-based data loading mode")
        if model_type == 'inception':
            input_size = 299
        else:
            input_size = 224
            
        # train data augmentation
        train_aug_list = [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(randomrotation),
        ]
        if shear > 0:
            train_aug_list.append(transforms.RandomAffine(degrees=0, shear=shear))
        train_aug_list.extend([
            transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_transform = transforms.Compose(train_aug_list)
        
        val_test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)
        
        # create data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # get class names
        class_names = train_dataset.classes
        num_classes = len(class_names)
        
        return train_loader, val_loader, test_loader, class_names, num_classes

def get_model(model_type, num_classes, device, freeze_backbone=False, unfreeze_layers=0, vit_small_weights=None):
    if model_type == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
    
    elif model_type == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

    elif model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

    elif model_type == 'maxvit':
        # torchvision MaxVit-T: classifier indices end with Linear at [5]
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[5].parameters():
                param.requires_grad = True

    elif model_type == 'swin_tiny':
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
    elif model_type == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        
    elif model_type == 'inception':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.AuxLogits.parameters():
                param.requires_grad = True
        
    elif model_type == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        if freeze_backbone:
            total_layers = len(model.encoder.layers)
            print(f"ViT has {total_layers} encoder layers")
            
            if unfreeze_layers == 0:
                print("Freezing entire ViT backbone, only training classification head...")
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.conv_proj.parameters():
                    param.requires_grad = False
            else:
                print(f"Freezing ViT backbone except last {unfreeze_layers} layers...")
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for i in range(total_layers - unfreeze_layers, total_layers):
                    for param in model.encoder.layers[i].parameters():
                        param.requires_grad = True
                for param in model.heads.parameters():
                    param.requires_grad = True

    elif model_type == 'vit_small':
        if vit_small_weights:
            if not os.path.isfile(vit_small_weights):
                raise FileNotFoundError(f"--vit_small_weights not found: {vit_small_weights}")
            model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)
            from safetensors import safe_open
            state_dict = {}
            with safe_open(vit_small_weights, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
            skipped = [k for k in state_dict if k.startswith('head.')]
            if skipped:
                print(f"Skipping classifier keys ({len(skipped)}): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print("ViT-Small: loaded backbone from safetensors; classifier randomly initialized.")
            if missing_keys:
                print(f"  load_state_dict missing_keys: {missing_keys}")
            if unexpected_keys:
                print(f"  load_state_dict unexpected_keys: {unexpected_keys}")
        else:
            model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
            print("ViT-Small: timm ImageNet pretrained backbone, new classifier head.")

        if freeze_backbone:
            total_layers = len(model.blocks)
            print(f"ViT Small has {total_layers} encoder layers")
            if unfreeze_layers == 0:
                print("Freezing entire ViT Small backbone, only training classification head...")
                for param in model.patch_embed.parameters():
                    param.requires_grad = False
                for param in model.blocks.parameters():
                    param.requires_grad = False
                for param in model.norm.parameters():
                    param.requires_grad = False
            else:
                print(f"Freezing ViT Small backbone except last {unfreeze_layers} layers...")
                for param in model.blocks.parameters():
                    param.requires_grad = False
                for i in range(total_layers - unfreeze_layers, total_layers):
                    for param in model.blocks[i].parameters():
                        param.requires_grad = True
                for param in model.head.parameters():
                    param.requires_grad = True
                for param in model.norm.parameters():
                    param.requires_grad = True
        
    elif model_type == 'convnext':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
       
    return model.to(device)

# train function
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, save_dir, save_interval=5, model_type='resnet', freeze_backbone=False):
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    # create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # record training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        # each epoch has train and val phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to train mode
            else:
                model.eval()   # set model to eval mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero gradients
                optimizer.zero_grad()
                
                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    if model_type == 'inception' and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2  # Inception v3 auxiliary head (Szegedy et al.)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # backward pass + optimization (only in train phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                # update learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    history['lr'].append(current_lr)
                    print(f'Current learning rate: {current_lr:.6f}')
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
            # if best val accuracy, save model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save best model and delete previous best model
                torch.save(model.state_dict(), best_model_path)
                print(f'New best model saved with accuracy: {best_acc:.4f}')
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1} to {checkpoint_path}')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val accuracy: {best_acc:4f}')
    
    last_model_path = os.path.join(save_dir, 'last.pt')
    torch.save(model.state_dict(), last_model_path)
    print(f'Last epoch model saved to {last_model_path}')
    
    # save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    full_history = {
        'epochs': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'lr': history['lr'] if 'lr' in history else []
    }
    with open(os.path.join(save_dir, 'full_history.json'), 'w') as f:
        json.dump(full_history, f, indent=2)
    
    # plot training process
    plot_training_history(history, save_dir)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# plot training history
def plot_training_history(history, save_dir):
    # set plot style
    plt.style.use('seaborn-v0_8-dark')
    
    # set Times New Roman font
    import matplotlib.font_manager as fm
    
    # try to use Times New Roman font, if not exist, use default font
    try:
        times_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
        if not os.path.exists(times_font_path):
            times_font_path = fm.findfont('Times New Roman')
        font_prop = fm.FontProperties(fname=times_font_path)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['pdf.fonttype'] = 42  # editable text in PDF
    except:
        # if font setting failed, use default font
        font_prop = fm.FontProperties()
    
    # create square subplot - modify figsize to square ratio
    plt.figure(figsize=(12, 6))
    
    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontproperties=font_prop, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, alpha=0.3)
    
    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontproperties=font_prop, fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontproperties=font_prop, fontsize=12)
    plt.xlabel('Epoch', fontproperties=font_prop, fontsize=12)
    plt.legend(prop=font_prop)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # save PNG format (high DPI)
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    # save SVG format
    plt.savefig(os.path.join(save_dir, 'training_history.svg'), format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_history.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    plot_data = {
        'epochs': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'lr': history.get('lr', [])
    }
    with open(os.path.join(save_dir, 'training_history_plot_data.json'), 'w') as f:
        json.dump(plot_data, f, cls=NumpyEncoder, indent=2)
    
    np.save(os.path.join(save_dir, 'training_history_plot_data.npy'), plot_data)

def evaluate_model(model, test_loader, criterion, device, class_names, save_dir):
    model.eval()
    
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}')
    
    # analyze test set class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    
    print(f"\nTest set class distribution:")
    for i, (label_idx, count) in enumerate(zip(unique_labels, counts)):
        print(f"{class_names[label_idx]}: {count} samples")
    
    # calculate detailed classification metrics - use weighted average
    print("\n=== Detailed Classification Metrics ===")
    
    overall_accuracy = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print(f"\nMain Evaluation Metrics:")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Macro Average Precision: {precision_macro:.4f}")
    print(f"Macro Average Recall: {recall_macro:.4f}")
    print(f"Macro Average F1-Score: {f1_macro:.4f}")
    print(f"Weighted Average Precision: {precision_weighted:.4f}")
    print(f"Weighted Average Recall: {recall_weighted:.4f}")
    print(f"Weighted Average F1-Score: {f1_weighted:.4f}")
    
    # detailed metrics for each class
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"\nDetailed Classification Metrics:")
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Samples':<10}")
    print("-" * 75)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
              f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
    
    # calculate precision for each class
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    # fix class distribution dictionary creation
    class_distribution = {}
    for label_idx, count in zip(unique_labels, counts):
        class_distribution[class_names[label_idx]] = int(count)
    
    # save test results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc.item(),
        'dataset_analysis': {
            'class_distribution': class_distribution
        },
        'overall_metrics': {
            'accuracy': overall_accuracy,
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted)
        },
        'class_metrics': {},
        'class_accuracy': {}
    }
    
    # save detailed metrics for each class
    for i, class_name in enumerate(class_names):
        results['class_metrics'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
        
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            results['class_accuracy'][class_name] = float(accuracy)
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    cm_dict = {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names
    }
    with open(os.path.join(save_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm_dict, f, cls=NumpyEncoder, indent=2)
    
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
    np.save(os.path.join(save_dir, 'confusion_matrix_normalized.npy'), cm_normalized)
    
    predictions_data = {
        'all_predictions': all_preds,
        'all_labels': all_labels,
        'class_names': class_names
    }
    with open(os.path.join(save_dir, 'predictions_and_labels.json'), 'w') as f:
        json.dump(predictions_data, f, cls=NumpyEncoder, indent=2)
    
    np.save(os.path.join(save_dir, 'predictions.npy'), np.array(all_preds))
    np.save(os.path.join(save_dir, 'labels.npy'), np.array(all_labels))
    
    plot_confusion_matrix(cm, class_names, save_dir)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    def convert_keys_to_string(d):
        """Recursively stringify dict keys for JSON."""
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                new_key = str(k) if not isinstance(k, str) else k
                new_dict[new_key] = convert_keys_to_string(v)
            return new_dict
        elif isinstance(d, list):
            return [convert_keys_to_string(item) for item in d]
        else:
            return d
    
    report_converted = convert_keys_to_string(report)
    results['classification_report'] = report_converted
    
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    
    # save a readable text report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== Image Classification Evaluation Report ===\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        
        f.write("Main Evaluation Metrics:\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Macro Average Precision: {precision_macro:.4f}\n")
        f.write(f"Macro Average Recall: {recall_macro:.4f}\n")
        f.write(f"Macro Average F1-Score: {f1_macro:.4f}\n")
        f.write(f"Weighted Average Precision: {precision_weighted:.4f}\n")
        f.write(f"Weighted Average Recall: {recall_weighted:.4f}\n")
        f.write(f"Weighted Average F1-Score: {f1_weighted:.4f}\n\n")
        
        f.write("Detailed Classification Metrics:\n")
        f.write(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Samples':<10}\n")
        f.write("-" * 75 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{str(class_name):<25} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                   f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}\n")
        
        f.write(f"\nDetailed Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=[str(cls) for cls in class_names], zero_division=0))
    
    return results

def plot_confusion_matrix(cm, class_names, save_dir):
    """Plot confusion matrix (counts + row-normalized) to PNG/SVG/PDF."""
    # set Times New Roman font
    import matplotlib.font_manager as fm
    
    try:
        times_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
        times_italic_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf'
        
        if not os.path.exists(times_font_path):
            times_font_path = fm.findfont('Times New Roman')
            times_italic_font_path = times_font_path
        
        font_prop = fm.FontProperties(fname=times_font_path)
        font_prop_italic = fm.FontProperties(fname=times_italic_font_path)
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['pdf.fonttype'] = 42
    except:
        font_prop = fm.FontProperties()
        font_prop_italic = fm.FontProperties()
    
    plt.figure(figsize=(10, 8))
    
    # plot heatmap - use original number (not normalized)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontproperties=font_prop, fontsize=16)
    plt.ylabel('True Label', fontproperties=font_prop, fontsize=14)
    plt.xlabel('Predicted Label', fontproperties=font_prop, fontsize=14)
    
    # set tick labels to italic
    plt.xticks(fontproperties=font_prop_italic, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontproperties=font_prop_italic, fontsize=10)
    
    plt.tight_layout()
    
    # save PNG format (high DPI)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    # save SVG format
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    cm_normalized_dict = {
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names
    }
    with open(os.path.join(save_dir, 'confusion_matrix_normalized_data.json'), 'w') as f:
        json.dump(cm_normalized_dict, f, cls=NumpyEncoder, indent=2)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1)
    
    plt.title('Normalized Confusion Matrix (Recall)', fontproperties=font_prop, fontsize=16)
    plt.ylabel('True Label', fontproperties=font_prop, fontsize=14)
    plt.xlabel('Predicted Label', fontproperties=font_prop, fontsize=14)
    
    # set tick labels to italic
    plt.xticks(fontproperties=font_prop_italic, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontproperties=font_prop_italic, fontsize=10)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    with PdfPages(os.path.join(save_dir, 'confusion_matrix.pdf')) as pdf:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix (Counts)', fontproperties=font_prop, fontsize=16)
        plt.ylabel('True Label', fontproperties=font_prop, fontsize=14)
        plt.xlabel('Predicted Label', fontproperties=font_prop, fontsize=14)
        
        plt.xticks(fontproperties=font_prop_italic, rotation=45, ha='right', fontsize=10)
        plt.yticks(fontproperties=font_prop_italic, fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1)
        
        plt.title('Normalized Confusion Matrix (Recall)', fontproperties=font_prop, fontsize=16)
        plt.ylabel('True Label', fontproperties=font_prop, fontsize=14)
        plt.xlabel('Predicted Label', fontproperties=font_prop, fontsize=14)
        
        plt.xticks(fontproperties=font_prop_italic, rotation=45, ha='right', fontsize=10)
        plt.yticks(fontproperties=font_prop_italic, fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(dpi=300, bbox_inches='tight')
        plt.close()
    
    return cm

def main():
    args = parse_args()
    set_seed(42)
    
    # create output directory
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, cls=NumpyEncoder, indent=2)
    
    config = {
        'data_dir': args.data_dir,
        'csv_file': args.csv_file,
        'image_dir': args.image_dir,
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'device': args.device,
        'freeze_backbone': args.freeze_backbone,
        'unfreeze_layers': args.unfreeze_layers,
        'save_interval': args.save_interval,
        'split_column': args.split_column,
        'image_path_column': args.image_path_column,
        'label_column': args.label_column,
        'test_dir': args.test_dir,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'randomrotation': args.randomrotation,
        'color_jitter': args.color_jitter,
        'shear': args.shear,
        'vit_small_weights': args.vit_small_weights,
    }
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, cls=NumpyEncoder, indent=2)
        
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names, num_classes = get_data_loaders(
        args.data_dir, args.batch_size, args.model_type,
        csv_file=args.csv_file, 
        image_dir=args.image_dir,
        split_column=args.split_column,
        image_path_column=args.image_path_column,
        label_column=args.label_column,
        test_dir=args.test_dir,
        randomrotation=args.randomrotation,
        color_jitter=args.color_jitter,
        shear=args.shear
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    class_names_serializable = []
    for cls in class_names:
        if isinstance(cls, np.integer):
            class_names_serializable.append(int(cls))
        elif isinstance(cls, np.floating):
            class_names_serializable.append(float(cls))
        elif isinstance(cls, np.ndarray):
            class_names_serializable.append(cls.tolist())
        else:
            class_names_serializable.append(cls)
            
    class_names = class_names_serializable
    
    with open(os.path.join(output_dir, 'classes.json'), 'w') as f:
        json.dump(class_names_serializable, f, cls=NumpyEncoder, indent=2)
    
    print(f"Detected {num_classes} classes: {class_names}")
    
    # initialize model
    print(f"Initializing {args.model_type} model...")
    model = get_model(
        args.model_type,
        num_classes,
        args.device,
        args.freeze_backbone,
        args.unfreeze_layers,
        vit_small_weights=args.vit_small_weights,
    )
    
    if args.freeze_backbone:
        print("Using frozen backbone - only training specified layers")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, weight_decay=0.01
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    # use CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
    )
    
    # train model
    print("Starting training...")
    model = train_model(model, dataloaders, criterion, optimizer, 
                        scheduler, args.device, args.num_epochs, output_dir,
                        save_interval=args.save_interval, model_type=args.model_type,
                        freeze_backbone=args.freeze_backbone)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(f"# Model Training Results\n\n")
        f.write(f"**Model Type**: {args.model_type}\n")
        f.write(f"**Training Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Number of Classes**: {num_classes}\n")
        f.write(f"**Classes**: {', '.join(str(cls) for cls in class_names)}\n")
        f.write(f"**CSV File**: {args.csv_file}\n")
        f.write(f"**Image Directory**: {args.image_dir}\n")
        f.write(f"**Freeze Backbone**: {args.freeze_backbone}\n")
        f.write(f"**Unfreeze Layers**: {args.unfreeze_layers}\n")
        f.write(f"**Learning Rate**: {args.lr}\n")
        f.write(f"**Batch Size**: {args.batch_size}\n")
        f.write(f"**Epochs**: {args.num_epochs}\n")
        if args.model_type == 'vit_small':
            f.write(f"**vit_small_weights**: {args.vit_small_weights!r}\n")
        f.write("\n")
        
        f.write("## Files\n\n")
        f.write("- `best_model.pth`: Best model weights (PyTorch format)\n")
        f.write("- `last.pt`: Last epoch model weights (used for evaluation)\n")
        f.write("- `pytorch_model.bin`: Model weights in binary format\n")
        f.write("- `epoch_*.pth`: Model checkpoints at different epochs\n")
        f.write("- `training_history.png/svg/pdf`: Training loss and accuracy plots\n")
        f.write("- `training_history_plot_data.json/npy`: Raw data for training history plots\n")
        f.write("- `confusion_matrix.png/svg/pdf`: Confusion matrix visualizations\n")
        f.write("- `confusion_matrix.json`: Confusion matrix data in JSON format (includes normalized)\n")
        f.write("- `confusion_matrix.npy`: Confusion matrix data in numpy format\n")
        f.write("- `confusion_matrix_normalized.npy`: Normalized confusion matrix data\n")
        f.write("- `confusion_matrix_normalized_data.json`: Normalized confusion matrix in JSON format\n")
        f.write("- `predictions_and_labels.json`: All predictions and true labels\n")
        f.write("- `predictions.npy`: Predictions array\n")
        f.write("- `labels.npy`: True labels array\n")
        f.write("- `classification_report.json`: Detailed classification metrics\n")
        f.write("- `test_results.json`: Complete test results\n")
        f.write("- `training_config.json`: Training configuration\n")
        f.write("- `classes.json`: Class names\n")
        f.write("- `history.json`: Training history\n")
        f.write("- `full_history.json`: Complete training history with all epochs\n")
        f.write("- `args.json`: Command line arguments\n")
    
    if test_loader is not None:
        print("Loading last epoch model (last.pt) for evaluation...")
        last_model_path = os.path.join(output_dir, 'last.pt')
        if os.path.exists(last_model_path):
            model.load_state_dict(torch.load(last_model_path))
            print(f"Loaded last epoch model from {last_model_path}")
        else:
            print(f"Warning: {last_model_path} not found, using current model state")
        
        print("Evaluating model on test set...")
        results = evaluate_model(model, test_loader, criterion, args.device, class_names, output_dir)
    else:
        print("Warning: Test loader is None. Skipping test evaluation.")
        results = None
    
    print(f"All results and models saved to {output_dir}")
    
    print(f"\nGenerated files in {output_dir}:")
    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_kb = size / 1024
            if size_kb < 1:
                size_str = f"{size} B"
            elif size_kb < 1024:
                size_str = f"{size_kb:.1f} KB"
            else:
                size_str = f"{size_kb/1024:.1f} MB"
            print(f"  - {file}: {size_str}")

if __name__ == "__main__":
    main()