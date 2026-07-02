# True Fruit Fly Multi-angle Validation

Training and testing scripts for multi-view fruit fly image classification.

## Installation

```bash
pip install -r requirements.txt
```

Install PyTorch from [pytorch.org](https://pytorch.org) with the CUDA version that matches your system.

## Data Format

```
data/
├── dataset1_1.csv
└── Dataset1/
    ├── 1001001_dataset1_D.jpg
    ├── 1001002_dataset1_L.jpg
    └── ...
```

CSV example (`dataset1_1.csv`):

```csv
image_path,label,specimen,split
Dataset1/1001001_dataset1_D.jpg,1001,1001001,train
Dataset1/1001002_dataset1_L.jpg,1001,1001002,train
Dataset1/1001019_dataset1_D.jpg,1001,1001019,val
```

| Column | Description |
|--------|-------------|
| `image_path` | Path to the image relative to the image root directory |
| `label` | Species ID (e.g., 1001, 1003) |
| `specimen` | Specimen ID (not used by the training script; may be kept) |
| `split` | `train` / `val` / `test` |

Specimen IDs are independent across Dataset 1, 2, and 3. For example, `Dataset1_1001001` and `Dataset2_1001001` refer to different specimens.

In Dataset1, the filename suffixes `_D` / `_L` indicate different viewing angles (dorsal / lateral).


## Training

```bash
python train.py \
  --data_dir ./data \
  --csv_file dataset1_1.csv \
  --image_dir . \
  --model_type resnet50 \
  --batch_size 128 \
  --num_epochs 60 \
  --lr 0.0005 \
  --output_dir ./outputs
```

> `--image_dir .` means image paths are relative to the `data_dir` root. If images are under `data/images/`, use `--image_dir images` instead.

Supported models: `ResNet-18` `ResNet-50` `EfficientNet-B0` `MobileNet-v2` `Swin-Tiny` `ViT-Small` `ConvNeXt-B` `Inception-v3` `MaxViT-Tiny`

## Test Set Prediction

**Option 1: Add rows with `split=test` in the CSV**

```bash
python train.py \
  --data_dir ./data \
  --csv_file dataset1_1.csv \
  --image_dir . \
  --test_dir /path/to/test_set \
  --model_type resnet50 \
  --output_dir ./outputs
```
