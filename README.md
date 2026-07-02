# True Fruit Fly Multi-angle Validation

实蝇多视角图像分类训练与测试脚本。

## 安装

```bash
pip install -r requirements.txt
```

PyTorch 需按 [pytorch.org](https://pytorch.org) 选择对应 CUDA 版本安装。

## 数据格式

```
data/
├── dataset1_1.csv
└── Dataset1/
    ├── 1001001_dataset1_D.jpg
    ├── 1001002_dataset1_L.jpg
    └── ...
```

CSV 示例（`dataset1_1.csv`）：

```csv
image_path,label,specimen,split
Dataset1/1001001_dataset1_D.jpg,1001,1001001,train
Dataset1/1001002_dataset1_L.jpg,1001,1001002,train
Dataset1/1001019_dataset1_D.jpg,1001,1001019,val
```

| 列名 | 说明 |
|------|------|
| `image_path` | 相对图像根目录的路径 |
| `label` | 物种编号（如 1001、1003） |
| `specimen` | 标本编号（训练脚本未使用，可保留） |
| `split` | `train` / `val` / `test` |

Dataset1中文件名后缀 `_D` / `_L` 表示不同拍摄角度（背视 / 侧视）。

## 训练

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

> `--image_dir .` 表示图像路径相对 `data_dir` 根目录；若图像在 `data/images/` 下则改为 `--image_dir images`。

可选模型：`resnet18` `resnet50` `efficientnet` `mobilenet` `swin_tiny` `vit` `vit_small` `convnext` `inception` `maxvit`

## 指定 test 集预测

**方式 1：CSV 中增加 `split=test` 行**

```bash
python train.py \
  --data_dir ./data \
  --csv_file dataset1_1.csv \
  --image_dir . \
  --test_dir /path/to/test_set \
  --model_type resnet50 \
  --output_dir ./outputs
```