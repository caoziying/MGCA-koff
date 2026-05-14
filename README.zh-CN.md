# KOFF Prediction

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供 KOFF 预测模型的论文公开代码版本。模型使用 ESM2 蛋白特征和 Morgan 药物指纹。目录结构参考 DeepDTA、GraphDTA、AttentionDTA 等经典 DTA 论文代码仓库：数据放在 `data/`，神经网络定义放在 `models/`，通用工具放在 `utils/`，主要流程由根目录脚本运行。

## 模型架构

![MGCA-KOFF 模型架构](assets/MGCA-koff.png)

## 仓库结构

```text
online/
  assets/
    MGCA-koff.png                # 模型架构图
  training.py                    # 训练和评估入口
  create_data.py                 # 将 pkl folds 导出为 train/test CSV
  data/
    koff.csv                     # FASTA, SMILES, pkoff 监督数据集
    example.csv                  # 最小示例数据
    raw/                         # 原始和辅助预训练数据
    folds/
      drug/unified_folds.pkl     # 药物冷启动 folds
      target/unified_folds.pkl   # 靶标/蛋白冷启动 folds
      pair/unified_folds.pkl     # 药物-蛋白对冷启动 folds
  models/
    layers.py                    # 专家融合、交叉注意力和 MoE 模块
    esm2_morgan_moe.py           # FullRegressionTransformer
  utils/
    cold_start.py                # 冷启动划分策略
    dataset.py                   # PyTorch 数据集封装
    features.py                  # ESM2 和 Morgan 特征提取
    fold_io.py                   # Fold pickle 读写
    metrics.py                   # 回归指标
    seed.py                      # 随机种子设置
    trainer.py                   # 交叉验证训练流程
    visualization.py             # 专家权重和注意力可视化
```

## 安装

使用 Python 3.10 或更新版本，然后在该目录下安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集

默认监督数据集为 `data/koff.csv`，包含以下列：

```csv
FASTA,SMILES,pkoff
MSEQUENCE...,CCO,1.23
```

预处理好的 fold 文件位于：

```text
data/folds/drug/unified_folds.pkl
data/folds/target/unified_folds.pkl
data/folds/pair/unified_folds.pkl
```

如需将 fold 文件导出为每折的 CSV 文件：

```bash
python create_data.py \
  --data_csv data/koff.csv \
  --folds_pkl data/folds/drug/unified_folds.pkl \
  --output_dir data/folds/drug_csv
```

## ESM2 骨干模型

模型使用 Hugging Face `transformers` 提取 ESM2 蛋白表示。论文配置使用 ESM2 t36 3B checkpoint，其隐藏层维度为 `2560`。

可以使用本地 checkpoint 目录：

```bash
python training.py \
  --dataset_csv data/koff.csv \
  --esm2_path ../pretrained_model/esm2_t36 \
  --cold_start_mode drug
```

也可以直接传入 Hugging Face 模型 ID：

```bash
python training.py \
  --dataset_csv data/koff.csv \
  --esm2_path facebook/esm2_t36_3B_UR50D \
  --cold_start_mode drug
```

离线或可复现实验建议先下载 checkpoint：

```bash
huggingface-cli download facebook/esm2_t36_3B_UR50D \
  --local-dir ../pretrained_model/esm2_t36 \
  --local-dir-use-symlinks False
```

首次运行会将提取好的 ESM2 特征缓存到 `esm2_feats_update_avg2.pt`。后续运行会复用该缓存，除非删除该文件或通过 `--esm_cache` 指定新的缓存路径。

## 训练

运行药物冷启动训练：

```bash
python training.py \
  --dataset_csv data/koff.csv \
  --esm2_path ../pretrained_model/esm2_t36 \
  --cold_start_mode drug
```

支持的冷启动模式：

```text
drug
target
pair
```

默认情况下，`training.py` 从以下路径加载 fold：

```text
data/folds/{cold_start_mode}/unified_folds.pkl
```

如果 fold 文件不存在，脚本会回退到 shuffled KFold，用于调试；该行为与原始开发脚本保持一致。

## 输出

默认输出目录为：

```text
results/esm2_morgan_gated_crossatt_moe/{cold_start_mode}/
```

每折 checkpoint 保存为：

```text
model_{cold_start_mode}_fold{fold_id}.pt
```

checkpoint 中包含 `model_state_dict`、模型重建配置和该折测试指标。

## 说明

- 本仓库不包含预训练 ESM2 权重。
