# KOFF Prediction

[English](README.md) | [简体中文](README.zh-CN.md)

This repository provides the public-release implementation of the ESM2 and Morgan fingerprint model for KOFF prediction. The layout follows the common style of classic DTA codebases such as DeepDTA, GraphDTA, and AttentionDTA: data files are kept under `data/`, neural network definitions under `models/`, shared helpers under `utils/`, and root-level scripts run the main workflows.

## Repository Layout

```text
online/
  training.py                    # Main training and evaluation entry point
  create_data.py                 # Export pkl folds to train/test CSV files
  data/
    koff.csv                     # FASTA, SMILES, pkoff supervised dataset
    example.csv                  # Minimal example row
    raw/                         # Raw and auxiliary pretraining files
    folds/
      drug/unified_folds.pkl     # Drug cold-start folds
      target/unified_folds.pkl   # Target/protein cold-start folds
      pair/unified_folds.pkl     # Drug-protein pair cold-start folds
  models/
    layers.py                    # Expert fusion, cross attention, and MoE blocks
    esm2_morgan_moe.py           # FullRegressionTransformer
  utils/
    cold_start.py                # Cold-start split strategies
    dataset.py                   # PyTorch dataset wrapper
    features.py                  # ESM2 and Morgan feature extraction
    fold_io.py                   # Fold pickle I/O
    metrics.py                   # Regression metrics
    seed.py                      # Reproducibility helper
    trainer.py                   # Cross-validation training loop
    visualization.py             # Expert and attention visualizations
```

## Installation

Use Python 3.10 or newer, then install dependencies from this directory:

```bash
pip install -r requirements.txt
```

## Dataset

The default supervised dataset is `data/koff.csv`. It contains:

```csv
FASTA,SMILES,pkoff
MSEQUENCE...,CCO,1.23
```

Precomputed fold files are stored as:

```text
data/folds/drug/unified_folds.pkl
data/folds/target/unified_folds.pkl
data/folds/pair/unified_folds.pkl
```

To export a fold file into per-fold CSV files:

```bash
python create_data.py \
  --data_csv data/koff.csv \
  --folds_pkl data/folds/drug/unified_folds.pkl \
  --output_dir data/folds/drug_csv
```

## ESM2 Backbone

The model uses ESM2 protein representations extracted with Hugging Face `transformers`. The paper configuration uses the ESM2 t36 3B checkpoint, whose hidden size is `2560`.

Use a local checkpoint directory:

```bash
python training.py \
  --dataset_csv data/koff.csv \
  --esm2_path ../pretrained_model/esm2_t36 \
  --cold_start_mode drug
```

Or pass the Hugging Face model id directly:

```bash
python training.py \
  --dataset_csv data/koff.csv \
  --esm2_path facebook/esm2_t36_3B_UR50D \
  --cold_start_mode drug
```

For offline or reproducible runs, download the checkpoint first:

```bash
huggingface-cli download facebook/esm2_t36_3B_UR50D \
  --local-dir ../pretrained_model/esm2_t36 \
  --local-dir-use-symlinks False
```

The first run caches extracted ESM2 features to `esm2_feats_update_avg2.pt`. Later runs reuse this cache unless the file is removed or `--esm_cache` is changed.

## Training

Run drug cold-start training:

```bash
python training.py \
  --dataset_csv data/koff.csv \
  --esm2_path ../pretrained_model/esm2_t36 \
  --cold_start_mode drug
```

Supported cold-start modes are:

```text
drug
target
pair
```

By default, `training.py` loads folds from:

```text
data/folds/{cold_start_mode}/unified_folds.pkl
```

If a fold file is unavailable, the script falls back to shuffled KFold for debugging, matching the original development behavior.

## Outputs

By default, outputs are written to:

```text
results/esm2_morgan_gated_crossatt_moe/{cold_start_mode}/
```

Each fold checkpoint is saved as:

```text
model_{cold_start_mode}_fold{fold_id}.pt
```

The checkpoint contains `model_state_dict`, model reconstruction config, and fold test metrics.

## Notes

- Pretrained ESM2 weights are not included.
