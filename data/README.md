# Data

This folder contains the public dataset files used by `training.py`.

## Layout

```text
data/
  koff.csv                     # Main FASTA, SMILES, pkoff dataset
  example.csv                  # Minimal example row
  raw/                         # Source and auxiliary raw/pretraining files
  folds/
    drug/unified_folds.pkl     # Drug cold-start folds
    target/unified_folds.pkl   # Target/protein cold-start folds
    pair/unified_folds.pkl     # Drug-protein pair cold-start folds
```

## Fold CSV Export

Example:

```bash
python create_data.py \
  --data_csv data/koff.csv \
  --folds_pkl data/folds/drug/unified_folds.pkl \
  --output_dir data/folds/drug_csv
```
