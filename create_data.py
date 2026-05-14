from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export train/test CSV files from a unified_folds_*.pkl split file."
    )
    parser.add_argument("--data_csv", type=Path, default=Path("data/koff.csv"))
    parser.add_argument("--folds_pkl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _resolve_fold_indices(fold_data):
    train_idx, test_idx = None, None

    if isinstance(fold_data, (tuple, list)) and len(fold_data) >= 2:
        train_idx = fold_data[0]
        test_idx = fold_data[1]
    elif isinstance(fold_data, dict):
        train_keys = ["train", "train_idx", "train_index", "train_indices"]
        test_keys = ["test", "test_idx", "test_index", "test_indices"]

        for key in train_keys:
            if key in fold_data:
                train_idx = fold_data[key]
                break
        for key in test_keys:
            if key in fold_data:
                test_idx = fold_data[key]
                break

    return train_idx, test_idx


def main():
    args = parse_args()
    if not args.data_csv.exists() or not args.folds_pkl.exists():
        raise FileNotFoundError(f"Missing input files: data={args.data_csv}, folds={args.folds_pkl}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv).reset_index(drop=True)
    with open(args.folds_pkl, "rb") as f:
        folds = pickle.load(f)

    print(f"Loaded {len(df)} rows from {args.data_csv}")
    print(f"Loaded {len(folds)} folds from {args.folds_pkl}")

    for fold_id, fold_data in enumerate(folds):
        train_idx, test_idx = _resolve_fold_indices(fold_data)
        if train_idx is None or test_idx is None:
            print(f"[Warning] Fold {fold_id}: unsupported fold structure {type(fold_data)}; skipped.")
            continue

        try:
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            if len(train_idx) > 0 and not isinstance(train_idx[0], (int, np.integer)):
                print(
                    f"[Warning] Fold {fold_id}: non-integer fold ids detected. "
                    "Update this script to match the dataset id column if needed."
                )

            train_df.to_csv(args.output_dir / f"train_fold_{fold_id}.csv", index=False)
            test_df.to_csv(args.output_dir / f"test_fold_{fold_id}.csv", index=False)
            print(f"Fold {fold_id} saved: train {train_df.shape}, test {test_df.shape}")
        except Exception as exc:
            print(f"[Error] Fold {fold_id} failed: {exc}")

    print(f"Done. Files saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
