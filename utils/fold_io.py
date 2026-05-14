from __future__ import annotations

import pickle
from pathlib import Path


def load_folds(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_folds(folds, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(folds, f)
