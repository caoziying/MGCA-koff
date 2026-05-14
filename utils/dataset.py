from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ESM2MorganDataset(Dataset):
    def __init__(self, smiles, fasta, labels):
        self.smiles = smiles
        self.fasta = fasta
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.fasta[idx], self.labels[idx]
