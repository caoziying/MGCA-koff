from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_fingerprint(r: int, mols: Sequence, n_bits: int = 2048, device: str | torch.device = "cpu"):
    fps = []
    valid_mask = []
    for mol in mols:
        if mol is None:
            fps.append(np.zeros((n_bits,), dtype=np.uint8))
            valid_mask.append(0)
            continue
        try:
            bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            fps.append(arr)
            valid_mask.append(1)
        except Exception:
            fps.append(np.zeros((n_bits,), dtype=np.uint8))
            valid_mask.append(0)
    fps = torch.from_numpy(np.stack(fps)).float().to(device)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)
    return fps, valid_mask


@torch.no_grad()
def batch_extract_esm2(
    fasta_list: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 8,
    window_size: int = 2,
):
    """
    Strided window averaging over four ESM2 depths:
    bottom, 1/3 depth, 2/3 depth, and top.
    """
    model.eval()
    feats = []

    n_layers = model.config.num_hidden_layers

    w1 = list(range(1, 1 + window_size))
    start_2 = n_layers // 3
    w2 = list(range(start_2, start_2 + window_size))
    start_3 = (n_layers * 2) // 3
    w3 = list(range(start_3, start_3 + window_size))
    w4 = list(range(n_layers - window_size + 1, n_layers + 1))

    target_windows = [w1, w2, w3, w4]

    print(f"Using strided ESM2 window averaging (window_size={window_size}):")
    labels = ["Bottom (Seq)", "Low-Mid (Struct)", "High-Mid (Pocket)", "Top (Semantic)"]
    for i, window in enumerate(target_windows):
        print(f"   Expert {i + 1} [{labels[i]}]: Layers {window} (avg of {len(window)})")

    if any(idx > n_layers for window in target_windows for idx in window):
        raise ValueError(f"Window size {window_size} is too large for model depth {n_layers}.")

    for i in range(0, len(fasta_list), batch_size):
        batch = fasta_list[i : i + batch_size]

        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}

        mask = enc["attention_mask"].unsqueeze(-1).float()
        lengths = mask.sum(dim=1) + 1e-9

        out = model(**enc, output_hidden_states=True)

        batch_experts = []
        for window_indices in target_windows:
            window_vectors = []
            for layer_idx in window_indices:
                h = out.hidden_states[layer_idx]
                layer_rep = (h * mask).sum(dim=1) / lengths
                window_vectors.append(layer_rep)

            window_stack = torch.stack(window_vectors, dim=1)
            expert_feat = window_stack.mean(dim=1)
            batch_experts.append(expert_feat)

        expert_stack = torch.stack(batch_experts, dim=1)
        feats.append(expert_stack)

    return torch.cat(feats, dim=0)


def load_and_preprocess_data(
    csv_path: str,
    esm2_path: str,
    device: torch.device,
    esm_cache: str = "esm2_feats_update_avg2.pt",
):
    print("Loading data...")
    rows: list[tuple[str, str, float]] = []
    with open(csv_path, mode="r") as f:
        header = f.readline().strip().split(",")
        try:
            fasta_idx = header.index("FASTA")
            smiles_idx = header.index("SMILES")
            label_idx = header.index("pkoff")
        except ValueError:
            fasta_idx, smiles_idx, label_idx = 0, 1, 2

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) > 3:
                fasta = parts[fasta_idx]
                label = float(parts[-1])
                smiles = ",".join(parts[1:-1]) if smiles_idx == 1 else parts[smiles_idx]
            else:
                fasta, smiles, label = parts[fasta_idx], parts[smiles_idx], float(parts[label_idx])
            rows.append((fasta, smiles, float(label)))

    smiles_list = [row[1] for row in rows]
    fasta_list = [row[0] for row in rows]
    labels = np.array([row[2] for row in rows], dtype=float)

    print(f"Loaded {len(smiles_list)} samples.")

    if esm_cache and os.path.exists(esm_cache):
        print(f"Loading cached ESM2 features: {esm_cache}")
        fasta_en = torch.load(esm_cache, map_location=device)
    else:
        print("Extracting ESM2 features. This can take a long time.")
        tokenizer = AutoTokenizer.from_pretrained(esm2_path)
        esm_model = AutoModelForMaskedLM.from_pretrained(esm2_path).to(device)
        fasta_en = batch_extract_esm2(fasta_list, tokenizer, esm_model, device, batch_size=8)
        if esm_cache:
            torch.save(fasta_en.cpu(), esm_cache)
        fasta_en = fasta_en.to(device)

    print(f"ESM2 feature shape: {fasta_en.shape}")

    print("Extracting Morgan fingerprints...")
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    smiles_en_list = []
    for r in range(4):
        fp, _ = get_fingerprint(r, mols, device=device)
        smiles_en_list.append(fp.unsqueeze(1))
    smiles_en = torch.cat(smiles_en_list, dim=1)
    print(f"Morgan fingerprint shape: {smiles_en.shape}")

    y = torch.from_numpy(labels).float().to(device)
    return fasta_en, smiles_en, y, rows
