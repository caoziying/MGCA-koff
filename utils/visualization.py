from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def analyze_and_plot_weights(prot_weights_list, drug_weights_list, save_dir, fold_id):
    all_prot_w = np.concatenate(prot_weights_list, axis=0)
    all_drug_w = np.concatenate(drug_weights_list, axis=0)

    avg_prot_w = all_prot_w.mean(axis=0)
    avg_drug_w = all_drug_w.mean(axis=0)

    prot_labels = ["ESM Layer -4", "ESM Layer -3", "ESM Layer -2", "ESM Layer -1"]
    drug_labels = ["Morgan r=0", "Morgan r=1", "Morgan r=2", "Morgan r=3"]

    sns.set(font_scale=1.5)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        avg_prot_w.reshape(-1, 1),
        annot=True,
        cmap="Blues",
        fmt=".3f",
        yticklabels=prot_labels,
        xticklabels=["Weight"],
        annot_kws={"size": 18},
    )
    plt.title(f"Fold {fold_id}: Protein Layer Importance (Gating)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(1, 2, 2)
    sns.heatmap(
        avg_drug_w.reshape(-1, 1),
        annot=True,
        cmap="Oranges",
        fmt=".3f",
        yticklabels=drug_labels,
        xticklabels=["Weight"],
        annot_kws={"size": 18},
    )
    plt.title(f"Fold {fold_id}: Drug Radius Importance (Gating)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_expert_weights.png"), dpi=300)
    plt.close()

    np.save(os.path.join(save_dir, f"fold_{fold_id}_prot_weights.npy"), all_prot_w)
    np.save(os.path.join(save_dir, f"fold_{fold_id}_drug_weights.npy"), all_drug_w)


def analyze_and_plot_cross_attention(p2d_weights_list, d2p_weights_list, save_dir, fold_id):
    all_p2d = np.concatenate(p2d_weights_list, axis=0)
    all_d2p = np.concatenate(d2p_weights_list, axis=0)

    avg_p2d = all_p2d.mean(axis=0)
    avg_d2p = all_d2p.mean(axis=0)

    prot_labels = ["ESM L-4", "ESM L-3", "ESM L-2", "ESM L-1"]
    drug_labels = ["r=0", "r=1", "r=2", "r=3"]

    sns.set(font_scale=1.5)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        avg_p2d,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=drug_labels,
        yticklabels=prot_labels,
        annot_kws={"size": 18},
    )
    plt.title(f"Fold {fold_id}: Prot Attends Drug (P->D)", fontsize=20)
    plt.xlabel("Drug Experts (Key/Value)", fontsize=16)
    plt.ylabel("Protein Experts (Query)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(1, 2, 2)
    sns.heatmap(
        avg_d2p,
        annot=True,
        fmt=".3f",
        cmap="magma",
        xticklabels=prot_labels,
        yticklabels=drug_labels,
        annot_kws={"size": 18},
    )
    plt.title(f"Fold {fold_id}: Drug Attends Prot (D->P)", fontsize=20)
    plt.xlabel("Protein Experts (Key/Value)", fontsize=16)
    plt.ylabel("Drug Experts (Query)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_cross_attn_map.png"), dpi=600)
    plt.close()


def plot_grand_average_attention(fold_p2d_matrices, fold_d2p_matrices, save_dir, cold_start_mode):
    grand_avg_p2d = np.stack(fold_p2d_matrices).mean(axis=0)
    grand_avg_d2p = np.stack(fold_d2p_matrices).mean(axis=0)

    prot_labels = ["ESM L-4", "ESM L-3", "ESM L-2", "ESM L-1"]
    drug_labels = ["r=0", "r=1", "r=2", "r=3"]

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        grand_avg_p2d,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=drug_labels,
        yticklabels=prot_labels,
    )
    plt.title("Grand Average (5-Fold): Prot Attends Drug")
    plt.xlabel("Drug Experts (Scales)")
    plt.ylabel("Protein Experts (Depths)")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        grand_avg_d2p,
        annot=True,
        fmt=".3f",
        cmap="magma",
        xticklabels=prot_labels,
        yticklabels=drug_labels,
    )
    plt.title("Grand Average (5-Fold): Drug Attends Prot")
    plt.xlabel("Protein Experts (Depths)")
    plt.ylabel("Drug Experts (Scales)")

    plt.tight_layout()
    path = os.path.join(save_dir, "grand_average_attention_map_" + cold_start_mode + "_cold.png")
    plt.savefig(path, dpi=600)
    plt.close()
    return path
