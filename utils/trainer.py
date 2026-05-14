from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from models import FullRegressionTransformer
from utils.dataset import ESM2MorganDataset
from utils.features import load_and_preprocess_data
from utils.fold_io import load_folds
from utils.metrics import compute_metrics
from utils.seed import set_seed
from utils.visualization import (
    analyze_and_plot_cross_attention,
    analyze_and_plot_weights,
    plot_grand_average_attention,
)


def train_one_fold(model, train_loader, val_loader, device, epochs, lr, val_freq=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for smiles_batch, fasta_batch, labels_batch in train_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)

            optimizer.zero_grad()
            preds, _ = model(fasta_batch, smiles_batch)
            loss = criterion(preds.squeeze(), labels_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if epoch % val_freq == 0 or epoch == epochs:
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for smiles_batch, fasta_batch, labels_batch in val_loader:
                    smiles_batch = smiles_batch.float().to(device)
                    fasta_batch = fasta_batch.float().to(device)
                    labels_batch = labels_batch.float().to(device)
                    preds, _ = model(fasta_batch, smiles_batch)
                    val_preds.append(preds.squeeze().cpu().numpy())
                    val_trues.append(labels_batch.cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_trues = np.concatenate(val_trues)
            metrics = compute_metrics(val_trues, val_preds)
            rmse = metrics["rmse"]
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(
                f"Epoch {epoch}: Val RMSE={rmse:.6f}, "
                f"Pearson={metrics['pearson']:.4f} (Best RMSE={best_val_rmse:.6f})"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    train_preds, train_trues = [], []
    with torch.no_grad():
        for smiles_batch, fasta_batch, labels_batch in train_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)
            preds, _ = model(fasta_batch, smiles_batch)
            train_preds.append(preds.squeeze().cpu().numpy())
            train_trues.append(labels_batch.cpu().numpy())
    train_metrics = compute_metrics(np.concatenate(train_trues), np.concatenate(train_preds))

    val_preds, val_trues = [], []
    prot_weights_list, drug_weights_list = [], []
    p2d_attn_list, d2p_attn_list = [], []

    with torch.no_grad():
        for smiles_batch, fasta_batch, labels_batch in val_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)

            preds, (w_p, w_d, w_p2d, w_d2p) = model(fasta_batch, smiles_batch)

            val_preds.append(preds.squeeze().cpu().numpy())
            val_trues.append(labels_batch.cpu().numpy())

            prot_weights_list.append(w_p.cpu().numpy())
            drug_weights_list.append(w_d.cpu().numpy())
            p2d_attn_list.append(w_p2d.cpu().numpy())
            d2p_attn_list.append(w_d2p.cpu().numpy())

    val_metrics = compute_metrics(np.concatenate(val_trues), np.concatenate(val_preds))

    return (
        train_metrics,
        val_metrics,
        np.concatenate(val_trues),
        np.concatenate(val_preds),
        prot_weights_list,
        drug_weights_list,
        p2d_attn_list,
        d2p_attn_list,
    )


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, default="data/koff.csv")
    parser.add_argument("--esm2_path", type=str, default="../pretrained_model/esm2_t36")
    parser.add_argument("--output_dir", type=str, default="results/esm2_morgan_gated_crossatt_moe")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.17)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cold_start_mode", type=str, default="drug", choices=["drug", "target", "pair"])
    parser.add_argument("--ablation", type=str, default="no", choices=["no", "drug", "target", "bicross"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--save_models", type=_parse_bool, default=True)
    parser.add_argument("--esm_cache", type=str, default="esm2_feats_update_avg2.pt")
    parser.add_argument("--folds_path", type=str, default=None)
    return parser.parse_args(argv)


def train_cross_validation(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.ablation != "no":
        args.output_dir = args.output_dir + "_ablation_" + args.ablation
    args.output_dir = os.path.join(args.output_dir, args.cold_start_mode)
    os.makedirs(args.output_dir, exist_ok=True)

    total_start = time.time()

    fasta_en, smiles_en, y, rows = load_and_preprocess_data(
        args.dataset_csv,
        args.esm2_path,
        device,
        esm_cache=args.esm_cache,
    )

    mode_names = {"drug": "drug", "target": "protein", "pair": "drug-protein pair"}
    print(f"Loading {mode_names.get(args.cold_start_mode, 'drug')} cold-start folds...")

    folds_path = args.folds_path or f"data/folds/{args.cold_start_mode}/unified_folds.pkl"
    try:
        folds = load_folds(folds_path)
    except Exception:
        print("Fold file not found. Falling back to shuffled KFold for debugging.")
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        folds = list(kf.split(np.arange(len(y))))

    print(f"Start training with {len(folds)} folds.")
    all_metrics = []
    all_preds = np.zeros((len(y),), dtype=np.float32)
    all_trues = np.zeros((len(y),), dtype=np.float32)

    fold_p2d_matrices = []
    fold_d2p_matrices = []

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_id + 1} / {len(folds)} ===")
        fold_start = time.time()

        train_ds = ESM2MorganDataset(smiles_en[train_idx], fasta_en[train_idx], y[train_idx])
        test_ds = ESM2MorganDataset(smiles_en[test_idx], fasta_en[test_idx], y[test_idx])

        train_sub_len = int(0.9 * len(train_ds))
        train_sub_ds, val_sub_ds = torch.utils.data.random_split(
            train_ds,
            [train_sub_len, len(train_ds) - train_sub_len],
            generator=torch.Generator().manual_seed(args.seed),
        )

        train_loader = DataLoader(train_sub_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_sub_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = FullRegressionTransformer(
            proj_dim1=2560,
            proj_dim2=2048,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            nums_of_experts=4,
            num_heads=8,
            moe_num_experts=4,
            ablation=args.ablation,
        ).to(device)

        train_met, val_met, _, _, prot_w_val, drug_w_val, p2d_attn_val, d2p_attn_val = train_one_fold(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
        )

        model.eval()
        test_preds, test_trues = [], []
        prot_w_test, drug_w_test = [], []
        p2d_attn_test, d2p_attn_test = [], []

        with torch.no_grad():
            for smiles_batch, fasta_batch, labels_batch in test_loader:
                smiles_batch = smiles_batch.float().to(device)
                fasta_batch = fasta_batch.float().to(device)
                labels_batch = labels_batch.float().to(device)
                preds, (w_p, w_d, w_p2d, w_d2p) = model(fasta_batch, smiles_batch)

                test_preds.append(preds.squeeze().cpu().numpy())
                test_trues.append(labels_batch.cpu().numpy())
                prot_w_test.append(w_p.cpu().numpy())
                drug_w_test.append(w_d.cpu().numpy())
                p2d_attn_test.append(w_p2d.cpu().numpy())
                d2p_attn_test.append(w_d2p.cpu().numpy())

        test_preds = np.concatenate(test_preds)
        test_trues = np.concatenate(test_trues)
        test_metrics = compute_metrics(test_trues, test_preds)

        all_trues[test_idx] = test_trues
        all_preds[test_idx] = test_preds

        fold_dur = time.time() - fold_start

        print(f"Fold {fold_id + 1} Results:")
        print(f"  Train - RMSE: {train_met['rmse']:.4f}, R2: {train_met['r2']:.4f}, PCC: {train_met['pearson']:.4f}")
        print(f"  Val   - RMSE: {val_met['rmse']:.4f}, R2: {val_met['r2']:.4f}, PCC: {val_met['pearson']:.4f}")
        print(f"  Test  - RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}, PCC: {test_metrics['pearson']:.4f}")

        record = {
            "fold": fold_id + 1,
            "duration": fold_dur,
        }
        for phase, metrics in [("train", train_met), ("val", val_met), ("test", test_metrics)]:
            for k, v in metrics.items():
                record[f"{phase}_{k}"] = v
        all_metrics.append(record)

        analyze_and_plot_weights(prot_w_test, drug_w_test, args.output_dir, fold_id + 1)

        current_fold_p2d_mean = np.concatenate(p2d_attn_test, axis=0).mean(axis=0)
        current_fold_d2p_mean = np.concatenate(d2p_attn_test, axis=0).mean(axis=0)

        fold_p2d_matrices.append(current_fold_p2d_mean)
        fold_d2p_matrices.append(current_fold_d2p_mean)

        analyze_and_plot_cross_attention(p2d_attn_test, d2p_attn_test, args.output_dir, fold_id + 1)

        if args.save_models:
            save_name = f"model_{args.cold_start_mode}_fold{fold_id + 1}.pt"
            save_path = os.path.join(args.output_dir, save_name)

            checkpoint = {
                "fold": fold_id + 1,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "model_state_dict": model.state_dict(),
                "config": {
                    "proj_dim1": 2560,
                    "proj_dim2": 2048,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "nums_of_experts": 4,
                    "num_heads": 8,
                    "moe_num_experts": 4,
                    "ablation": args.ablation,
                },
                "metrics": {
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                },
            }

            torch.save(checkpoint, save_path)
            print(f"Model saved: {save_path}")

    total_dur = time.time() - total_start
    overall = compute_metrics(all_trues, all_preds)

    print("Plotting grand-average attention map...")
    attention_path = plot_grand_average_attention(
        fold_p2d_matrices,
        fold_d2p_matrices,
        args.output_dir,
        args.cold_start_mode,
    )
    print(f"Grand-average attention map saved to: {attention_path}")

    keys = ["mse", "rmse", "mae", "r2", "pearson", "spearman"]
    avg_metrics = {}
    for phase in ["train", "val", "test"]:
        for k in keys:
            avg_metrics[f"avg_{phase}_{k}"] = np.mean([m[f"{phase}_{k}"] for m in all_metrics])

    ts = time.strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(args.output_dir, f"metrics_{args.cold_start_mode}_{ts}.txt")

    print("\n" + "=" * 60)
    print("Fold-average metrics:")
    print(f"Train - RMSE: {avg_metrics['avg_train_rmse']:.4f}, R2: {avg_metrics['avg_train_r2']:.4f}, PCC: {avg_metrics['avg_train_pearson']:.4f}")
    print(f"Val   - RMSE: {avg_metrics['avg_val_rmse']:.4f}, R2: {avg_metrics['avg_val_r2']:.4f}, PCC: {avg_metrics['avg_val_pearson']:.4f}")
    print(f"Test  - RMSE: {avg_metrics['avg_test_rmse']:.4f}, R2: {avg_metrics['avg_test_r2']:.4f}, PCC: {avg_metrics['avg_test_pearson']:.4f}")
    print("-" * 60)
    print("Overall OOF metrics:")
    print(f"MSE: {overall['mse']:.4f}")
    print(f"RMSE: {overall['rmse']:.4f}")
    print(f"MAE: {overall['mae']:.4f}")
    print(f"R2: {overall['r2']:.4f}")
    print(f"Pearson: {overall['pearson']:.4f}")
    print(f"Spearman: {overall['spearman']:.4f}")
    print("=" * 60)

    with open(metrics_file, "w") as f:
        f.write(f"ESM2+Morgan Gated+CrossAtt+MoE - {args.cold_start_mode} Cold Start\n")
        f.write(f"Total Duration: {total_dur:.2f}s\n\n")
        header = ["Fold", "Duration"]
        for phase in ["train", "val", "test"]:
            for k in keys:
                header.append(f"{phase}_{k}")
        f.write("\t".join(header) + "\n")
        for m in all_metrics:
            row = [str(m["fold"]), f"{m['duration']:.2f}"]
            for phase in ["train", "val", "test"]:
                for k in keys:
                    row.append(f"{m[f'{phase}_{k}']:.6f}")
            f.write("\t".join(row) + "\n")
        f.write("\nAverage Metrics:\n")
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nOverall OOF Metrics:\n")
        for k, v in overall.items():
            f.write(f"{k}: {v:.6f}\n")

    pred_file = os.path.join(args.output_dir, f"oof_predictions_{args.cold_start_mode}_{ts}.txt")
    np.savetxt(
        pred_file,
        np.vstack([all_trues, all_preds]).T,
        header="True\tPred",
        delimiter="\t",
    )

    print(f"\nResults saved to: {args.output_dir}")
    return {
        "metrics_file": metrics_file,
        "prediction_file": pred_file,
        "output_dir": args.output_dir,
        "overall": overall,
        "average": avg_metrics,
    }
