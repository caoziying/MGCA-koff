from __future__ import annotations

import torch
import torch.nn as nn


class MaskGenerator(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.W = nn.Linear(proj_dim, 1)

    def forward(self, x):
        batch_size, n_experts, dim = x.size()
        x_flat = x.view(-1, dim)
        logits = self.W(x_flat).view(batch_size, n_experts, 1)
        mask = torch.softmax(logits, dim=1)
        return x * mask


class MultiExpertEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_experts=4):
        super().__init__()
        self.max_experts = max_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(max_experts)
            ]
        )

    def forward(self, x):
        batch_size, n_experts, _ = x.size()
        outs = []
        for i in range(n_experts):
            outs.append(self.experts[i](x[:, i, :]))
        return torch.stack(outs, dim=1)


class GatedExpertFusion(nn.Module):
    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(num_experts * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, experts_out):
        batch_size, n_experts, hidden_dim = experts_out.size()
        flat = experts_out.view(batch_size, n_experts * hidden_dim)
        weights = self.gate(flat)
        weights_exp = weights.unsqueeze(-1)
        fused = (experts_out * weights_exp).sum(dim=1)
        return fused, weights


class ExpertBiCrossAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        self.attn_p = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_d = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, prot_experts, drug_experts):
        out_p, w_p2d = self.attn_p(
            query=prot_experts,
            key=drug_experts,
            value=drug_experts,
            need_weights=True,
        )
        out_d, w_d2p = self.attn_d(
            query=drug_experts,
            key=prot_experts,
            value=prot_experts,
            need_weights=True,
        )

        x1 = out_p.mean(dim=1)
        x2 = out_d.mean(dim=1)
        return x1, x2, w_p2d, w_d2p


class MoEBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=4, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        gate_w = self.gate(x)
        expert_outs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outs, dim=1)
        gate_exp = gate_w.unsqueeze(-1)
        out = (expert_stack * gate_exp).sum(dim=1)
        return out, gate_w
