from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    ExpertBiCrossAttention,
    GatedExpertFusion,
    MaskGenerator,
    MoEBlock,
    MultiExpertEncoder,
)


class FullRegressionTransformer(nn.Module):
    def __init__(
        self,
        proj_dim1=2560,
        proj_dim2=2048,
        hidden_dim=512,
        dropout=0.1,
        nums_of_experts=4,
        num_heads=8,
        moe_num_experts=4,
        ablation="no",
    ):
        super().__init__()
        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.n_experts = nums_of_experts

        self.mask1 = MaskGenerator(proj_dim=proj_dim1)
        self.mask2 = MaskGenerator(proj_dim=proj_dim2)

        self.protein_expert_encoder = MultiExpertEncoder(
            in_dim=proj_dim1,
            hidden_dim=hidden_dim,
            max_experts=nums_of_experts,
        )
        self.drug_expert_encoder = MultiExpertEncoder(
            in_dim=proj_dim2,
            hidden_dim=hidden_dim,
            max_experts=nums_of_experts,
        )

        self.prot_gate = GatedExpertFusion(num_experts=nums_of_experts, hidden_dim=hidden_dim)
        self.drug_gate = GatedExpertFusion(num_experts=nums_of_experts, hidden_dim=hidden_dim)

        self.expert_cross_att = ExpertBiCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.moe = MoEBlock(
            in_dim=hidden_dim * 3 if ablation == "no" else hidden_dim * 2,
            out_dim=hidden_dim,
            num_experts=moe_num_experts,
            hidden_dim=hidden_dim * 2,
            dropout=dropout,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input1, input2):
        # input1: [B, 4, 2560], ESM2 experts.
        # input2: [B, 4, 2048], Morgan r=0..3 experts.

        # Kept disabled to match the original model.py behavior.
        # input1 = self.mask1(input1)
        # input2 = self.mask2(input2)

        prot_experts = self.protein_expert_encoder(input1)
        drug_experts = self.drug_expert_encoder(input2)

        prot_fused, prot_weights = self.prot_gate(prot_experts)
        drug_fused, drug_weights = self.drug_gate(drug_experts)

        x1, x2, w_p2d, w_d2p = self.expert_cross_att(prot_experts, drug_experts)
        cross_feat = torch.cat([x1, x2], dim=-1)
        cross_fused = self.cross_proj(cross_feat)

        combined = None
        if self.ablation == "no":
            combined = torch.cat([prot_fused, drug_fused, cross_fused], dim=-1)
        elif self.ablation == "drug":
            combined = torch.cat([prot_fused, cross_fused], dim=-1)
        elif self.ablation == "target":
            combined = torch.cat([drug_fused, cross_fused], dim=-1)
        elif self.ablation == "bicross":
            combined = torch.cat([prot_fused, drug_fused], dim=-1)

        moe_out, moe_weights = self.moe(combined)
        out = self.regressor(moe_out)
        return out, (prot_weights, drug_weights, w_p2d, w_d2p)
