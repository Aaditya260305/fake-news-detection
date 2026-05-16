"""End-to-end EKNet wrapper.

This module wires together the **already-computed** text and KG
features and trains the MLP decision maker. The reason for this split:

* The text encoder is non-trainable (tf-idf weighted GloVe).
* The KG encoder is also non-trainable at this stage (it consumes
  pre-trained TransE embeddings).
* Only the decision maker has learnable parameters here.

This matches the paper's recipe ("text encoder", "KG encoder", "decision
maker" as three separable components) and keeps CPU training fast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .decision_maker import DecisionMaker


@dataclass
class EKNetFeatures:
    text_emb: np.ndarray            # [N, text_dim]
    kg_emb: np.ndarray              # [N, kg_dim]
    labels: np.ndarray              # [N]


class EKNet(nn.Module):
    """Top-level EKNet module exposing the three ablation modes."""

    MODES = ("text_only", "ontology_only", "both")

    def __init__(
        self,
        text_dim: int,
        kg_dim: int,
        mode: str = "both",
        hidden_dims=(128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        assert mode in self.MODES, f"Unknown mode {mode!r}"
        self.mode = mode
        self.text_dim = text_dim
        self.kg_dim = kg_dim
        self.decision = DecisionMaker(
            text_dim=text_dim,
            kg_dim=kg_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def _mask(self, text_emb: torch.Tensor, kg_emb: torch.Tensor):
        if self.mode == "text_only":
            kg_emb = torch.zeros_like(kg_emb)
        elif self.mode == "ontology_only":
            text_emb = torch.zeros_like(text_emb)
        return text_emb, kg_emb

    def forward(self, text_emb: torch.Tensor, kg_emb: torch.Tensor) -> torch.Tensor:
        t, k = self._mask(text_emb, kg_emb)
        return self.decision(t, k)

    def predict_proba(self, text_emb: torch.Tensor, kg_emb: torch.Tensor) -> torch.Tensor:
        logits = self.forward(text_emb, kg_emb)
        return torch.sigmoid(logits)
