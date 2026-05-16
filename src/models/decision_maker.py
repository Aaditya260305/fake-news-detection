"""MLP decision maker for EKNet (Sec. IV-A, "Decision maker")."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class DecisionMaker(nn.Module):
    """``[text_emb ; kg_emb]`` -> sigmoid(real/fake)."""

    def __init__(
        self,
        text_dim: int,
        kg_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = text_dim + kg_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, text_emb: torch.Tensor, kg_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_emb, kg_emb], dim=-1)
        return self.net(x).squeeze(-1)
