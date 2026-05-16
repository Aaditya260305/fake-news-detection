"""Small Transformer baseline (Vaswani et al. 2017) -- 2 layers, 4 heads."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from ._common import prepare_baseline_data, make_loaders_from_datasets, save_vocab, train_loop


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        max_len: int = 400,
        pretrained=None,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained is not None:
            self.embed.weight.data.copy_(torch.from_numpy(pretrained))
        self.pos = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=0.2, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x, lengths):
        N, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(N, L)
        h = self.embed(x) + self.pos(pos)
        key_padding_mask = x == 0
        h = self.enc(h, src_key_padding_mask=key_padding_mask)
        mask = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train(cfg, splits, outdir: Path) -> dict:
    max_len = int(getattr(cfg.baselines, "max_len", 256))
    vocab, pretrained, datasets = prepare_baseline_data(cfg, splits, max_len=max_len)
    model = SmallTransformer(
        vocab_size=len(vocab),
        embed_dim=cfg.text_encoder.glove_dim,
        max_len=max_len,
        pretrained=pretrained,
    )
    loaders = make_loaders_from_datasets(datasets, batch_size=cfg.baselines.batch_size)
    report, history = train_loop(cfg, model, loaders)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "transformer_small.pt")
    save_vocab(vocab, outdir / "transformer_small_vocab.json", max_len=max_len)
    with open(outdir / "transformer_small_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"model": "transformer_small", **report}
