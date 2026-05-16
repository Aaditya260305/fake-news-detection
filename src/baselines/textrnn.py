"""TextRNN baseline (Lai et al. 2015) -- BiLSTM over word embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from ._common import prepare_baseline_data, make_loaders_from_datasets, save_vocab, train_loop


class TextRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pretrained=None, padding_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained is not None:
            self.embed.weight.data.copy_(torch.from_numpy(pretrained))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(self.dropout(pooled)).squeeze(-1)


def train(cfg, splits, outdir: Path) -> dict:
    max_len = int(getattr(cfg.baselines, "max_len", 256))
    vocab, pretrained, datasets = prepare_baseline_data(cfg, splits, max_len=max_len)
    model = TextRNN(
        vocab_size=len(vocab),
        embed_dim=cfg.text_encoder.glove_dim,
        hidden_dim=cfg.baselines.hidden_dim,
        pretrained=pretrained,
    )
    loaders = make_loaders_from_datasets(datasets, batch_size=cfg.baselines.batch_size)
    report, history = train_loop(cfg, model, loaders)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "textrnn.pt")
    save_vocab(vocab, outdir / "textrnn_vocab.json", max_len=max_len)
    with open(outdir / "textrnn_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"model": "textrnn", **report}
