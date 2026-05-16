"""TextRCNN baseline (Lai et al. 2015) -- BiLSTM + Conv1D max-pool."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from ._common import prepare_baseline_data, make_loaders_from_datasets, save_vocab, train_loop


class TextRCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pretrained=None, padding_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained is not None:
            self.embed.weight.data.copy_(torch.from_numpy(pretrained))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(2 * hidden_dim + embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        cat = torch.cat([out, emb], dim=-1)
        conv_in = cat.transpose(1, 2)
        conv_out = torch.tanh(self.conv(conv_in))
        pooled = conv_out.max(dim=-1).values
        return self.head(self.dropout(pooled)).squeeze(-1)


def train(cfg, splits, outdir: Path) -> dict:
    max_len = int(getattr(cfg.baselines, "max_len", 256))
    vocab, pretrained, datasets = prepare_baseline_data(cfg, splits, max_len=max_len)
    model = TextRCNN(
        vocab_size=len(vocab),
        embed_dim=cfg.text_encoder.glove_dim,
        hidden_dim=cfg.baselines.hidden_dim,
        pretrained=pretrained,
    )
    loaders = make_loaders_from_datasets(datasets, batch_size=cfg.baselines.batch_size)
    report, history = train_loop(cfg, model, loaders)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "textrcnn.pt")
    save_vocab(vocab, outdir / "textrcnn_vocab.json", max_len=max_len)
    with open(outdir / "textrcnn_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return {"model": "textrcnn", **report}
