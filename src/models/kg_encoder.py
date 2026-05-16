"""Per-article KG encoder.

Given a set of QIDs that appear in an article (plus their pre-trained
TransE embeddings), produce a single fixed-size vector for the article.

We use a tf-idf-weighted attention pool: each entity's weight is its
mention-frequency-times-idf in the article (or uniform if tf-idf is not
configured). This mirrors the "weighted average" the paper applies on
the text side.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from ..kg.transE import KGEmbeddings


class KGEncoder:
    def __init__(self, embeddings: KGEmbeddings):
        self.emb = embeddings

    @property
    def dim(self) -> int:
        return self.emb.dim

    def encode(
        self,
        entity_ids: Iterable[str],
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        ids = list(entity_ids)
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        counts = Counter(ids)
        vecs: list[np.ndarray] = []
        ws: list[float] = []
        for qid, c in counts.items():
            v = self.emb.get(qid)
            if not np.any(v):
                continue
            w = float(weights.get(qid, 1.0)) if weights else 1.0
            vecs.append(v)
            ws.append(w * c)
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        V = np.stack(vecs, axis=0)
        W = np.asarray(ws, dtype=np.float32)
        denom = float(W.sum()) or 1.0
        return (W[:, None] * V).sum(axis=0) / denom
