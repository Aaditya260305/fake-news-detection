"""TransE knowledge graph embeddings.

We try to use **PyKEEN** for a fast, well-tested implementation. If
PyKEEN is unavailable in the runtime environment, a tiny PyTorch
TransE falls back transparently so the rest of the pipeline still runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


log = logging.getLogger(__name__)


@dataclass
class KGEmbeddings:
    entity_to_idx: dict[str, int]
    relation_to_idx: dict[str, int]
    entity_emb: np.ndarray            # [num_entities, dim]
    relation_emb: np.ndarray          # [num_relations, dim]

    @property
    def dim(self) -> int:
        return self.entity_emb.shape[1]

    def get(self, entity: str) -> np.ndarray:
        idx = self.entity_to_idx.get(entity)
        if idx is None:
            return np.zeros(self.dim, dtype=np.float32)
        return self.entity_emb[idx]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            entity_emb=self.entity_emb,
            relation_emb=self.relation_emb,
            entity_keys=np.asarray(list(self.entity_to_idx.keys())),
            relation_keys=np.asarray(list(self.relation_to_idx.keys())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "KGEmbeddings":
        data = np.load(path, allow_pickle=False)
        ekeys = data["entity_keys"].tolist()
        rkeys = data["relation_keys"].tolist()
        return cls(
            entity_to_idx={k: i for i, k in enumerate(ekeys)},
            relation_to_idx={k: i for i, k in enumerate(rkeys)},
            entity_emb=data["entity_emb"],
            relation_emb=data["relation_emb"],
        )


def _train_pykeen(
    triples: list[tuple[str, str, str]],
    dim: int,
    epochs: int,
    batch_size: int,
) -> KGEmbeddings:
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    arr = np.asarray(triples, dtype=str)
    tf = TriplesFactory.from_labeled_triples(arr)
    result = pipeline(
        training=tf,
        testing=tf,
        model="TransE",
        model_kwargs={"embedding_dim": dim},
        training_kwargs={"num_epochs": epochs, "batch_size": batch_size},
        random_seed=42,
    )
    entity_emb = result.model.entity_representations[0]().detach().cpu().numpy()
    relation_emb = result.model.relation_representations[0]().detach().cpu().numpy()
    return KGEmbeddings(
        entity_to_idx=dict(tf.entity_to_id),
        relation_to_idx=dict(tf.relation_to_id),
        entity_emb=entity_emb.astype(np.float32),
        relation_emb=relation_emb.astype(np.float32),
    )


def _train_native_pytorch(
    triples: list[tuple[str, str, str]],
    dim: int,
    epochs: int,
    batch_size: int,
    *,
    patience: int = 5,
    min_delta: float = 1e-3,
) -> KGEmbeddings:
    """Minimal-dependency fallback TransE in PyTorch.

    Adds early stopping (``patience`` epochs without ``min_delta``
    improvement on the EMA of the loss) and a tqdm progress bar so a
    long TransE run is never a black box.
    """
    import time as _time

    import torch
    import torch.nn as nn

    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except Exception:
        _HAS_TQDM = False

    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    e2i = {e: i for i, e in enumerate(entities)}
    r2i = {r: i for i, r in enumerate(relations)}

    heads = torch.tensor([e2i[h] for h, _, _ in triples], dtype=torch.long)
    rels = torch.tensor([r2i[r] for _, r, _ in triples], dtype=torch.long)
    tails = torch.tensor([e2i[t] for _, _, t in triples], dtype=torch.long)

    n_e = len(entities)
    n_r = len(relations)
    log.info(
        "TransE-native: %d triples, %d entities, %d relations, dim=%d, batches/ep=%d",
        len(triples), n_e, n_r, dim, max(1, (len(triples) + batch_size - 1) // batch_size),
    )
    ent_emb = nn.Embedding(n_e, dim)
    rel_emb = nn.Embedding(n_r, dim)
    nn.init.xavier_uniform_(ent_emb.weight)
    nn.init.xavier_uniform_(rel_emb.weight)

    opt = torch.optim.Adam(list(ent_emb.parameters()) + list(rel_emb.parameters()), lr=1e-3)
    n = len(triples)

    best_loss = float("inf")
    bad = 0
    ema_loss = None
    epoch_iter = (
        tqdm(range(epochs), desc="transE", unit="ep", leave=True) if _HAS_TQDM else range(epochs)
    )
    for ep in epoch_iter:
        t0 = _time.time()
        perm = torch.randperm(n)
        total = 0.0
        n_batches = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            h = ent_emb(heads[idx])
            r = rel_emb(rels[idx])
            t = ent_emb(tails[idx])
            neg_t_idx = torch.randint(0, n_e, idx.shape)
            t_neg = ent_emb(neg_t_idx)
            pos = (h + r - t).norm(p=2, dim=-1)
            neg = (h + r - t_neg).norm(p=2, dim=-1)
            loss = torch.relu(1.0 + pos - neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        epoch_loss = total / max(n_batches, 1)
        ema_loss = epoch_loss if ema_loss is None else 0.8 * ema_loss + 0.2 * epoch_loss
        elapsed = _time.time() - t0
        log.info(
            "transE-native ep %d/%d  loss=%.4f  ema=%.4f  (%.1fs)",
            ep + 1, epochs, epoch_loss, ema_loss, elapsed,
        )
        if _HAS_TQDM:
            epoch_iter.set_postfix(loss=f"{ema_loss:.4f}")

        if ema_loss < best_loss - min_delta:
            best_loss = ema_loss
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info(
                    "transE-native: early stop at ep %d (no improvement for %d epochs)",
                    ep + 1, patience,
                )
                if _HAS_TQDM:
                    epoch_iter.close()
                break

    return KGEmbeddings(
        entity_to_idx=e2i,
        relation_to_idx=r2i,
        entity_emb=ent_emb.weight.detach().cpu().numpy().astype(np.float32),
        relation_emb=rel_emb.weight.detach().cpu().numpy().astype(np.float32),
    )


def train_transE(
    triples: Iterable[tuple[str, str, str]],
    *,
    dim: int = 64,
    epochs: int = 30,
    batch_size: int = 256,
) -> KGEmbeddings:
    triples = list(triples)
    if not triples:
        raise ValueError("No triples to train on.")
    try:
        return _train_pykeen(triples, dim=dim, epochs=epochs, batch_size=batch_size)
    except Exception as e:
        log.warning("PyKEEN unavailable (%s) -- falling back to native PyTorch TransE.", e)
        return _train_native_pytorch(triples, dim=dim, epochs=epochs, batch_size=batch_size)
