"""OntoEA-style membership loss.

OntoEA (Xiang et al., ACL'21) jointly trains:

* L_E (entity embedding objective, e.g. TransE)
* L_M (membership loss: ||entity - class_centroid||)
* L_C (ontology embedding objective)

We borrow only L_M -- it is the cheapest way to push an entity vector
toward its ontology-class centroid and reproduce the "ontology-guided"
flavour of the paper without re-implementing all of OntoEA.

There are two ways to apply it:

1. **Post-hoc** (default): after TransE finishes, we compute per-class
   centroids and run a few epochs of pure-membership refinement on the
   entity matrix. Cheap, no need to re-implement TransE training.

2. **Joint** (optional): users with a PyTorch training loop can call
   ``membership_loss(...)`` per batch and add it to L_E with a weight.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np


def class_centroids(
    entity_emb: np.ndarray,
    entity_to_idx: Mapping[str, int],
    crossview: Mapping[str, str],
) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for ent, cls in crossview.items():
        i = entity_to_idx.get(ent)
        if i is None:
            continue
        buckets[cls].append(entity_emb[i])
    return {
        cls: np.mean(np.stack(vs, axis=0), axis=0).astype(np.float32)
        for cls, vs in buckets.items()
        if vs
    }


def refine_with_membership(
    entity_emb: np.ndarray,
    entity_to_idx: Mapping[str, int],
    crossview: Mapping[str, str],
    *,
    weight: float = 0.5,
    iterations: int = 5,
) -> np.ndarray:
    """Pull every entity vector ``weight`` of the way toward its class centroid."""
    emb = entity_emb.copy().astype(np.float32)
    for _ in range(iterations):
        cents = class_centroids(emb, entity_to_idx, crossview)
        for ent, cls in crossview.items():
            i = entity_to_idx.get(ent)
            c = cents.get(cls)
            if i is None or c is None:
                continue
            emb[i] = (1.0 - weight) * emb[i] + weight * c
    return emb


def membership_loss_torch(
    entity_emb,                  # torch.nn.Parameter [N, D]
    entity_to_idx: Mapping[str, int],
    crossview: Mapping[str, str],
):
    """Squared-distance membership loss usable inside a PyTorch loop."""
    import torch

    by_class: dict[str, list[int]] = defaultdict(list)
    for ent, cls in crossview.items():
        i = entity_to_idx.get(ent)
        if i is not None:
            by_class[cls].append(i)

    total = entity_emb.new_zeros(())
    count = 0
    for cls, idxs in by_class.items():
        vecs = entity_emb[idxs]
        cent = vecs.mean(dim=0, keepdim=True).detach()
        total = total + ((vecs - cent) ** 2).sum(dim=-1).mean()
        count += 1
    return total / max(count, 1)
