"""MinHash + LSH near-duplicate detection for the news corpus.

News datasets routinely contain near-duplicate articles: wire-service
reprints, slight edits, syndicated content. Training a model with
duplicates leaked from the train set into the val/test set inflates
F1 artificially. We detect duplicates with the standard locality-
sensitive-hashing pipeline:

1. **Shingle** each article into overlapping n-grams (size ``shingle``).
2. **MinHash** each shingle set: compute ``num_perm`` random hash
   functions, keep the *minimum* per hash function -- this gives a
   ``num_perm``-dimensional signature whose pairwise Hamming distance
   approximates 1 - Jaccard similarity of the original sets.
3. **LSH banding**: split each signature into ``bands`` chunks of
   ``rows`` rows. Two signatures with at least one identical chunk go
   into the same candidate bucket. Tune ``(bands, rows)`` so the
   probability of a collision for sims >= ``threshold`` is high.

This is exactly how Google News, arxiv-sanity, the Common Crawl WET
deduper, and Spark's ``BucketedRandomProjectionLSH`` find near-
duplicates at web scale.

Uses ``datasketch`` if installed (faster, vectorised); falls back to a
pure-Python implementation otherwise.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except Exception:
    _HAS_DATASKETCH = False


log = logging.getLogger(__name__)


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles of ``text``."""
    toks = [m.group(0).lower() for m in _TOKEN.finditer(text)]
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


# ----------------------------------------------------------------------
#  pure-Python MinHash fallback
# ----------------------------------------------------------------------

class _PyMinHash:
    """Compact MinHash using Python's built-in hash with seed mixing."""

    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.sig = [(1 << 63) - 1] * num_perm

    def update(self, item: str) -> None:
        for i in range(self.num_perm):
            h = (hash((i + 1, item)) ^ (i * 0x9E3779B97F4A7C15)) & ((1 << 63) - 1)
            if h < self.sig[i]:
                self.sig[i] = h

    def jaccard(self, other: "_PyMinHash") -> float:
        eq = sum(1 for a, b in zip(self.sig, other.sig) if a == b)
        return eq / float(self.num_perm)


def _build_minhash(text: str, *, num_perm: int, shingle: int):
    sh = _shingles(text, k=shingle)
    if _HAS_DATASKETCH:
        m = MinHash(num_perm=num_perm)
        for s in sh:
            m.update(s.encode("utf-8"))
        return m
    m = _PyMinHash(num_perm=num_perm)
    for s in sh:
        m.update(s)
    return m


# ----------------------------------------------------------------------
#  public API
# ----------------------------------------------------------------------

@dataclass
class DupReport:
    n_articles: int
    n_duplicate_pairs: int
    n_articles_in_duplicate_cluster: int
    dup_ratio: float
    clusters: list[list[str]]      # connected components of article ids
    threshold: float
    num_perm: int
    shingle: int
    backend: str                   # "datasketch" or "py"
    sample_pairs: list[tuple[str, str, float]]   # (id_a, id_b, est_jaccard)


def find_near_duplicates(
    articles: list,
    *,
    text_attr: str = "text",
    id_attr: str = "id",
    threshold: float = 0.85,
    num_perm: int = 128,
    shingle: int = 5,
    max_sample_pairs: int = 10,
) -> DupReport:
    """Return a :class:`DupReport` describing the duplicate clusters.

    Each ``articles[i]`` must have ``id`` and ``text`` attributes (the
    project's :class:`src.data.Article` dataclass satisfies this).
    """
    if not articles:
        return DupReport(0, 0, 0, 0.0, [], threshold, num_perm, shingle,
                         "datasketch" if _HAS_DATASKETCH else "py", [])

    backend = "datasketch" if _HAS_DATASKETCH else "py"
    log.info("MinHash dedup: n=%d  num_perm=%d  shingle=%d  threshold=%.2f  backend=%s",
             len(articles), num_perm, shingle, threshold, backend)

    sigs: dict[str, object] = {}
    for art in articles:
        text = getattr(art, text_attr, "") or ""
        sigs[getattr(art, id_attr)] = _build_minhash(text, num_perm=num_perm, shingle=shingle)

    candidate_pairs: set[tuple[str, str]] = set()
    if _HAS_DATASKETCH:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for aid, m in sigs.items():
            lsh.insert(aid, m)
        for aid, m in sigs.items():
            for other in lsh.query(m):
                if other != aid:
                    pair = tuple(sorted((aid, other)))
                    candidate_pairs.add(pair)
    else:
        # Pure-Python LSH banding.
        rows = max(1, int(round(num_perm / (1.0 - threshold) / 4)))
        bands = max(1, num_perm // rows)
        buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
        for aid, m in sigs.items():
            sig = m.sig  # type: ignore[attr-defined]
            for b in range(bands):
                chunk = tuple(sig[b * rows : (b + 1) * rows])
                buckets[(b, chunk)].append(aid)
        for ids in buckets.values():
            if len(ids) < 2:
                continue
            for i, a in enumerate(ids):
                for c in ids[i + 1 :]:
                    pair = tuple(sorted((a, c)))
                    candidate_pairs.add(pair)

    # Verify candidates with the actual MinHash estimate.
    pairs: list[tuple[str, str, float]] = []
    for a, b in candidate_pairs:
        ma, mb = sigs[a], sigs[b]
        if _HAS_DATASKETCH:
            j = float(ma.jaccard(mb))  # type: ignore[attr-defined]
        else:
            j = ma.jaccard(mb)  # type: ignore[attr-defined]
        if j >= threshold:
            pairs.append((a, b, j))

    # Connected components -> clusters.
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        _union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for n in parent:
        groups[_find(n)].append(n)
    clusters = [sorted(g) for g in groups.values() if len(g) >= 2]
    in_dup = sum(len(c) for c in clusters)
    pairs.sort(key=lambda p: -p[2])

    return DupReport(
        n_articles=len(articles),
        n_duplicate_pairs=len(pairs),
        n_articles_in_duplicate_cluster=in_dup,
        dup_ratio=in_dup / max(len(articles), 1),
        clusters=clusters,
        threshold=threshold,
        num_perm=num_perm,
        shingle=shingle,
        backend=backend,
        sample_pairs=pairs[:max_sample_pairs],
    )
