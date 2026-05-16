"""FastTextRank keyword extraction (Sec. IV-C-1, Fig. 4 of the paper).

Implements a co-occurrence-window TextRank over the words of a document,
biased by an external tf-idf score so the algorithm converges quickly
(this is the "Fast" part: the tf-idf prior is what lets us stop after
~10 power iterations instead of waiting for full PageRank convergence).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .tokenize_en import tokenize_words


def _build_graph(tokens: list[str], window: int) -> dict[str, dict[str, float]]:
    g: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    n = len(tokens)
    for i, w in enumerate(tokens):
        for j in range(i + 1, min(i + window + 1, n)):
            v = tokens[j]
            if v == w:
                continue
            g[w][v] += 1.0
            g[v][w] += 1.0
    return g


def textrank_keywords(
    text: str,
    *,
    window: int = 4,
    top_k: int = 15,
    iterations: int = 20,
    damping: float = 0.85,
    tfidf_prior: dict[str, float] | None = None,
    spacy_model: str = "en_core_web_sm",
) -> list[tuple[str, float]]:
    """Return ``[(keyword, score), ...]`` ranked by FastTextRank.

    Parameters
    ----------
    tfidf_prior
        Optional mapping ``token -> idf`` that biases convergence. When
        supplied, node masses are initialised to the tf-idf weight
        instead of uniform — this is the speedup the paper alludes to.
    """
    tokens = tokenize_words(text, spacy_model=spacy_model)
    if not tokens:
        return []

    graph = _build_graph(tokens, window=window)
    nodes = list(graph.keys())
    if not nodes:
        return []

    if tfidf_prior:
        scores = np.asarray([tfidf_prior.get(n, 1.0) for n in nodes], dtype=np.float32)
        scores /= scores.sum() or 1.0
    else:
        scores = np.full(len(nodes), 1.0 / len(nodes), dtype=np.float32)

    idx = {n: i for i, n in enumerate(nodes)}
    out_sum = {n: sum(graph[n].values()) or 1.0 for n in nodes}

    for _ in range(iterations):
        new = np.full_like(scores, (1.0 - damping) / len(nodes))
        for n in nodes:
            i = idx[n]
            contrib = 0.0
            for nb, w in graph[n].items():
                j = idx[nb]
                contrib += w * scores[j] / out_sum[nb]
            new[i] += damping * contrib
        if np.allclose(new, scores, atol=1e-6):
            scores = new
            break
        scores = new

    pairs = sorted(zip(nodes, scores.tolist()), key=lambda x: -x[1])
    return pairs[:top_k]
