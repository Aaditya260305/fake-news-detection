"""Fit / load a tf-idf vectorizer on the training corpus.

The paper's Sec. IV-A specifies: "tf-idf method is employed to compute
the weight of each word in every sentence ... weighted average of word
embeddings ..." — so we keep both the *vocabulary* idf weights and a
helper that returns per-token weights for any new text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfWeights:
    """Wraps a fitted TfidfVectorizer for per-token weight lookups."""

    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer
        self._idf: Mapping[str, float] = {
            w: float(vectorizer.idf_[i]) for w, i in vectorizer.vocabulary_.items()
        }
        self._mean_idf = float(np.mean(list(self._idf.values()))) if self._idf else 1.0

    def weight(self, token: str) -> float:
        return self._idf.get(token.lower(), self._mean_idf)

    def weights_for(self, tokens: Iterable[str]) -> np.ndarray:
        return np.asarray([self.weight(t) for t in tokens], dtype=np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str | Path) -> "TfIdfWeights":
        vec = joblib.load(path)
        return cls(vec)


def fit_tfidf(
    documents: Iterable[str],
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = 50_000,
) -> TfIdfWeights:
    vec = TfidfVectorizer(
        lowercase=True,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-']+\b",
    )
    vec.fit(list(documents))
    return TfIdfWeights(vec)
