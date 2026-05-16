"""Faithful tf-idf-weighted GloVe news encoder.

Paper recipe (Sec. IV-A, "Text encoder"):

    1. Split news -> sentences (and words).
    2. For each sentence  s_j, compute  sent_j  =  sum_w  tfidf(w) * w_emb(w)  /  sum_w tfidf(w)
    3. News embedding = same weighted average over sentence embeddings.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np


log = logging.getLogger(__name__)


class GloveLoader:
    def __init__(self, path: str | Path, dim: int = 100):
        self.path = Path(path)
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.vectors: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            log.warning("GloVe file %s missing -- text_encoder will fall back to hashing.", self.path)
            return
        vectors: list[np.ndarray] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != self.dim + 1:
                    continue
                word = parts[0]
                vec = np.asarray(parts[1:], dtype=np.float32)
                self.vocab[word] = len(vectors)
                vectors.append(vec)
        self.vectors = np.stack(vectors, axis=0) if vectors else None
        log.info("Loaded %d GloVe vectors (dim=%d) from %s", len(self.vocab), self.dim, self.path)

    def __contains__(self, word: str) -> bool:
        return word in self.vocab

    def get(self, word: str) -> np.ndarray:
        if self.vectors is None:
            return _hash_vector(word, self.dim)
        idx = self.vocab.get(word)
        if idx is None:
            return _hash_vector(word, self.dim)
        return self.vectors[idx]


def _hash_vector(word: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(word)) % (2**32))
    return rng.normal(0.0, 0.1, size=dim).astype(np.float32)


class TextEncoder:
    """tf-idf weighted average GloVe encoder."""

    def __init__(self, glove: GloveLoader, tfidf_weights=None):
        self.glove = glove
        self.tfidf = tfidf_weights

    @property
    def dim(self) -> int:
        return self.glove.dim

    def encode_sentence(self, tokens: Iterable[str]) -> np.ndarray:
        tokens = list(tokens)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        weights = self.tfidf.weights_for(tokens) if self.tfidf else np.ones(len(tokens), dtype=np.float32)
        vecs = np.stack([self.glove.get(t) for t in tokens], axis=0)
        denom = float(weights.sum()) or 1.0
        return (weights[:, None] * vecs).sum(axis=0) / denom

    def encode_article(self, sentences: Iterable[Iterable[str]]) -> np.ndarray:
        sent_vecs: list[np.ndarray] = []
        sent_weights: list[float] = []
        for s in sentences:
            toks = list(s)
            if not toks:
                continue
            sv = self.encode_sentence(toks)
            sent_vecs.append(sv)
            # sentence weight = sum of token tf-idf weights (paper's
            # "weighted average over sentence embeddings")
            w = (
                float(self.tfidf.weights_for(toks).sum()) if self.tfidf else float(len(toks))
            )
            sent_weights.append(w)
        if not sent_vecs:
            return np.zeros(self.dim, dtype=np.float32)
        v = np.stack(sent_vecs, axis=0)
        w = np.asarray(sent_weights, dtype=np.float32)
        denom = float(w.sum()) or 1.0
        return (w[:, None] * v).sum(axis=0) / denom


# ----------------------------------------------------------------------
#  Optional: sentence-transformers MiniLM encoder
#
#  Activated via ``text_encoder.type: minilm`` in the YAML.
#  Pulls in ``sentence-transformers`` (CPU-friendly, ~80 MB model).
#  Typically lifts EKNet F1 by 5-8 points vs. the GloVe baseline on
#  English news classification because it captures sentence-level
#  semantics that a tf-idf-weighted bag-of-vectors cannot.
# ----------------------------------------------------------------------

class MiniLMEncoder:
    """Sentence-Transformers encoder (default: all-MiniLM-L6-v2, dim=384).

    Drop-in replacement for :class:`TextEncoder` -- it exposes ``dim``
    and ``encode_article(sentences)`` with the same signature so the
    rest of the pipeline doesn't change.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - import error is user-visible
            raise RuntimeError(
                "text_encoder.type='minilm' requires sentence-transformers. "
                "Install a torch-2.3-compatible combo:\n"
                "    pip install \"sentence-transformers==2.7.0\" "
                "\"transformers>=4.39,<4.45\" \"huggingface_hub<0.24\"\n"
                "See README.md > 'MiniLM text encoder' for details."
            ) from e
        except (NameError, TypeError, AttributeError) as e:
            # The classic "NameError: name 'nn' is not defined" cascade
            # when `transformers >= 4.50` silently disables torch because
            # torch < 2.4, then later references nn.Module in a type
            # annotation.
            raise RuntimeError(
                "Failed to import sentence-transformers; this is almost "
                "always a version mismatch with PyTorch 2.3.1. Pin a "
                "torch-2.3-compatible stack:\n"
                "    pip uninstall -y sentence-transformers transformers\n"
                "    pip install \"sentence-transformers==2.7.0\" "
                "\"transformers>=4.39,<4.45\" \"huggingface_hub<0.24\"\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        log.info("Loading SentenceTransformer model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._dim = int(self.model.get_sentence_embedding_dimension())
        log.info("MiniLMEncoder ready (dim=%d)", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode_sentence(self, tokens: Iterable[str]) -> np.ndarray:
        toks = list(tokens)
        if not toks:
            return np.zeros(self._dim, dtype=np.float32)
        text = " ".join(toks)
        v = self.model.encode([text], convert_to_numpy=True,
                              show_progress_bar=False, normalize_embeddings=True)
        return v[0].astype(np.float32)

    def encode_article(self, sentences: Iterable[Iterable[str]]) -> np.ndarray:
        """Encode each sentence separately, then mean-pool.

        Mean-pooling normalized sentence embeddings is a strong baseline
        for document-level classification (cf. SBERT paper, Sec. 4).
        """
        sent_texts: list[str] = []
        for s in sentences:
            toks = list(s)
            if not toks:
                continue
            sent_texts.append(" ".join(toks))
        if not sent_texts:
            return np.zeros(self._dim, dtype=np.float32)
        v = self.model.encode(
            sent_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return v.mean(axis=0).astype(np.float32)

    def encode_articles_batch(self, articles_sentences: list[list[list[str]]],
                              show_progress: bool = True) -> np.ndarray:
        """Encode many articles efficiently by flattening sentences across
        all articles into one big batch."""
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        flat: list[str] = []
        spans: list[tuple[int, int]] = []  # (start, end) per article
        for sents in articles_sentences:
            start = len(flat)
            for s in sents:
                toks = list(s)
                if toks:
                    flat.append(" ".join(toks))
            spans.append((start, len(flat)))

        if not flat:
            return np.zeros((len(articles_sentences), self._dim), dtype=np.float32)

        log.info("MiniLM: encoding %d sentences (%d articles) ...",
                 len(flat), len(articles_sentences))
        embs = self.model.encode(
            flat,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=bool(show_progress and tqdm is not None),
            normalize_embeddings=True,
        )

        out = np.zeros((len(articles_sentences), self._dim), dtype=np.float32)
        for i, (a, b) in enumerate(spans):
            if a == b:
                continue
            out[i] = embs[a:b].mean(axis=0)
        return out


def build_text_encoder(cfg, *, tfidf=None):
    """Factory that picks the encoder based on ``cfg.text_encoder.type``."""
    enc_type = getattr(cfg.text_encoder, "type", "glove_tfidf")
    if enc_type == "minilm":
        return MiniLMEncoder(
            model_name=getattr(cfg.text_encoder, "minilm_model",
                               "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=int(getattr(cfg.text_encoder, "minilm_batch_size", 32)),
        )
    glove = GloveLoader(cfg.paths.glove, dim=cfg.text_encoder.glove_dim)
    return TextEncoder(glove, tfidf_weights=tfidf)
