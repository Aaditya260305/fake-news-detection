"""Feature extraction pipeline.

For each article we produce three vectors:

* ``text_emb`` -- tf-idf weighted GloVe over (tokenised) sentences
* ``kg_emb``   -- mean of pre-trained TransE entity vectors
* ``label``    -- {0, 1}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .data import Article
from .preprocessing.tokenize_en import tokenize_article
from .preprocessing.tfidf import TfIdfWeights
from .models.text_encoder import GloveLoader, TextEncoder
from .models.kg_encoder import KGEncoder
from .kg.transE import KGEmbeddings


log = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    text_emb: np.ndarray
    kg_emb: np.ndarray
    labels: np.ndarray
    ids: list[str]


def build_features(
    articles: Iterable[Article],
    *,
    text_encoder: TextEncoder,
    kg_encoder: KGEncoder,
    article_to_entities: dict[str, list[str]] | None = None,
    spacy_model: str = "en_core_web_sm",
    max_sentences: int = 40,
    max_tokens_per_sentence: int = 64,
) -> FeatureSet:
    text_vs: list[np.ndarray] = []
    kg_vs: list[np.ndarray] = []
    ys: list[int] = []
    ids: list[str] = []
    a2e = article_to_entities or {}

    for art in tqdm(list(articles), desc="features"):
        sents = tokenize_article(
            art.text,
            spacy_model=spacy_model,
            max_sentences=max_sentences,
            max_tokens_per_sentence=max_tokens_per_sentence,
        )
        text_vs.append(text_encoder.encode_article(sents))
        kg_vs.append(kg_encoder.encode(a2e.get(art.id, [])))
        ys.append(int(art.label))
        ids.append(art.id)

    return FeatureSet(
        text_emb=np.stack(text_vs, axis=0),
        kg_emb=np.stack(kg_vs, axis=0),
        labels=np.asarray(ys, dtype=np.int64),
        ids=ids,
    )


def save_features(fs: FeatureSet, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        text_emb=fs.text_emb,
        kg_emb=fs.kg_emb,
        labels=fs.labels,
        ids=np.asarray(fs.ids),
    )


def load_features(path: str | Path) -> FeatureSet:
    data = np.load(path, allow_pickle=False)
    return FeatureSet(
        text_emb=data["text_emb"],
        kg_emb=data["kg_emb"],
        labels=data["labels"],
        ids=data["ids"].tolist(),
    )
