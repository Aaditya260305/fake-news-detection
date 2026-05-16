"""Dataset loaders + stratified splits for both Kaggle datasets.

The original paper's Sec. V-A-3/4 references these datasets exactly:
- "Real or Fake" (rchitic17): 6 060 articles, 50/50 balanced.
- "Fake News Detection" (jruvika): 3 988 articles.

We unify them under a common schema:
    {id: str, title: str, text: str, label: int}   (1 == fake, 0 == real)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Article:
    id: str
    title: str
    text: str
    label: int  # 1 == FAKE, 0 == REAL


def _to_label(value, pos_label) -> int:
    """Normalise heterogeneous label encodings to {0, 1}."""
    if isinstance(value, str):
        return 1 if value.strip().upper() == str(pos_label).strip().upper() else 0
    # numeric labels (jruvika dataset uses 0/1 directly)
    return 1 if int(value) == int(pos_label) else 0


def load_real_or_fake(cfg) -> list[Article]:
    csv = Path(cfg.dataset.real_or_fake.csv)
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run scripts/download_data.py or drop the file manually."
        )
    df = pd.read_csv(csv)
    pos = cfg.dataset.real_or_fake.pos_label
    out: list[Article] = []
    for i, row in df.iterrows():
        out.append(
            Article(
                id=str(row.get("id", i)),
                title=str(row.get("title", "")),
                text=str(row.get("text", "")),
                label=_to_label(row["label"], pos),
            )
        )
    return out


def load_fake_news_detection(cfg) -> list[Article]:
    csv = Path(cfg.dataset.fake_news_detection.csv)
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run scripts/download_data.py or drop the file manually."
        )
    df = pd.read_csv(csv)
    pos = cfg.dataset.fake_news_detection.pos_label
    out: list[Article] = []
    for i, row in df.iterrows():
        out.append(
            Article(
                id=str(i),
                title=str(row.get("Headline", "")),
                text=str(row.get("Body", "")),
                label=_to_label(row["Label"], pos),
            )
        )
    return out


def stratified_split(
    items: list[Article],
    train: float,
    val: float,
    test: float,
    seed: int = 42,
) -> tuple[list[Article], list[Article], list[Article]]:
    assert abs(train + val + test - 1.0) < 1e-6
    labels = [a.label for a in items]
    rest, test_set = train_test_split(
        items, test_size=test, stratify=labels, random_state=seed
    )
    rest_labels = [a.label for a in rest]
    val_ratio = val / (train + val)
    train_set, val_set = train_test_split(
        rest, test_size=val_ratio, stratify=rest_labels, random_state=seed
    )
    return train_set, val_set, test_set


def articles_to_dataframe(items: Iterable[Article]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": a.id, "title": a.title, "text": a.text, "label": a.label}
            for a in items
        ]
    )
