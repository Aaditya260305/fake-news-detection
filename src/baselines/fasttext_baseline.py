"""FastText baseline (Joulin et al. 2017).

Tries the official ``fasttext`` Python bindings first; if they are
unavailable (common on Windows), we fall back to scikit-learn's
``HashingVectorizer`` + ``LogisticRegression``. The fallback preserves
the *spirit* of FastText (bag-of-features + linear classifier with
n-grams) so the Table I comparison still makes sense.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..evaluation.metrics import classification_report


log = logging.getLogger(__name__)


def _train_native_fasttext(cfg, splits, outdir: Path) -> dict:
    import fasttext

    train_articles, val_articles, test_articles = splits
    outdir.mkdir(parents=True, exist_ok=True)
    tr_path = outdir / "fasttext_train.txt"
    with open(tr_path, "w", encoding="utf-8") as f:
        for a in train_articles + val_articles:
            label = "__label__fake" if a.label == 1 else "__label__real"
            text = " ".join(a.text.split())
            f.write(f"{label} {text}\n")

    model = fasttext.train_supervised(
        input=str(tr_path),
        wordNgrams=2,
        epoch=10,
        lr=0.5,
        dim=cfg.text_encoder.glove_dim,
        loss="softmax",
    )
    model.save_model(str(outdir / "fasttext.bin"))

    y_true, y_pred, y_prob = [], [], []
    for a in test_articles:
        text = " ".join(a.text.split())
        labels, probs = model.predict(text, k=2)
        prob_fake = 0.0
        for lbl, p in zip(labels, probs):
            if lbl == "__label__fake":
                prob_fake = float(p)
                break
        y_prob.append(prob_fake)
        y_pred.append(1 if prob_fake >= 0.5 else 0)
        y_true.append(int(a.label))
    return {"model": "fasttext", **classification_report(y_true, y_pred, y_prob)}


def _train_fallback(cfg, splits, outdir: Path) -> dict:
    import pickle

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train_articles, val_articles, test_articles = splits
    tr_x = [a.text for a in train_articles + val_articles]
    tr_y = [a.label for a in train_articles + val_articles]
    pipe = Pipeline(
        [
            ("hash", HashingVectorizer(ngram_range=(1, 2), n_features=2**18, alternate_sign=False)),
            ("clf", LogisticRegression(max_iter=200, n_jobs=-1)),
        ]
    )
    pipe.fit(tr_x, tr_y)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "fasttext.pkl", "wb") as f:
        pickle.dump(pipe, f)
    test_x = [a.text for a in test_articles]
    test_y = [a.label for a in test_articles]
    prob = pipe.predict_proba(test_x)[:, 1].tolist()
    pred = [1 if p >= 0.5 else 0 for p in prob]
    return {"model": "fasttext", **classification_report(test_y, pred, prob)}


def train(cfg, splits, outdir: Path) -> dict:
    try:
        return _train_native_fasttext(cfg, splits, outdir)
    except Exception as e:
        log.warning("fasttext library unavailable (%s) -- falling back to HashingVectorizer+LR", e)
        return _train_fallback(cfg, splits, outdir)
