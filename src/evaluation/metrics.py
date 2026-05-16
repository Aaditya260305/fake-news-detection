"""Metrics for paper Sec. V-B.

* Precision, Recall, F1 -- standard sklearn definitions (fake == positive).
* Miss-rate == false-negative-rate == 1 - recall.
* Confusion matrix [TP, FP, FN, TN] in the paper's order.
* ROUGE-1/2/L (string-level) via ``rouge-score`` (with a minimal-
  dependency fallback).
"""
from __future__ import annotations

from typing import Iterable

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def classification_report(y_true: Iterable[int], y_pred: Iterable[int], y_prob: Iterable[float] | None = None) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f = float(f1_score(y_true, y_pred, zero_division=0))
    miss = 1.0 - r
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel().tolist()
    # sklearn returns [TP, FN, FP, TN] with labels=[1, 0] so re-order
    # to the paper's [TP, FP, FN, TN] convention:
    tp, fn, fp, tn = cm
    auc = None
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, list(y_prob)))
        except Exception:
            auc = None
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "miss_rate": miss,
        "confusion_matrix": [tp, fp, fn, tn],
        "auc": auc,
        "n": len(y_true),
    }


# -------- ROUGE ----------

def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Mean ROUGE-1, ROUGE-2, ROUGE-L F1 over a parallel corpus."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        agg: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref or "", pred or "")
            for k in agg:
                agg[k].append(float(s[k].fmeasure))
        return {k: float(sum(v) / max(len(v), 1)) for k, v in agg.items()}
    except Exception:
        return _rouge_fallback(predictions, references)


def _rouge_fallback(predictions, references) -> dict:
    """Minimal ROUGE-1/2/L F1 implementation when rouge_score is missing."""

    def _ngrams(toks, n):
        return list(zip(*[toks[i:] for i in range(n)]))

    def _lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, x in enumerate(a, 1):
            for j, y in enumerate(b, 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    def _f1(overlap, total_p, total_r):
        if not overlap or not total_p or not total_r:
            return 0.0
        p = overlap / total_p
        r = overlap / total_r
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        pt = (pred or "").lower().split()
        rt = (ref or "").lower().split()
        from collections import Counter

        c1 = Counter(pt) & Counter(rt)
        r1.append(_f1(sum(c1.values()), len(pt), len(rt)))
        c2 = Counter(_ngrams(pt, 2)) & Counter(_ngrams(rt, 2))
        r2.append(_f1(sum(c2.values()), max(len(pt) - 1, 0), max(len(rt) - 1, 0)))
        rl.append(_f1(_lcs(pt, rt), len(pt), len(rt)))

    return {
        "rouge1": sum(r1) / max(len(r1), 1),
        "rouge2": sum(r2) / max(len(r2), 1),
        "rougeL": sum(rl) / max(len(rl), 1),
    }
