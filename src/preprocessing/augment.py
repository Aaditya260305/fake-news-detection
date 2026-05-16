"""Data enhancement (paper Sec. IV-C-1, Algorithm 1).

Three techniques are implemented:

1. **Synonym replacement** — WordNet replaces low-tf-idf words so the
   "topic words" of an article stay intact (paper recommends ranking by
   tf-idf and replacing only the bottom of the list).
2. **Coverage mechanism** — when a generated word would create a long
   exact-bigram repeat, we suppress it (this is the same "CVG" the
   paper applies in the comment-generator decoder; we expose it here so
   sample augmentation cannot produce repetitive nonsense).
3. **Self-service sample generation** — once an EKNet checkpoint
   exists, the trainer can call back into this module to produce
   paraphrased *labelled* samples and re-train. We expose the hook
   (``self_service_generate``) but leave it disabled by default.
"""
from __future__ import annotations

import random
from typing import Iterable

try:
    import nltk
    from nltk.corpus import wordnet as wn
    _HAS_WN = True
except Exception:
    _HAS_WN = False


def _ensure_wordnet() -> bool:
    if not _HAS_WN:
        return False
    try:
        wn.synsets("test")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            return False
    return True


def _synonyms(word: str) -> list[str]:
    if not _ensure_wordnet():
        return []
    found: set[str] = set()
    for syn in wn.synsets(word):
        for lem in syn.lemma_names():
            if lem.lower() != word.lower() and "_" not in lem:
                found.add(lem.lower())
    return sorted(found)


def synonym_replace(
    tokens: list[str],
    *,
    keep_topic_words: set[str],
    prob: float = 0.1,
    seed: int | None = None,
) -> list[str]:
    """Replace ~``prob`` fraction of non-topic tokens with WordNet synonyms."""
    rng = random.Random(seed)
    out: list[str] = []
    for t in tokens:
        if t.lower() in keep_topic_words:
            out.append(t)
            continue
        if rng.random() < prob:
            cands = _synonyms(t)
            if cands:
                out.append(rng.choice(cands))
                continue
        out.append(t)
    return out


def coverage_filter(tokens: list[str], *, threshold: float = 0.6) -> list[str]:
    """Suppress repeated bigrams whose density exceeds ``threshold``.

    A simple coverage analogue: walk tokens left-to-right, count how
    many bigrams we have already emitted, and drop a token whose
    appended bigram would make the running bigram-repeat ratio exceed
    ``threshold``.
    """
    out: list[str] = []
    seen: dict[tuple[str, str], int] = {}
    repeats = 0
    for tok in tokens:
        if out:
            bg = (out[-1], tok)
            seen[bg] = seen.get(bg, 0) + 1
            if seen[bg] > 1:
                repeats += 1
            if out and (repeats / max(len(out), 1)) > threshold:
                continue
        out.append(tok)
    return out


def augment_one(
    tokens: list[str],
    *,
    topic_words: Iterable[str],
    prob: float = 0.1,
    threshold: float = 0.6,
    seed: int | None = None,
) -> list[str]:
    keep = {w.lower() for w in topic_words}
    replaced = synonym_replace(tokens, keep_topic_words=keep, prob=prob, seed=seed)
    return coverage_filter(replaced, threshold=threshold)


def self_service_generate(model, texts: Iterable[str]) -> list[str]:
    """Hook for paper's "self-service sample generation".

    Calls ``model.paraphrase(text)`` on each input. Most baselines do
    not support paraphrasing, so this is a no-op fallback that returns
    the texts unchanged. Users who plug in a Seq2Seq comment generator
    can override the hook to produce labelled augmented data.
    """
    if not hasattr(model, "paraphrase"):
        return list(texts)
    return [model.paraphrase(t) for t in texts]
