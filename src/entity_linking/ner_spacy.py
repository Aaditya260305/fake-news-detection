"""spaCy-backed Named Entity Recognition.

Returns ``Mention`` records of (text, label, start, end) for downstream
linking. The model defaults to ``en_core_web_sm``; ``en_core_web_md`` is
better but heavier -- switch via ``configs/default.yaml``.

Performance notes
-----------------
* ``_load()`` returns a MINIMAL NER pipeline (tokenizer + tok2vec + NER
  only -- no parser, no tagger, no lemmatizer, no attribute_ruler).
  That is roughly 3-5x faster than the default load for our use case.
* ``extract_mentions_batch()`` runs ``nlp.pipe()`` over many texts at
  once, which is another 2-3x speedup on top of the pipeline trim.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mention:
    text: str
    label: str          # spaCy NER label, e.g. PERSON, ORG, GPE
    start: int
    end: int


# crude PERSON/ORG fallback for when spaCy is unavailable: capitalised n-grams
_CAP_RE = re.compile(r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})\b")
_STOP = {"The", "A", "An", "On", "In", "Of", "At", "And", "To"}
_MAX_TEXT_CHARS = 1_000_000


@lru_cache(maxsize=1)
def _load(model: str):
    """Load a *minimal* spaCy pipeline for NER.

    We keep only the components NER actually depends on:
        tok2vec -> ner
    Everything else (parser, tagger, attribute_ruler, lemmatizer,
    senter) is disabled because we don't need it for entity extraction.
    """
    if not _HAS_SPACY:
        return None
    try:
        nlp = spacy.load(
            model,
            disable=["lemmatizer", "tagger", "attribute_ruler", "parser", "senter"],
        )
        log.debug("Loaded minimal NER spaCy pipeline: %s", nlp.pipe_names)
        return nlp
    except Exception as e:
        log.warning("spaCy NER load failed (%s); falling back to regex.", e)
        return None


# ----------------------------------------------------------------------
#  fallback (no spaCy installed)
# ----------------------------------------------------------------------

def _fallback_extract(text: str) -> list[Mention]:
    out: list[Mention] = []
    for m in _CAP_RE.finditer(text):
        chunk = m.group(0)
        if chunk.split()[0] in _STOP:
            continue
        out.append(Mention(text=chunk, label="MISC", start=m.start(), end=m.end()))
    return out


# ----------------------------------------------------------------------
#  per-text + batch entry points
# ----------------------------------------------------------------------

def _doc_to_mentions(
    doc,
    *,
    keep_labels: set[str] | None,
    max_mentions: int | None,
) -> list[Mention]:
    out = [
        Mention(text=e.text, label=e.label_, start=e.start_char, end=e.end_char)
        for e in doc.ents
    ]
    if keep_labels is not None:
        out = [m for m in out if m.label in keep_labels]
    seen: set[str] = set()
    uniq: list[Mention] = []
    for m in out:
        key = m.text.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    if max_mentions is not None:
        uniq = uniq[:max_mentions]
    return uniq


def _post(
    mentions: list[Mention],
    keep_labels: set[str] | None,
    max_mentions: int | None,
) -> list[Mention]:
    if keep_labels is not None:
        mentions = [m for m in mentions if m.label in keep_labels]
    seen: set[str] = set()
    uniq: list[Mention] = []
    for m in mentions:
        key = m.text.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    if max_mentions is not None:
        uniq = uniq[:max_mentions]
    return uniq


def extract_mentions(
    text: str,
    model: str = "en_core_web_sm",
    keep_labels: Iterable[str] | None = None,
    max_mentions: int | None = None,
) -> list[Mention]:
    nlp = _load(model)
    keep = set(keep_labels) if keep_labels else None
    if nlp is None:
        return _post(_fallback_extract(text), keep, max_mentions)
    doc = nlp(text[:_MAX_TEXT_CHARS])
    return _doc_to_mentions(doc, keep_labels=keep, max_mentions=max_mentions)


def extract_mentions_batch(
    texts: Sequence[str],
    model: str = "en_core_web_sm",
    keep_labels: Iterable[str] | None = None,
    max_mentions: int | None = None,
    batch_size: int = 64,
    n_process: int = 1,
    show_progress: bool = True,
    desc: str = "ner",
) -> list[list[Mention]]:
    """Batched NER for many texts, with a tqdm progress bar.

    Returns a list parallel to ``texts``: each element is the list of
    deduplicated ``Mention`` records for that article.
    """
    keep = set(keep_labels) if keep_labels else None
    if not texts:
        return []

    nlp = _load(model)
    if nlp is None:
        iterator: Iterable = texts
        if show_progress and _HAS_TQDM:
            iterator = tqdm(texts, desc=desc, unit="art")
        return [_post(_fallback_extract(t or ""), keep, max_mentions) for t in iterator]

    capped = [(t or "")[:_MAX_TEXT_CHARS] for t in texts]
    pipe = nlp.pipe(capped, batch_size=batch_size, n_process=n_process)
    if show_progress and _HAS_TQDM:
        pipe = tqdm(pipe, total=len(capped), desc=desc, unit="art")
    out: list[list[Mention]] = []
    for doc in pipe:
        out.append(_doc_to_mentions(doc, keep_labels=keep, max_mentions=max_mentions))
    return out
