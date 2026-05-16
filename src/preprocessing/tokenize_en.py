"""English tokenizer + sentence splitter.

The paper's Sec. IV-C-2 introduces a *Chinese* word segmentation method;
English does not need it. We instead lean on spaCy and gracefully fall
back to NLTK / regex if spaCy is unavailable.

Performance notes
-----------------
* ``_load_spacy()`` returns a MINIMAL pipeline (only the tokenizer +
  rule-based sentencizer) -- ~5-10x faster than the default pipeline
  for our purposes since we don't need parser/tagger/NER here.
* ``tokenize_article()`` parses the whole article ONCE and walks
  ``doc.sents`` + tokens, instead of the old behaviour which parsed
  the whole text once for sentence boundaries and then re-parsed every
  sentence to get tokens (so ~40 parses per article).
* ``tokenize_articles_batch()`` uses ``nlp.pipe()`` with progress so
  callers get visibility into long-running tokenisation phases.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable, Sequence

try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


log = logging.getLogger(__name__)


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201c])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_MAX_TEXT_CHARS = 1_000_000


# ----------------------------------------------------------------------
#  spaCy loader
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_spacy(model: str = "en_core_web_sm"):
    """Return a MINIMAL spaCy pipeline (tokenizer + sentencizer only).

    Disabling parser/tagger/NER/lemmatizer/attribute_ruler gives a big
    speedup for pure tokenisation, and we add a rule-based
    ``sentencizer`` so ``doc.sents`` still works.
    """
    if not _HAS_SPACY:
        return None
    try:
        nlp = spacy.load(
            model,
            disable=["lemmatizer", "tagger", "attribute_ruler", "parser", "ner"],
        )
        if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        log.debug("Loaded minimal spaCy pipeline: %s", nlp.pipe_names)
        return nlp
    except Exception as e:
        log.warning("spaCy load failed (%s); falling back to NLTK/regex.", e)
        return None


def _ensure_nltk() -> bool:
    if not _HAS_NLTK:
        return False
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            return False
    return True


# ----------------------------------------------------------------------
#  per-article helpers (single text)
# ----------------------------------------------------------------------

def split_sentences(text: str, spacy_model: str = "en_core_web_sm") -> list[str]:
    """Return a list of sentence strings."""
    if not text:
        return []
    nlp = _load_spacy(spacy_model)
    if nlp is not None:
        doc = nlp(text[:_MAX_TEXT_CHARS])
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    if _ensure_nltk():
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def tokenize_words(text: str, spacy_model: str = "en_core_web_sm") -> list[str]:
    """Return a list of lowercase alphabetic-ish tokens."""
    nlp = _load_spacy(spacy_model)
    if nlp is not None:
        doc = nlp(text[:_MAX_TEXT_CHARS])
        return [t.text.lower() for t in doc if t.is_alpha]
    if _ensure_nltk():
        return [t.lower() for t in word_tokenize(text) if any(c.isalpha() for c in t)]
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _doc_to_sentences(
    doc,
    *,
    max_sentences: int | None,
    max_tokens_per_sentence: int | None,
) -> list[list[str]]:
    """Walk an already-parsed spaCy doc and produce [[tok, ...], ...]."""
    out: list[list[str]] = []
    for i, sent in enumerate(doc.sents):
        if max_sentences is not None and i >= max_sentences:
            break
        toks = [t.text.lower() for t in sent if t.is_alpha]
        if max_tokens_per_sentence is not None:
            toks = toks[:max_tokens_per_sentence]
        if toks:
            out.append(toks)
    return out


def _fallback_tokenize(
    text: str,
    *,
    max_sentences: int | None,
    max_tokens_per_sentence: int | None,
) -> list[list[str]]:
    sents = split_sentences(text)
    if max_sentences is not None:
        sents = sents[:max_sentences]
    out = []
    for s in sents:
        toks = tokenize_words(s)
        if max_tokens_per_sentence is not None:
            toks = toks[:max_tokens_per_sentence]
        if toks:
            out.append(toks)
    return out


def tokenize_article(
    text: str,
    spacy_model: str = "en_core_web_sm",
    max_sentences: int | None = None,
    max_tokens_per_sentence: int | None = None,
) -> list[list[str]]:
    """Tokenise a single article into sentences-of-tokens.

    Performs exactly ONE spaCy parse per call (the old version did
    ~N parses for an N-sentence article).
    """
    if not text:
        return []
    nlp = _load_spacy(spacy_model)
    if nlp is None:
        return _fallback_tokenize(
            text,
            max_sentences=max_sentences,
            max_tokens_per_sentence=max_tokens_per_sentence,
        )
    doc = nlp(text[:_MAX_TEXT_CHARS])
    return _doc_to_sentences(
        doc,
        max_sentences=max_sentences,
        max_tokens_per_sentence=max_tokens_per_sentence,
    )


# ----------------------------------------------------------------------
#  batched helper (many texts)
# ----------------------------------------------------------------------

def tokenize_articles_batch(
    texts: Sequence[str],
    spacy_model: str = "en_core_web_sm",
    max_sentences: int | None = None,
    max_tokens_per_sentence: int | None = None,
    batch_size: int = 64,
    n_process: int = 1,
    desc: str = "tokenize",
    show_progress: bool = True,
) -> list[list[list[str]]]:
    """Tokenise many articles with ``nlp.pipe()`` + a tqdm progress bar.

    Returns a parallel list of ``tokenize_article()`` outputs.

    Notes
    -----
    * ``n_process=1`` is the safe default on Windows. spaCy's
      multi-process pipe has known issues there; bump only on Linux.
    * The progress bar shows position/total/throughput/ETA.
    """
    if not texts:
        return []
    nlp = _load_spacy(spacy_model)
    if nlp is None:
        # fallback: serial loop with progress
        iterator: Iterable = texts
        if show_progress and _HAS_TQDM:
            iterator = tqdm(texts, desc=desc, unit="art")
        return [
            _fallback_tokenize(
                t or "",
                max_sentences=max_sentences,
                max_tokens_per_sentence=max_tokens_per_sentence,
            )
            for t in iterator
        ]

    capped = [(t or "")[:_MAX_TEXT_CHARS] for t in texts]
    out: list[list[list[str]]] = []
    pipe = nlp.pipe(capped, batch_size=batch_size, n_process=n_process)
    if show_progress and _HAS_TQDM:
        pipe = tqdm(pipe, total=len(capped), desc=desc, unit="art")
    for doc in pipe:
        out.append(
            _doc_to_sentences(
                doc,
                max_sentences=max_sentences,
                max_tokens_per_sentence=max_tokens_per_sentence,
            )
        )
    return out


def join_tokens(sentences: Iterable[Iterable[str]]) -> str:
    """Flatten sentence-of-tokens to a single space-joined string (for tf-idf)."""
    return " ".join(t for s in sentences for t in s)
