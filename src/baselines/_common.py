"""Shared infrastructure for the neural baselines (TextRNN/RCNN/Transformer).

All three share the same:
  * vocabulary built from training tokens (top-N most frequent),
  * GloVe-initialised embedding lookup table,
  * dataloader yielding ``(token_ids, length, label)``,
  * training loop that early-stops on validation loss.

Performance notes
-----------------
The original implementation tokenised every article from scratch in
``ArticleDataset.__getitem__``, which meant spaCy was invoked once per
sample per epoch (~5067 train * 10 epochs * spaCy parse). On CPU that
looked like a hang. We now pre-tokenise every split exactly once and
keep the token-id sequences in memory so dataloading is O(1) per batch.

The pre-tokenised token strings are cached to disk under
``artifacts/baseline_token_cache/`` so all three neural baselines (and
re-runs of the same baseline) share the same expensive spaCy pass.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

from ..data import Article
from ..models.text_encoder import GloveLoader
from ..preprocessing.tokenize_en import tokenize_articles_batch


log = logging.getLogger(__name__)


PAD = "<PAD>"
UNK = "<UNK>"


# ----------------------------------------------------------------------
#  pre-tokenisation cache (shared across baselines)
# ----------------------------------------------------------------------

def _cache_key(splits, cfg) -> str:
    """Stable hash of the inputs that affect tokenisation."""
    payload = {
        "spacy_model": cfg.preprocessing.spacy_model,
        "max_sentences": int(cfg.preprocessing.max_sentences),
        "max_tokens_per_sentence": int(cfg.preprocessing.max_tokens_per_sentence),
        "seed": int(cfg.seed),
        "split_sizes": [len(s) for s in splits],
        # use the id list per split so re-runs with the same split are stable
        "split_ids": [[a.id for a in s] for s in splits],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _flatten_tokens(per_article: list[list[list[str]]]) -> list[list[str]]:
    """Drop sentence boundaries -- baselines work on a flat token stream."""
    return [[t for sent in toks for t in sent] for toks in per_article]


def tokenize_splits_cached(splits, cfg) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Pre-tokenise (train, val, test) ONCE with on-disk cache.

    Returns three lists of flat token lists, one per split, indexed
    parallel to ``splits``.
    """
    train, val, test = splits
    key = _cache_key(splits, cfg)
    cache_dir = Path("artifacts/baseline_token_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info("Reusing baseline token cache (key=%s): %d/%d/%d articles",
                     key, len(data["train"]), len(data["val"]), len(data["test"]))
            return data["train"], data["val"], data["test"]
        except Exception as e:
            log.warning("Could not load token cache (%s); rebuilding.", e)

    log.info("Pre-tokenising baselines (key=%s)  train=%d val=%d test=%d",
             key, len(train), len(val), len(test))
    out: dict[str, list[list[str]]] = {}
    for name, articles in [("train", train), ("val", val), ("test", test)]:
        t0 = time.time()
        per_article = tokenize_articles_batch(
            [a.text for a in articles],
            spacy_model=cfg.preprocessing.spacy_model,
            max_sentences=cfg.preprocessing.max_sentences,
            max_tokens_per_sentence=cfg.preprocessing.max_tokens_per_sentence,
            batch_size=64,
            desc=f"tokenise/{name}",
        )
        out[name] = _flatten_tokens(per_article)
        log.info("  %s: %d articles in %.1fs", name, len(out[name]), time.time() - t0)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        log.info("Token cache saved at %s", cache_path)
    except Exception as e:
        log.warning("Could not write token cache: %s", e)
    return out["train"], out["val"], out["test"]


# ----------------------------------------------------------------------
#  vocabulary and embedding matrix
# ----------------------------------------------------------------------

def build_vocab_from_tokens(
    train_tokens: Sequence[Sequence[str]],
    max_size: int = 30_000,
    min_count: int = 2,
) -> dict[str, int]:
    cnt: Counter = Counter()
    for toks in train_tokens:
        cnt.update(toks)
    most = [w for w, c in cnt.most_common() if c >= min_count][: max_size - 2]
    vocab = {PAD: 0, UNK: 1}
    for w in most:
        vocab[w] = len(vocab)
    log.info("Vocab built: %d types (corpus tokens=%d)", len(vocab), sum(cnt.values()))
    return vocab


def build_embedding_matrix(vocab: dict[str, int], glove: GloveLoader) -> np.ndarray:
    mat = np.zeros((len(vocab), glove.dim), dtype=np.float32)
    hits = 0
    for w, i in vocab.items():
        if w in (PAD, UNK):
            continue
        vec = glove.get(w)
        if np.any(vec):
            hits += 1
        mat[i] = vec
    log.info("GloVe coverage: %d / %d (%.1f%%)", hits, len(vocab), 100.0 * hits / max(len(vocab), 1))
    return mat


# ----------------------------------------------------------------------
#  encoded dataset
# ----------------------------------------------------------------------

def _encode_to_ids(tokens: Sequence[str], vocab: dict[str, int], max_len: int) -> list[int]:
    unk = vocab[UNK]
    ids: list[int] = []
    for t in tokens:
        ids.append(vocab.get(t, unk))
        if len(ids) >= max_len:
            break
    if not ids:
        ids = [unk]
    return ids


class EncodedDataset(Dataset):
    """Holds pre-encoded ``list[int]`` token ids + labels."""

    def __init__(self, articles: list[Article], encoded: list[list[int]]):
        assert len(articles) == len(encoded)
        self.labels = torch.tensor([a.label for a in articles], dtype=torch.float)
        self.encoded = [torch.tensor(ids, dtype=torch.long) for ids in encoded]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        return self.encoded[idx], self.labels[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = max(s.shape[0] for s in seqs)
    padded = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s
    return padded, lengths, torch.stack(labels)


def prepare_baseline_data(
    cfg,
    splits,
    *,
    max_len: int | None = None,
) -> tuple[dict[str, int], np.ndarray, tuple[EncodedDataset, EncodedDataset, EncodedDataset]]:
    """One-stop helper: tokenise, build vocab + embeddings, encode splits.

    Returns ``(vocab, embedding_matrix, (train_ds, val_ds, test_ds))``.
    """
    train, val, test = splits
    if max_len is None:
        max_len = int(getattr(cfg.baselines, "max_len", 256))

    log.info("=" * 70)
    log.info("=  Baseline data prep (1/3): tokenisation")
    log.info("=" * 70)
    train_toks, val_toks, test_toks = tokenize_splits_cached(splits, cfg)

    log.info("=" * 70)
    log.info("=  Baseline data prep (2/3): vocab + GloVe lookup")
    log.info("=" * 70)
    vocab = build_vocab_from_tokens(train_toks)
    glove = GloveLoader(cfg.paths.glove, dim=cfg.text_encoder.glove_dim)
    pretrained = build_embedding_matrix(vocab, glove)

    log.info("=" * 70)
    log.info("=  Baseline data prep (3/3): id encoding (max_len=%d)", max_len)
    log.info("=" * 70)
    train_ids = [_encode_to_ids(t, vocab, max_len) for t in train_toks]
    val_ids = [_encode_to_ids(t, vocab, max_len) for t in val_toks]
    test_ids = [_encode_to_ids(t, vocab, max_len) for t in test_toks]
    log.info("Encoded: train=%d val=%d test=%d (mean len train=%.1f)",
             len(train_ids), len(val_ids), len(test_ids),
             np.mean([len(x) for x in train_ids]) if train_ids else 0.0)

    return vocab, pretrained, (
        EncodedDataset(train, train_ids),
        EncodedDataset(val, val_ids),
        EncodedDataset(test, test_ids),
    )


def make_loaders_from_datasets(datasets, batch_size: int):
    train_ds, val_ds, test_ds = datasets
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    te = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return tr, va, te


# ----------------------------------------------------------------------
#  vocab persistence (used by the Streamlit demo to run baselines live)
# ----------------------------------------------------------------------

def save_vocab(vocab: dict[str, int], path: str | Path, *, max_len: int) -> None:
    """Persist the vocab + max_len alongside the model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "max_len": int(max_len)}, f)


def load_vocab(path: str | Path) -> tuple[dict[str, int], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["vocab"], int(data.get("max_len", 256))


def encode_text_for_inference(
    text: str,
    vocab: dict[str, int],
    max_len: int,
    *,
    spacy_model: str = "en_core_web_sm",
) -> torch.Tensor:
    """Tokenise + encode a single article for live baseline inference."""
    from ..preprocessing.tokenize_en import tokenize_article
    sents = tokenize_article(text, spacy_model=spacy_model, max_sentences=40, max_tokens_per_sentence=64)
    flat = [t for s in sents for t in s]
    ids = _encode_to_ids(flat, vocab, max_len)
    return torch.tensor(ids, dtype=torch.long)


# ----------------------------------------------------------------------
#  generic training loop
# ----------------------------------------------------------------------

def _set_torch_threads(cfg) -> None:
    """Configure PyTorch CPU threading. Idempotent across calls."""
    import os
    n = int(getattr(cfg.baselines, "num_threads", 0) or 0)
    if n <= 0:
        n = os.cpu_count() or 1
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
    except Exception:
        pass
    log.info("PyTorch threads: intra=%d  inter=%d", torch.get_num_threads(), torch.get_num_interop_threads())


def train_loop(
    cfg,
    model: nn.Module,
    loaders,
    *,
    epochs: int | None = None,
    lr: float | None = None,
) -> tuple[dict, dict]:
    """Generic train + early-stopping loop. Returns (report, history)."""
    from ..evaluation.metrics import classification_report

    _set_torch_threads(cfg)

    train_loader, val_loader, test_loader = loaders
    lr = lr or cfg.training.lr
    epochs = epochs or cfg.baselines.epochs
    log.info("Training: %d epochs, lr=%.4g, batch_size=%d, train_batches=%d",
             epochs, lr, train_loader.batch_size, len(train_loader))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.model.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    patience = 0
    history = []

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        iterator = train_loader
        if _HAS_TQDM:
            iterator = tqdm(train_loader, desc=f"ep {ep + 1}/{epochs}", leave=False)
        for x, lens, y in iterator:
            opt.zero_grad()
            logits = model(x, lens)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach()) * y.shape[0]
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, lens, y in val_loader:
                va_loss += float(loss_fn(model(x, lens), y)) * y.shape[0]
        va_loss /= len(val_loader.dataset)
        history.append({"epoch": ep + 1, "train_loss": tr_loss, "val_loss": va_loss})
        log.info("ep %02d  train=%.4f  val=%.4f  (%.1fs)",
                 ep + 1, tr_loss, va_loss, time.time() - t0)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.training.early_stopping_patience:
                log.info("Early stopping at epoch %d", ep + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, lens, y in test_loader:
            probs = torch.sigmoid(model(x, lens))
            preds = (probs >= 0.5).long()
            y_true.extend(y.long().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())
    return classification_report(y_true, y_pred, y_prob), {"history": history}


# ----------------------------------------------------------------------
#  backwards-compatible shims (so old call sites keep working)
# ----------------------------------------------------------------------

def build_vocab(articles, max_size: int = 30_000, min_count: int = 2) -> dict[str, int]:
    """Legacy entry point -- left for safety; new baselines use
    ``prepare_baseline_data`` which is much faster."""
    log.warning(
        "build_vocab(articles) is deprecated; switch to "
        "tokenize_splits_cached + build_vocab_from_tokens."
    )
    from ..preprocessing.tokenize_en import tokenize_article
    cnt: Counter = Counter()
    for a in articles:
        for sent in tokenize_article(a.text, max_sentences=40, max_tokens_per_sentence=64):
            cnt.update(sent)
    most = [w for w, c in cnt.most_common() if c >= min_count][: max_size - 2]
    vocab = {PAD: 0, UNK: 1}
    for w in most:
        vocab[w] = len(vocab)
    return vocab


def make_loaders(splits, vocab, max_len: int, batch_size: int):
    """Legacy entry point -- still works but slow (no token cache).
    Prefer ``prepare_baseline_data`` + ``make_loaders_from_datasets``."""
    log.warning("make_loaders() is deprecated; switch to prepare_baseline_data().")
    from ..preprocessing.tokenize_en import tokenize_article

    train, val, test = splits

    class _LegacyDataset(Dataset):
        def __init__(self, arts):
            self.arts = arts

        def __len__(self):
            return len(self.arts)

        def __getitem__(self, idx):
            a = self.arts[idx]
            sents = tokenize_article(a.text, max_sentences=40, max_tokens_per_sentence=64)
            ids = []
            unk = vocab[UNK]
            for s in sents:
                for t in s:
                    ids.append(vocab.get(t, unk))
                    if len(ids) >= max_len:
                        break
                if len(ids) >= max_len:
                    break
            if not ids:
                ids = [unk]
            return torch.tensor(ids, dtype=torch.long), torch.tensor(a.label, dtype=torch.float)

    tr = DataLoader(_LegacyDataset(train), batch_size=batch_size, shuffle=True, collate_fn=collate)
    va = DataLoader(_LegacyDataset(val), batch_size=batch_size, shuffle=False, collate_fn=collate)
    te = DataLoader(_LegacyDataset(test), batch_size=batch_size, shuffle=False, collate_fn=collate)
    return tr, va, te
