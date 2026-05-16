"""Disk cache for the expensive deterministic feature pipeline.

Every model run currently repeats Steps 1-5 of ``_prepare_features_eknet``
(tokenise, fit tf-idf, NER, KG build, TransE, encode). Those steps are
deterministic for a given config, so the result can safely be cached.

Implementation
--------------
* A *fingerprint* of all config knobs that influence features is
  recorded alongside the cached arrays. Re-runs that share the same
  fingerprint reuse the cache; any change invalidates it.
* The cache lives under ``artifacts/feature_cache/`` so it can be
  cleaned with a single ``rm -rf``.
* Side artifacts (fitted tf-idf vectorizer, TransE embeddings,
  article->entity map) are also persisted so the Streamlit demo and
  notebooks don't have to re-train them.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .features import FeatureSet


log = logging.getLogger(__name__)


CACHE_VERSION = 2  # bump if the layout / contents change


# ----------------------------------------------------------------------
#  fingerprint
# ----------------------------------------------------------------------

def _to_python(obj: Any) -> Any:
    """Recursively coerce OmegaConf / numpy / Path into JSON-friendly types."""
    if hasattr(obj, "_content"):  # OmegaConf node
        try:
            from omegaconf import OmegaConf
            return _to_python(OmegaConf.to_container(obj, resolve=True))
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _file_signature(path: str | Path) -> dict[str, Any]:
    """Return {path, size, mtime} for ``path`` (None if missing)."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "size": int(st.st_size),
        "mtime": int(st.st_mtime),
    }


def compute_fingerprint(cfg) -> dict[str, Any]:
    """Build a stable fingerprint dict from the parts of the config that
    affect feature values."""
    fp = {
        "version": CACHE_VERSION,
        "seed": int(cfg.seed),
        "dataset": {
            "primary_csv": _file_signature(cfg.dataset.real_or_fake.csv),
            "split": _to_python(cfg.dataset.split),
        },
        "preprocessing": {
            "spacy_model": cfg.preprocessing.spacy_model,
            "max_sentences": int(cfg.preprocessing.max_sentences),
            "max_tokens_per_sentence": int(cfg.preprocessing.max_tokens_per_sentence),
        },
        "text_encoder": {
            "type": cfg.text_encoder.type,
            "glove_dim": int(cfg.text_encoder.glove_dim),
            "glove_file": _file_signature(cfg.paths.glove),
            "minilm_model": getattr(cfg.text_encoder, "minilm_model", None),
        },
        "entity_linking": {
            "ner_model": cfg.entity_linking.spacy_ner_model,
            "max_entities_per_article": int(cfg.entity_linking.max_entities_per_article),
            "top_relations": _to_python(cfg.entity_linking.top_relations),
        },
        "ontology": {
            "top_level_types": _to_python(cfg.ontology.top_level_types),
        },
        "kg": {
            "embedding_dim": int(cfg.kg.embedding_dim),
            "transe_epochs": int(cfg.kg.transe_epochs),
            "transe_batch_size": int(cfg.kg.transe_batch_size),
            "membership_loss_weight": float(cfg.kg.membership_loss_weight),
        },
        # KB cache state -- if Wikidata cache changed materially since
        # the feature build, invalidate. We use the mtime+size of the
        # JSONL files as a coarse signal.
        "kb_cache": {
            "search_jsonl": _file_signature(Path(cfg.paths.kb_cache) / "wikidata_search.jsonl"),
            "claims_jsonl": _file_signature(Path(cfg.paths.kb_cache) / "wikidata_claims.jsonl"),
            "crossview_json": _file_signature(Path(cfg.paths.kb_cache) / "crossview.json"),
        },
    }
    return fp


def fingerprint_digest(fp: Mapping[str, Any]) -> str:
    """Short stable hash for display."""
    s = json.dumps(fp, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------
#  cache I/O
# ----------------------------------------------------------------------

@dataclass
class FeatureCachePaths:
    root: Path
    fingerprint: Path
    features: Path
    tfidf: Path
    transE: Path
    article_entities: Path
    crossview_snapshot: Path

    @classmethod
    def at(cls, root: str | Path) -> "FeatureCachePaths":
        r = Path(root)
        r.mkdir(parents=True, exist_ok=True)
        return cls(
            root=r,
            fingerprint=r / "fingerprint.json",
            features=r / "features.npz",
            tfidf=r / "tfidf.pkl",
            transE=r / "transE.npz",
            article_entities=r / "article_entities.json",
            crossview_snapshot=r / "crossview_snapshot.json",
        )


class FeatureCache:
    """High-level wrapper over the four artifact files."""

    def __init__(self, cfg, root: str | Path | None = None):
        self.cfg = cfg
        self.paths = FeatureCachePaths.at(root or "artifacts/feature_cache")
        self.fingerprint = compute_fingerprint(cfg)
        self.digest = fingerprint_digest(self.fingerprint)

    # ---- validity check ----

    def is_valid(self) -> bool:
        if not self.paths.fingerprint.exists() or not self.paths.features.exists():
            return False
        try:
            with open(self.paths.fingerprint, "r", encoding="utf-8") as f:
                stored = json.load(f)
        except Exception:
            return False
        if stored.get("version") != CACHE_VERSION:
            return False
        # compare the structured fingerprint
        return stored == self.fingerprint

    # ---- save / load ----

    def save(
        self,
        feats: dict[str, FeatureSet],
        *,
        tfidf=None,
        kg_embeddings=None,
        article_to_entities: dict[str, list[str]] | None = None,
        crossview: dict[str, str] | None = None,
    ) -> None:
        # features.npz: stack splits side-by-side
        payload: dict[str, np.ndarray] = {}
        for split in ("train", "val", "test"):
            fs = feats[split]
            payload[f"{split}_text_emb"] = fs.text_emb
            payload[f"{split}_kg_emb"] = fs.kg_emb
            payload[f"{split}_labels"] = fs.labels
            payload[f"{split}_ids"] = np.asarray(fs.ids)
        np.savez(self.paths.features, **payload)

        # side artifacts
        if tfidf is not None:
            try:
                tfidf.save(self.paths.tfidf)
            except Exception as e:
                log.warning("Could not save tf-idf vectorizer: %s", e)
        if kg_embeddings is not None:
            try:
                kg_embeddings.save(self.paths.transE)
            except Exception as e:
                log.warning("Could not save KG embeddings: %s", e)
        if article_to_entities is not None:
            with open(self.paths.article_entities, "w", encoding="utf-8") as f:
                json.dump(article_to_entities, f)
        if crossview is not None:
            with open(self.paths.crossview_snapshot, "w", encoding="utf-8") as f:
                json.dump(crossview, f)

        # fingerprint last (so we never report 'valid' for a half-written cache)
        with open(self.paths.fingerprint, "w", encoding="utf-8") as f:
            json.dump(self.fingerprint, f, indent=2)
        log.info("Feature cache saved (digest=%s) at %s", self.digest, self.paths.root)

    def load(self) -> dict[str, FeatureSet]:
        data = np.load(self.paths.features, allow_pickle=False)
        feats: dict[str, FeatureSet] = {}
        for split in ("train", "val", "test"):
            feats[split] = FeatureSet(
                text_emb=data[f"{split}_text_emb"],
                kg_emb=data[f"{split}_kg_emb"],
                labels=data[f"{split}_labels"],
                ids=data[f"{split}_ids"].tolist(),
            )
        return feats

    # ---- inspection ----

    def explain_mismatch(self) -> str:
        """Return a human-readable diff between the stored and current fingerprints."""
        if not self.paths.fingerprint.exists():
            return "(no cached fingerprint)"
        try:
            with open(self.paths.fingerprint, "r", encoding="utf-8") as f:
                stored = json.load(f)
        except Exception as e:
            return f"(cached fingerprint unreadable: {e})"
        diffs = []
        _diff_recursive(stored, self.fingerprint, prefix="", out=diffs)
        return "\n".join(diffs) or "(fingerprints already match)"


def _diff_recursive(a, b, *, prefix: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            _diff_recursive(a.get(key), b.get(key), prefix=f"{prefix}.{key}" if prefix else key, out=out)
        return
    if a != b:
        out.append(f"  {prefix}: cached={a!r}  current={b!r}")


def clear_cache(root: str | Path = "artifacts/feature_cache") -> int:
    """Remove every file under the cache root. Returns the number removed."""
    r = Path(root)
    if not r.exists():
        return 0
    removed = 0
    for f in r.rglob("*"):
        if f.is_file():
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                log.warning("Could not remove %s: %s", f, e)
    return removed
