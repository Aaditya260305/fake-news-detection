"""Build an ``EoBSchema`` from the Wikidata KB cache.

We walk every linked entity in the cache and use ``P31`` (instance_of)
+ ``P279`` (subclass_of) chains to derive a 4-level class path. The
mapping from Wikidata QIDs to readable class names uses the cached
labels where possible and falls back to QID strings otherwise.

This is the English analogue of the paper's hand-curated Fig. 2:

    PERSON / Politician / HeadOfState / [dob, party]
    ORG    / Company    / TechCompany / [hq, founded]
    GPE    / Country    / -           / [capital, currency]
    ...

The result is saved to ``data/kb/eob_schema.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from .eob_schema import EoBSchema
from ..entity_linking.kb_cache import KBCache


log = logging.getLogger(__name__)


# spaCy NER label -> top-level dimension (D1)
NER_TO_D1: dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "LOC",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "PRODUCT": "PRODUCT",
    "NORP": "NORP",
    "FAC": "FAC",
    "LAW": "LAW",
}


def _resolve_class_chain(
    qid: str,
    cache: KBCache,
    *,
    max_hops: int = 3,
) -> list[str]:
    """Return up-to-3 class QIDs following P31 then P279 from ``qid``."""
    chain: list[str] = []
    visited = {qid}
    current = qid
    for _ in range(max_hops):
        claims = cache.get_claims(current) or {}
        next_qid: str | None = None
        for prop in ("P31", "P279"):
            for v in claims.get(prop, []):
                if v.startswith("Q") and v not in visited:
                    next_qid = v
                    break
            if next_qid:
                break
        if not next_qid:
            break
        chain.append(next_qid)
        visited.add(next_qid)
        current = next_qid
    return chain


def populate_from_wikidata(
    cache: KBCache,
    *,
    ner_label_for_qid: dict[str, str] | None = None,
    top_level_types: Iterable[str] | None = None,
) -> tuple[EoBSchema, dict[str, str]]:
    """Build an ``EoBSchema`` plus a ``qid -> dotted_class_path`` map."""
    schema = EoBSchema(top_level_types=top_level_types)
    crossview: dict[str, str] = {}
    ner_label_for_qid = ner_label_for_qid or {}

    for qid in cache._claims.keys():  # noqa: SLF001 -- direct read
        ner = ner_label_for_qid.get(qid, "MISC")
        d1 = NER_TO_D1.get(ner, "MISC")
        chain = _resolve_class_chain(qid, cache)
        path = [d1] + [c for c in chain[:3]]
        node = schema.add_path(path[:4])
        crossview[qid] = ".".join(path[:3])

        # claim properties become D4 attribute names on the deepest node
        attrs = list((cache.get_claims(qid) or {}).keys())
        for p in attrs:
            if p not in node.attributes:
                node.attributes.append(p)

    return schema, crossview


def main() -> None:
    from ..config import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--crossview-out", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)

    cache = KBCache(cfg.paths.kb_cache)
    schema, crossview = populate_from_wikidata(
        cache, top_level_types=cfg.ontology.top_level_types
    )

    out = Path(args.out or Path(cfg.paths.kb_cache) / "eob_schema.json")
    schema.save(out)
    log.info("Saved schema to %s (%d top-level classes)", out, len(schema.root.children))

    cv_out = Path(args.crossview_out or Path(cfg.paths.kb_cache) / "crossview.json")
    with open(cv_out, "w", encoding="utf-8") as f:
        json.dump(crossview, f, ensure_ascii=False)
    log.info("Saved crossview to %s (%d entities)", cv_out, len(crossview))


if __name__ == "__main__":
    main()
