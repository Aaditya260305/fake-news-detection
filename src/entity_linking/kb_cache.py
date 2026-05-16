"""On-disk KB cache for Wikidata responses.

Two JSONL files live under ``data/kb/``:

* ``wikidata_search.jsonl``  -- mention -> {qid, label, description, score}
* ``wikidata_claims.jsonl``  -- qid     -> {P##: [value, ...]}

We keep them as append-only JSONL so the cache is human-inspectable and
trivial to checkpoint into git-lfs / artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

from tqdm import tqdm


log = logging.getLogger(__name__)


class KBCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.search_path = self.root / "wikidata_search.jsonl"
        self.claims_path = self.root / "wikidata_claims.jsonl"
        self._search: dict[str, dict] = {}
        self._claims: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.search_path.exists():
            with open(self.search_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        # later lines silently override earlier ones for
                        # the same mention -- that's the JSONL contract
                        self._search[rec["mention"].lower()] = rec["record"]
                    except Exception:
                        continue
        if self.claims_path.exists():
            with open(self.claims_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._claims[rec["qid"]] = rec["claims"]
                    except Exception:
                        continue
        log.info("KB cache: %d searches, %d claim sets", len(self._search), len(self._claims))

    # ---- search ----

    def get_search(self, mention: str) -> dict | None:
        return self._search.get(mention.lower())

    def put_search(self, mention: str, record: Mapping) -> None:
        self._search[mention.lower()] = dict(record)
        with open(self.search_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"mention": mention, "record": dict(record)}) + "\n")

    # ---- claims ----

    def get_claims(self, qid: str) -> dict | None:
        return self._claims.get(qid)

    def put_claims(self, qid: str, claims: Mapping[str, Iterable[str]]) -> None:
        store = {k: list(v) for k, v in claims.items()}
        self._claims[qid] = store
        with open(self.claims_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"qid": qid, "claims": store}) + "\n")

    # ---- maintenance ----

    def stats(self) -> dict[str, int]:
        empty_searches = sum(
            1 for v in self._search.values()
            if not v or (not v.get("qid") and not v.get("_no_match"))
        )
        no_match_searches = sum(1 for v in self._search.values() if v.get("_no_match"))
        resolved_searches = sum(1 for v in self._search.values() if v.get("qid"))
        return {
            "searches_total": len(self._search),
            "searches_resolved": resolved_searches,
            "searches_no_match": no_match_searches,
            "searches_empty_legacy": empty_searches,
            "claims_total": len(self._claims),
        }

    def purge_empty_searches(self) -> int:
        """Remove cached searches with an *empty* record (legacy 429 poison).

        Returns the number of entries removed. The JSONL file is
        rewritten in place; a ``.bak`` backup is kept.
        """
        keep = {
            k: v for k, v in self._search.items()
            if v and (v.get("qid") or v.get("_no_match"))
        }
        removed = len(self._search) - len(keep)
        if removed == 0:
            return 0
        backup = self.search_path.with_suffix(self.search_path.suffix + ".bak")
        if self.search_path.exists():
            self.search_path.replace(backup)
        # rewrite -- order is no longer the original insertion order, but
        # since later lines override earlier ones the resulting state is
        # equivalent on re-load.
        with open(self.search_path, "w", encoding="utf-8") as f:
            for mention, record in keep.items():
                f.write(json.dumps({"mention": mention, "record": record}) + "\n")
        self._search = keep
        log.info("Purged %d empty search records (backup: %s)", removed, backup)
        return removed


# ----- CLI: warm-up the cache for the corpus -----


def _warm(cfg, *, args) -> None:
    from collections import Counter

    from .ner_spacy import extract_mentions
    from .wikidata_linker import WikidataLinker, _clean_mention

    from ..data import load_real_or_fake, load_fake_news_detection

    cache = KBCache(cfg.paths.kb_cache)
    linker = WikidataLinker(
        cache,
        endpoint=cfg.entity_linking.wikidata_endpoint,
        user_agent=cfg.entity_linking.wikidata_user_agent,
        top_relations=cfg.entity_linking.top_relations,
        rate_limit_s=args.rate_limit_s,
        max_retries=args.max_retries,
    )

    # ---- gather articles ----
    articles = []
    try:
        articles.extend(load_real_or_fake(cfg))
    except FileNotFoundError as e:
        log.warning("%s -- skipping", e)
    if not args.primary_only:
        try:
            articles.extend(load_fake_news_detection(cfg))
        except FileNotFoundError as e:
            log.warning("%s -- skipping", e)

    if args.max_articles and args.max_articles > 0:
        articles = articles[: args.max_articles]
    log.info("Articles to scan: %d", len(articles))

    # ---- extract mentions ----
    mention_counts: Counter = Counter()
    per_article_max = args.max_entities_per_article or cfg.entity_linking.max_entities_per_article
    for art in tqdm(articles, desc="extract"):
        ms = extract_mentions(
            art.text,
            model=cfg.entity_linking.spacy_ner_model,
            keep_labels=cfg.ontology.top_level_types,
            max_mentions=per_article_max,
        )
        for m in ms:
            cleaned = _clean_mention(m.text)
            if len(cleaned) < 2 or cleaned.isdigit():
                continue
            mention_counts[cleaned] += 1

    log.info("Unique cleaned mentions: %d", len(mention_counts))

    # rank by frequency: most-mentioned entities matter most for the KG
    ranked = [m for m, _ in mention_counts.most_common()]
    if args.max_mentions and args.max_mentions > 0:
        ranked = ranked[: args.max_mentions]
    log.info(
        "Linking %d mentions (rate=%.2f req/s, est. wall-time ~%.1f min)",
        len(ranked),
        1.0 / max(linker.rate_limit_s, 0.01),
        len(ranked) * linker.rate_limit_s / 60.0,
    )

    # ---- link mentions ----
    already_searched = 0
    new_resolved = 0
    new_unresolved = 0
    pending_qids: list[str] = []
    for mention in tqdm(ranked, desc="search"):
        if cache.get_search(mention) is not None:
            already_searched += 1
            rec = cache.get_search(mention) or {}
            if rec.get("qid"):
                pending_qids.append(rec["qid"])
            continue
        ent = linker.search(mention)
        if ent.qid:
            new_resolved += 1
            pending_qids.append(ent.qid)
        else:
            new_unresolved += 1

    log.info(
        "search done: cached=%d, new_resolved=%d, new_unresolved=%d",
        already_searched, new_resolved, new_unresolved,
    )

    # ---- batch-fetch claims (50 QIDs/call) ----
    unique_qids = sorted({q for q in pending_qids if q})
    log.info("Fetching claims for %d unique QIDs (batched 50/call) ...", len(unique_qids))
    pbar = tqdm(total=len(unique_qids), desc="claims")
    for start in range(0, len(unique_qids), 50):
        batch = unique_qids[start : start + 50]
        linker.fetch_claims_batch(batch)
        pbar.update(len(batch))
    pbar.close()


def main() -> None:
    from ..config import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--warm", action="store_true", help="warm the cache for both datasets")
    p.add_argument("--stats", action="store_true", help="print cache stats and exit")
    p.add_argument("--purge-empty", action="store_true",
                   help="remove cached searches whose record is {} (legacy 429-poison) so they can be retried")
    p.add_argument("--max-mentions", type=int, default=8000,
                   help="cap on the number of unique mentions to link (most-frequent first); 0 = unlimited")
    p.add_argument("--max-articles", type=int, default=0,
                   help="only scan the first N articles; 0 = all")
    p.add_argument("--max-entities-per-article", type=int, default=0,
                   help="override config; 0 = use config value")
    p.add_argument("--primary-only", action="store_true",
                   help="scan only the Real-or-Fake dataset (skip jruvika)")
    p.add_argument("--rate-limit-s", type=float, default=1.0,
                   help="seconds between successful requests (default 1.0 -- Wikidata's recommended ceiling)")
    p.add_argument("--max-retries", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)
    cache = KBCache(cfg.paths.kb_cache)

    if args.stats:
        for k, v in cache.stats().items():
            print(f"  {k}: {v}")
        return

    if args.purge_empty:
        removed = cache.purge_empty_searches()
        print(f"purged {removed} empty search records.")

    if args.warm:
        _warm(cfg, args=args)


if __name__ == "__main__":
    main()
