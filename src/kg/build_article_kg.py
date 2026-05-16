"""Build a small KG for each article.

For every article we:
1. Extract NER mentions (batched -- nlp.pipe with progress).
2. Resolve each mention via the *offline* cache lookup. Cache misses
   are silently treated as "no entity for this mention" -- they will
   NOT trigger live Wikidata calls (use ``src.entity_linking.kb_cache
   --warm`` to populate the cache up front).
3. For each resolved entity, pull its cached claims and add them as
   triples.
4. Add 1-hop "membership" triples ``(entity, MEMBER_OF, class)``.

The result is a list of per-article KGs plus a flat ``(h, r, t)``
triple list ready for TransE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..entity_linking.ner_spacy import extract_mentions, extract_mentions_batch
from ..entity_linking.wikidata_linker import WikidataLinker


log = logging.getLogger(__name__)


MEMBER_OF = "MEMBER_OF"  # virtual relation: entity -> ontology class


@dataclass
class ArticleKG:
    article_id: str
    entity_ids: list[str]               # QIDs in the article
    triples: list[tuple[str, str, str]] = field(default_factory=list)


# ----------------------------------------------------------------------
#  single-article (kept for callers that don't want batching)
# ----------------------------------------------------------------------

def build_for_article(
    article_id: str,
    text: str,
    *,
    linker: WikidataLinker,
    crossview: Mapping[str, str],
    ner_model: str = "en_core_web_sm",
    keep_labels: Iterable[str] | None = None,
    max_mentions: int = 25,
) -> ArticleKG:
    mentions = extract_mentions(
        text, model=ner_model, keep_labels=keep_labels, max_mentions=max_mentions
    )
    return _kg_from_mentions(
        article_id=article_id,
        mentions=[m.text for m in mentions],
        linker=linker,
        crossview=crossview,
    )


# ----------------------------------------------------------------------
#  corpus-level (fast path: batched NER + cache-only linking)
# ----------------------------------------------------------------------

def build_corpus(
    articles,
    *,
    linker: WikidataLinker,
    crossview: Mapping[str, str],
    ner_model: str = "en_core_web_sm",
    keep_labels: Iterable[str] | None = None,
    max_mentions: int = 25,
    show_progress: bool = True,
    batch_size: int = 64,
) -> tuple[list[ArticleKG], list[tuple[str, str, str]]]:
    """Build per-article KGs *and* return the global triple list.

    Implementation
    --------------
    1. Run NER once over the entire corpus with ``nlp.pipe()`` -- this
       is dramatically faster than calling spaCy article-by-article.
    2. For each article, walk its mentions and look them up in the
       linker's cache. If ``linker.offline_only`` is True (default for
       training), uncached mentions are silently dropped.
    """
    articles = list(articles)
    log.info("NER over %d articles ...", len(articles))
    per_article_mentions = extract_mentions_batch(
        [a.text or "" for a in articles],
        model=ner_model,
        keep_labels=keep_labels,
        max_mentions=max_mentions,
        batch_size=batch_size,
        show_progress=show_progress,
        desc="ner",
    )

    per_article: list[ArticleKG] = []
    global_triples: list[tuple[str, str, str]] = []
    cache_hits = 0
    cache_misses = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(
            zip(articles, per_article_mentions),
            total=len(articles),
            desc="kg-build",
            unit="art",
        ) if show_progress else zip(articles, per_article_mentions)
    except Exception:
        iterator = zip(articles, per_article_mentions)

    for a, mentions in iterator:
        entity_ids: list[str] = []
        triples: list[tuple[str, str, str]] = []
        for m in mentions:
            ent = linker.search(m.text)            # cache-only when offline
            if not ent.qid:
                cache_misses += 1
                continue
            cache_hits += 1
            entity_ids.append(ent.qid)
            claims = linker.fetch_claims(ent.qid)
            for p, vals in (claims or {}).items():
                for v in vals:
                    triples.append((ent.qid, p, str(v)))
            cls = crossview.get(ent.qid)
            if cls:
                triples.append((ent.qid, MEMBER_OF, cls))

        # de-duplicate entity ids, preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for q in entity_ids:
            if q in seen:
                continue
            seen.add(q)
            uniq.append(q)

        per_article.append(ArticleKG(article_id=a.id, entity_ids=uniq, triples=triples))
        global_triples.extend(triples)

    total = cache_hits + cache_misses
    if total:
        log.info(
            "mention->entity lookups: %d hits, %d misses (%.1f%% resolved)",
            cache_hits, cache_misses, 100.0 * cache_hits / total,
        )
    return per_article, global_triples


# ----------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------

def _kg_from_mentions(
    *,
    article_id: str,
    mentions: list[str],
    linker: WikidataLinker,
    crossview: Mapping[str, str],
) -> ArticleKG:
    entity_ids: list[str] = []
    triples: list[tuple[str, str, str]] = []
    for m in mentions:
        ent = linker.link(m)
        if not ent.qid:
            continue
        entity_ids.append(ent.qid)
        for p, vals in (ent.claims or {}).items():
            for v in vals:
                triples.append((ent.qid, p, str(v)))
        cls = crossview.get(ent.qid)
        if cls:
            triples.append((ent.qid, MEMBER_OF, cls))
    seen = set()
    uniq = []
    for q in entity_ids:
        if q in seen:
            continue
        seen.add(q)
        uniq.append(q)
    return ArticleKG(article_id=article_id, entity_ids=uniq, triples=triples)
