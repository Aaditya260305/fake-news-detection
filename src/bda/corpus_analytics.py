"""Corpus-wide big-data analytics report.

Run from the CLI as:

    python -m src.bda.corpus_analytics

Produces:

* ``reports/bda_corpus_stats.md`` -- a single self-contained Markdown
  document covering token / article / entity statistics, Count-Min
  Sketch heavy hitters, a Bloom-filter membership demo, and MinHash
  near-duplicate clusters.
* ``reports/figures/bda_zipf.png``         -- log-log token frequency
* ``reports/figures/bda_article_lengths.png`` -- article length histogram
* ``reports/figures/bda_label_balance.png``   -- label distribution
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..config import load_config
from ..data import Article, load_real_or_fake
from ..entity_linking.kb_cache import KBCache
from ..entity_linking.wikidata_linker import WikidataLinker
from .dedup import find_near_duplicates
from .sketches import BloomFilter, CountMinSketch


log = logging.getLogger(__name__)


_WORD = re.compile(r"[A-Za-z]+")


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


# ----------------------------------------------------------------------
#  individual analyses
# ----------------------------------------------------------------------

def _token_stats(articles: list[Article]) -> dict:
    """Build exact and sketched token-frequency stats."""
    log.info("Token stats: streaming over %d articles ...", len(articles))
    cms = CountMinSketch(width=4096, depth=6)
    exact: Counter[str] = Counter()
    lengths: list[int] = []

    for a in articles:
        toks = _tokens(a.text)
        lengths.append(len(toks))
        for t in toks:
            cms.add(t)
            exact[t] += 1

    candidates = [w for w, _ in exact.most_common(2000)]
    sketch_top = cms.heavy_hitters(candidates, top_k=50)
    exact_top = exact.most_common(50)

    err = 0.0
    for (w, est), (_, true) in zip(sketch_top, exact_top):
        err = max(err, (est - true) / max(true, 1))
    return {
        "total_articles": len(articles),
        "total_tokens": int(sum(lengths)),
        "unique_tokens": len(exact),
        "avg_article_length": float(sum(lengths) / max(len(lengths), 1)),
        "median_article_length": int(sorted(lengths)[len(lengths) // 2]) if lengths else 0,
        "min_article_length": int(min(lengths)) if lengths else 0,
        "max_article_length": int(max(lengths)) if lengths else 0,
        "exact_top_50": exact_top,
        "sketch_top_50": sketch_top,
        "sketch_max_relative_error": err,
        "cms_width": cms.width,
        "cms_depth": cms.depth,
        "cms_memory_kb": (cms.width * cms.depth * 8) / 1024,
    }


def _label_stats(articles: list[Article]) -> dict:
    cnt = Counter(a.label for a in articles)
    return {
        "real": int(cnt.get(0, 0)),
        "fake": int(cnt.get(1, 0)),
        "imbalance_ratio": (cnt.get(1, 0) / max(cnt.get(0, 0), 1)),
    }


def _kg_bloom_demo(cfg) -> dict:
    """Load every QID known to our local Wikidata cache into a Bloom
    filter and report membership-test statistics + memory savings."""
    cache = KBCache(cfg.paths.kb_cache)
    qids = [
        rec["qid"]
        for rec in cache._search.values()   # noqa: SLF001
        if isinstance(rec, dict) and rec.get("qid")
    ]
    qids_set = set(qids)
    if not qids:
        return {
            "n_qids": 0,
            "note": "KB cache is empty -- run `python -m src.entity_linking.kb_cache --warm` first.",
        }

    bf = BloomFilter(capacity=max(len(qids), 1000), fp_rate=0.01)
    bf.update_many(qids)

    hits = sum(1 for q in qids[: min(len(qids), 200)] if q in bf)
    # check false-positive rate on unseen QIDs
    unseen = [f"QNOTREAL{i}" for i in range(500)]
    fp = sum(1 for q in unseen if q in bf)
    return {
        "n_qids": len(qids_set),
        "bloom_stats": bf.stats(),
        "true_positive_rate": hits / max(min(len(qids), 200), 1),
        "false_positive_rate_observed": fp / len(unseen),
        # Memory comparison: 8-byte ptr + average 8-byte QID string vs
        # the Bloom filter's bitset.
        "naive_set_memory_kb_estimate": (len(qids_set) * 24) / 1024,
        "bloom_memory_kb": bf.stats()["memory_kb"],
        "compression_ratio": (len(qids_set) * 24) / max(bf.stats()["memory_kb"] * 1024, 1),
    }


def _entity_frequency_demo(cfg, articles: list[Article], *, max_articles: int = 1000) -> dict:
    """Count linked-entity frequencies across the corpus using only
    the offline cache (no network calls)."""
    cache = KBCache(cfg.paths.kb_cache)
    linker = WikidataLinker(
        cache,
        endpoint=cfg.entity_linking.wikidata_endpoint,
        user_agent=cfg.entity_linking.wikidata_user_agent,
        top_relations=cfg.entity_linking.top_relations,
        offline_only=True,
    )
    from ..entity_linking.ner_spacy import extract_mentions_batch

    take = articles[:max_articles]
    log.info("Entity frequency: NER on %d articles ...", len(take))
    mentions_per_article = extract_mentions_batch(
        [a.text for a in take],
        model=cfg.entity_linking.spacy_ner_model,
        keep_labels=cfg.ontology.top_level_types,
        batch_size=64,
        desc="bda-ner",
    )

    cms = CountMinSketch(width=4096, depth=6)
    exact: Counter[str] = Counter()
    n_mentions = 0
    n_resolved = 0
    for ments in mentions_per_article:
        for m in ments:
            le = linker.search(m.text)
            if not le.qid:
                continue
            n_resolved += 1
            label = le.label or le.qid
            cms.add(label)
            exact[label] += 1
        n_mentions += len(ments)

    return {
        "articles_scanned": len(take),
        "mentions": n_mentions,
        "resolved_to_qid": n_resolved,
        "resolution_rate": n_resolved / max(n_mentions, 1),
        "unique_entities": len(exact),
        "exact_top_30": exact.most_common(30),
        "sketch_top_30": cms.heavy_hitters([k for k, _ in exact.most_common(500)], top_k=30),
    }


# ----------------------------------------------------------------------
#  Markdown rendering
# ----------------------------------------------------------------------

def _bar(label: str, count: int, max_count: int, width: int = 30) -> str:
    n = int(round(width * count / max(max_count, 1)))
    return f"`{'█' * n}{' ' * (width - n)}` {count:>7}  {label}"


def _render_report(stats: dict) -> str:
    out: list[str] = []
    out.append("# Big-Data Analytics Report\n")
    out.append(
        "_Auto-generated by `python -m src.bda.corpus_analytics`. "
        "Demonstrates the big-data techniques referenced by the paper's "
        "title (Liu et al., 2024)._\n"
    )

    # ---- corpus overview ----
    ts = stats["tokens"]
    ls = stats["labels"]
    out.append("## 1. Corpus overview\n")
    out.append(
        f"- **Articles**: {ts['total_articles']:,}\n"
        f"- **Tokens (alphabetic)**: {ts['total_tokens']:,}\n"
        f"- **Unique tokens**: {ts['unique_tokens']:,}\n"
        f"- **Avg article length**: {ts['avg_article_length']:.1f} tokens\n"
        f"- **Median article length**: {ts['median_article_length']} tokens\n"
        f"- **Length range**: {ts['min_article_length']} .. {ts['max_article_length']} tokens\n"
        f"- **Label distribution**: REAL = {ls['real']:,}, FAKE = {ls['fake']:,} "
        f"(imbalance ratio = {ls['imbalance_ratio']:.2f})\n"
    )

    # ---- Count-Min Sketch ----
    out.append("\n## 2. Count-Min Sketch -- streaming token heavy hitters\n")
    out.append(
        "Standard CMS configured with **width=4096, depth=6** "
        f"(~{ts['cms_memory_kb']:.1f} KB of state regardless of vocab size). "
        "Estimates are guaranteed to be **>=** the true count; over-estimate "
        "is bounded by `2 * N / width` with probability `1 - (1/2)^depth`.\n\n"
        f"On this corpus the max relative error on the top-50 tokens is "
        f"**{ts['sketch_max_relative_error']:.2%}** vs an exact `collections.Counter`.\n\n"
        "Exact vs sketched top-10:\n"
    )
    out.append("| rank | token | exact count | sketch estimate |")
    out.append("|---:|---|---:|---:|")
    for i, ((w_e, c_e), (w_s, c_s)) in enumerate(zip(ts["exact_top_50"][:10],
                                                     ts["sketch_top_50"][:10]), 1):
        out.append(f"| {i} | `{w_e}` | {c_e:,} | {c_s:,} |")

    # ---- Bloom filter ----
    out.append("\n## 3. Bloom filter -- KG entity membership at scale\n")
    bf = stats["kg_bloom"]
    if bf.get("n_qids", 0) == 0:
        out.append(bf.get("note", "Bloom-filter step skipped."))
    else:
        s = bf["bloom_stats"]
        out.append(
            f"- **QIDs indexed**: {bf['n_qids']:,}\n"
            f"- **Bloom-filter size**: {s['num_bits']:,.0f} bits "
            f"({s['memory_kb']:.1f} KB), {s['num_hashes']:.0f} hash functions\n"
            f"- **Naive `set[str]` estimate**: ~{bf['naive_set_memory_kb_estimate']:.1f} KB "
            f"(24 B per entry overhead)\n"
            f"- **Compression vs. naive set**: ~{bf['compression_ratio']:.1f}x smaller\n"
            f"- **Empirical true-positive rate**: {bf['true_positive_rate']:.1%} "
            "(should be 100%)\n"
            f"- **Empirical false-positive rate**: {bf['false_positive_rate_observed']:.2%} "
            f"(target {s['configured_fp_rate']:.2%})\n"
        )

    # ---- Entity frequency via sketch ----
    if "entities" in stats:
        e = stats["entities"]
        out.append("\n## 4. Linked-entity heavy hitters (Count-Min over Wikidata-linked entities)\n")
        out.append(
            f"- **Articles scanned**: {e['articles_scanned']:,}\n"
            f"- **Mentions extracted**: {e['mentions']:,}\n"
            f"- **Resolved to a Wikidata QID**: {e['resolved_to_qid']:,} "
            f"({e['resolution_rate']:.1%} of mentions)\n"
            f"- **Unique entities seen**: {e['unique_entities']:,}\n\n"
            "Top-15 entities (exact vs sketched):\n\n"
        )
        out.append("| rank | entity | exact | sketch |")
        out.append("|---:|---|---:|---:|")
        for i, ((w_e, c_e), (w_s, c_s)) in enumerate(zip(e["exact_top_30"][:15],
                                                         e["sketch_top_30"][:15]), 1):
            out.append(f"| {i} | {w_e} | {c_e:,} | {c_s:,} |")

    # ---- MinHash / LSH dedup ----
    out.append("\n## 5. MinHash + LSH -- near-duplicate detection\n")
    d = stats["dedup"]
    out.append(
        f"- **Articles processed**: {d.n_articles:,}\n"
        f"- **MinHash signature size**: {d.num_perm} permutations\n"
        f"- **Shingle size**: {d.shingle}-grams (word level)\n"
        f"- **Jaccard threshold**: {d.threshold:.2f}\n"
        f"- **Backend**: `{d.backend}`\n"
        f"- **Near-duplicate pairs found**: {d.n_duplicate_pairs:,}\n"
        f"- **Articles in a duplicate cluster**: {d.n_articles_in_duplicate_cluster:,} "
        f"({d.dup_ratio:.2%})\n"
        f"- **Duplicate clusters (size >= 2)**: {len(d.clusters):,}\n"
    )
    if d.sample_pairs:
        out.append("\nTop sample duplicate pairs:\n\n")
        out.append("| article A | article B | est. Jaccard |")
        out.append("|---|---|---:|")
        for a, b, j in d.sample_pairs:
            out.append(f"| `{a}` | `{b}` | {j:.3f} |")

    # ---- Pipeline framing ----
    out.append("\n## 6. The KG-building pipeline as MapReduce\n")
    out.append(
        "The per-article knowledge-graph step (`src/kg/build_article_kg.py`) "
        "is already a textbook MapReduce job. Reading it in BDA terms:\n\n"
        "1. **Map.** For each article a streaming worker emits "
        "`(head_entity, relation, tail_value)` triples by combining (i) "
        "spaCy NER mentions, (ii) Wikidata claims fetched from the local "
        "JSONL cache, and (iii) special `MEMBER_OF` triples linking each "
        "entity to its ontology class via the EoBSchema.\n"
        "2. **Shuffle.** All triples are collected into a global multiset.\n"
        "3. **Reduce.** Duplicate triples are deduplicated and indexed by "
        "head, relation, and tail to build a single corpus-wide KG.\n"
        "4. **Embed.** TransE (PyKEEN or native fallback) trains entity "
        "and relation embeddings via mini-batch SGD over the global KG -- "
        "the standard BDA training pattern for graph embeddings at scale.\n\n"
        "At ~6k articles this all fits in memory, but the algorithms above "
        "(streaming NER, append-only JSONL Wikidata cache, mini-batch "
        "TransE) port unchanged to a Spark/Flink cluster on a 6 M-article "
        "corpus.\n"
    )

    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------
#  figures
# ----------------------------------------------------------------------

def _save_figures(stats: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log.warning("matplotlib unavailable -- skipping BDA figures.")
        return

    ts = stats["tokens"]

    # Zipf -- log-log token frequency
    counts = [c for _, c in ts["exact_top_50"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(range(1, len(counts) + 1), counts, marker="o")
    ax.set_xlabel("rank")
    ax.set_ylabel("count")
    ax.set_title("Token frequency (Zipf, top 50)")
    fig.tight_layout()
    fig.savefig(fig_dir / "bda_zipf.png", dpi=150)
    plt.close(fig)

    # Label balance
    ls = stats["labels"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["REAL", "FAKE"], [ls["real"], ls["fake"]], color=["#4c9", "#e76"])
    ax.set_title("Label balance")
    fig.tight_layout()
    fig.savefig(fig_dir / "bda_label_balance.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
#  driver
# ----------------------------------------------------------------------

def run(*, max_dedup_articles: int = 0, max_entity_articles: int = 1000,
        threshold: float = 0.85) -> dict:
    """Run every analysis and return a stats dict."""
    cfg = load_config()
    articles = load_real_or_fake(cfg)
    log.info("Loaded %d articles from %s", len(articles), cfg.dataset.real_or_fake.csv)

    take = articles if max_dedup_articles <= 0 else articles[:max_dedup_articles]
    stats: dict = {
        "config": {
            "n_articles": len(articles),
            "dedup_subset": len(take),
            "entity_subset": max_entity_articles,
            "dedup_threshold": threshold,
        },
        "tokens": _token_stats(articles),
        "labels": _label_stats(articles),
        "kg_bloom": _kg_bloom_demo(cfg),
    }
    try:
        stats["entities"] = _entity_frequency_demo(
            cfg, articles, max_articles=max_entity_articles
        )
    except Exception as e:
        log.warning("Entity sketch demo skipped (%s)", e)

    t0 = time.time()
    stats["dedup"] = find_near_duplicates(take, threshold=threshold)
    log.info("MinHash dedup: %.1fs", time.time() - t0)

    return stats


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Run the EKNet big-data analytics pipeline over the corpus."
    )
    p.add_argument("--reports", default="reports",
                   help="output directory (default: reports/)")
    p.add_argument("--max-dedup-articles", type=int, default=0,
                   help="cap on articles fed into MinHash dedup (0 = use all)")
    p.add_argument("--max-entity-articles", type=int, default=1000,
                   help="cap on articles for the entity-frequency demo (default 1000)")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="MinHash Jaccard threshold for duplicates (default 0.85)")
    p.add_argument("--json", default=None,
                   help="also write the raw stats dict to this JSON path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s | %(message)s")

    stats = run(
        max_dedup_articles=args.max_dedup_articles,
        max_entity_articles=args.max_entity_articles,
        threshold=args.threshold,
    )

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    md = _render_report(stats)
    out_path = reports / "bda_corpus_stats.md"
    out_path.write_text(md, encoding="utf-8")
    log.info("wrote %s", out_path)
    _save_figures(stats, reports / "figures")

    if args.json:
        # de-dataclass the DupReport for json serialisation
        d = stats["dedup"]
        stats_for_json = {
            **{k: v for k, v in stats.items() if k != "dedup"},
            "dedup": {
                "n_articles": d.n_articles,
                "n_duplicate_pairs": d.n_duplicate_pairs,
                "n_articles_in_duplicate_cluster": d.n_articles_in_duplicate_cluster,
                "dup_ratio": d.dup_ratio,
                "threshold": d.threshold,
                "num_perm": d.num_perm,
                "shingle": d.shingle,
                "backend": d.backend,
                "n_clusters": len(d.clusters),
                "sample_pairs": d.sample_pairs,
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats_for_json, f, indent=2, default=str)
        log.info("wrote %s", args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
