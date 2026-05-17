"""Aggregate all per-model reports into paper-style tables + plots.

After running ``src.train`` for every model + ablation, every report
lives at ``artifacts/<name>_report.json``. This script collects them and
emits:

* ``reports/table1_detection.md``   -- paper Table I analogue
* ``reports/table2_assessment.md``  -- paper Table II analogue (ROUGE)
* ``reports/table3_ablation.md``    -- paper Table III analogue
* ``reports/figures/val_loss.png``  -- Fig. 5/6 analogue
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .config import load_config
from .evaluation.plots import plot_f1_curves, table_markdown


log = logging.getLogger(__name__)


def _load_reports(artifacts: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(artifacts.glob("*_report.json")):
        with open(path, "r", encoding="utf-8") as f:
            out[path.stem.replace("_report", "")] = json.load(f)
    return out


# ----------------------------------------------------------------------
#  Table II -- Comment generation (paper Sec. V-D, Table II)
# ----------------------------------------------------------------------

# Published numbers from the paper, kept here so the regenerated
# table mirrors the paper's layout. These are NOT computed locally;
# they are reference rows for context only.
PAPER_TABLE2_REFERENCE: list[dict] = [
    {"Method": "RNN [paper]",          "ROUGE-1": 15.55, "ROUGE-2": 2.83, "ROUGE-L": 13.08, "source": "paper"},
    {"Method": "RNN-context [paper]",  "ROUGE-1": 15.85, "ROUGE-2": 3.23, "ROUGE-L": 14.48, "source": "paper"},
    {"Method": "SummaReranker [paper]","ROUGE-1": 23.24, "ROUGE-2": 1.32, "ROUGE-L": 22.72, "source": "paper"},
    {"Method": "EKNet [paper]",        "ROUGE-1": 17.58, "ROUGE-2": 3.43, "ROUGE-L": 15.34, "source": "paper"},
    {"Method": "EKNet+PGN [paper]",    "ROUGE-1": 24.87, "ROUGE-2": 3.85, "ROUGE-L": 23.42, "source": "paper"},
    {"Method": "EKNet+PGN+CVG [paper]","ROUGE-1": 27.45, "ROUGE-2": 4.33, "ROUGE-L": 25.65, "source": "paper"},
]


# Plain-English description of every row that appears in Table II, so
# the markdown file is self-contained -- a reader can understand each
# method without reading the original paper.
TABLE2_METHOD_DESCRIPTIONS: list[tuple[str, str]] = [
    ("RNN",
     "Vanilla recurrent encoder-decoder. The article body is fed to a "
     "BiLSTM encoder; an LSTM decoder emits the comment one token at a "
     "time. No copy mechanism, no entity awareness. This is the weakest "
     "of the paper's baselines."),
    ("RNN-context",
     "Same as RNN but the decoder also conditions on a small context "
     "window around each entity mention. Helps the decoder stay on topic "
     "but still cannot copy rare names verbatim."),
    ("SummaReranker",
     "Strong extractive-summarisation baseline (Liu & Lapata 2022 style). "
     "Generates several candidate comments and re-ranks them with a "
     "learned classifier. Wins on ROUGE-1/L but is content-agnostic "
     "(no KG signal)."),
    ("EKNet",
     "The paper's EKNet credibility model with a plain seq2seq comment "
     "head: text + KG embeddings -> LSTM decoder. No copy mechanism."),
    ("EKNet+PGN",
     "EKNet with a **Pointer-Generator Network**: at each decoding step "
     "the model can either generate from the vocabulary or *copy* a "
     "token directly from the article (great for proper nouns and rare "
     "entities)."),
    ("EKNet+PGN+CVG",
     "EKNet + PGN + **Coverage Mechanism**: an extra attention-coverage "
     "loss discourages the decoder from repeating the same source "
     "tokens. This is the paper's best system."),
    ("Rule-based KG-mismatch (ours)",
     "Our **non-trainable** comment generator. For each article it "
     "(a) runs spaCy NER, (b) links every mention to Wikidata, "
     "(c) flags entities whose NER label disagrees with the Wikidata "
     "`instance_of` claim (e.g. tagged as PERSON but Wikidata says "
     "`Q838948 work of art`), and (d) extracts the top FastTextRank "
     "keywords. The output is a two-sentence template:\n"
     "    \"This article references X, Y, Z but the linked Wikidata "
     "records do not match the claims in context. Key topics: A, B, C.\"\n"
     "ROUGE is computed against the article title as a proxy reference. "
     "This row is **fully reproduced locally** -- no learned decoder."),
]


def evaluate_comment_generator(
    cfg,
    *,
    max_articles: int = 200,
    out_json: Path | None = None,
) -> dict:
    """Run our rule-based comment generator over the test split and
    compute mean ROUGE-1 / 2 / L against article titles (the proxy
    reference defined by ``cfg.comment_generator.rouge_reference_field``).

    Returns the metrics dict and (if ``out_json`` is given) persists it.
    """
    from .data import load_real_or_fake, stratified_split
    from .entity_linking.kb_cache import KBCache
    from .entity_linking.wikidata_linker import WikidataLinker
    from .comment.rule_based import generate_comment
    from .evaluation.metrics import rouge_scores

    log.info("Comment-gen eval: loading data ...")
    items = load_real_or_fake(cfg)
    _, _, test = stratified_split(
        items,
        cfg.dataset.split.train,
        cfg.dataset.split.val,
        cfg.dataset.split.test,
        seed=cfg.seed,
    )
    if max_articles and max_articles > 0:
        test = test[:max_articles]
    log.info("Comment-gen eval: %d articles", len(test))

    cache = KBCache(cfg.paths.kb_cache)
    linker = WikidataLinker(
        cache,
        endpoint=cfg.entity_linking.wikidata_endpoint,
        user_agent=cfg.entity_linking.wikidata_user_agent,
        top_relations=cfg.entity_linking.top_relations,
        offline_only=True,  # never hit the network during evaluation
    )

    ref_field = getattr(cfg.comment_generator, "rouge_reference_field", "title")
    preds: list[str] = []
    refs: list[str] = []
    try:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(test, desc="generate", unit="art")
    except Exception:
        iterator = test

    samples: list[dict] = []
    t0 = time.time()
    for art in iterator:
        res = generate_comment(
            art.text,
            linker=linker,
            ner_model=cfg.entity_linking.spacy_ner_model,
            top_keywords=cfg.comment_generator.top_keywords,
        )
        preds.append(res.text)
        ref = getattr(art, ref_field, "") or ""
        refs.append(str(ref))
        if len(samples) < 5:  # keep a handful for the report
            samples.append({
                "article_id": art.id,
                "label": int(art.label),
                "title": art.title[:120],
                "comment": res.text,
            })

    log.info("Comments produced in %.1fs", time.time() - t0)
    rouge = rouge_scores(preds, refs)
    # rouge_scores returns F1 in [0, 1]; convert to the paper's %-points
    rouge_pct = {k: float(v) * 100.0 for k, v in rouge.items()}

    out = {
        "method": "rule_based_kg_mismatch (ours)",
        "n": len(test),
        "reference_field": ref_field,
        "rouge1": rouge_pct["rouge1"],
        "rouge2": rouge_pct["rouge2"],
        "rougeL": rouge_pct["rougeL"],
        "samples": samples,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log.info("wrote %s", out_json)
    return out


def emit_table2(cfg, artifacts: Path, reports_dir: Path,
                *, max_articles: int = 200) -> None:
    """Write reports/table2_assessment.md combining the paper's published
    numbers with our locally-computed rule-based row, plus a plain-English
    description of every method so the file is self-contained."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    ours = evaluate_comment_generator(
        cfg,
        max_articles=max_articles,
        out_json=artifacts / "comment_generator_report.json",
    )

    rows = list(PAPER_TABLE2_REFERENCE)
    rows.append({
        "Method": "Rule-based KG-mismatch (ours)",
        "ROUGE-1": ours["rouge1"],
        "ROUGE-2": ours["rouge2"],
        "ROUGE-L": ours["rougeL"],
        "source": "this repo",
    })
    table_md = table_markdown(
        rows,
        cols=["Method", "ROUGE-1", "ROUGE-2", "ROUGE-L", "source"],
        title="Table II -- Fake News Assessment (Comment Generation)",
        float_fmt="{:6.2f}",
    )

    # ---- legend: explain every method in plain English ----
    legend_lines = ["", "#### What each row means", ""]
    for name, desc in TABLE2_METHOD_DESCRIPTIONS:
        legend_lines.append(f"- **{name}**: {desc}")

    # ---- example output from our generator ----
    samples_md = ""
    if ours.get("samples"):
        samples_md = "\n#### Example comments produced by our generator\n\n"
        for s in ours["samples"][:3]:
            label = "FAKE" if s.get("label") == 1 else "REAL"
            samples_md += (
                f"- *Article (id={s['article_id']}, true label={label}):* "
                f"\"{s['title']}\"\n"
                f"  *Generated comment:* {s['comment']}\n\n"
            )

    footer = (
        "\n\n*Rows tagged `[paper]` are reproduced verbatim from the original "
        "paper (Liu et al., 2024, Table II) for context; they are not "
        "recomputed locally because the paper trains a learned seq2seq + PGN + "
        "Coverage decoder on Chinese rumor-comment pairs that no public English "
        "dataset ships. Our row is computed on the **Real or Fake** test split "
        f"(n={ours['n']}) with the article title used as the proxy reference, "
        "as configured in `configs/default.yaml -> comment_generator.rouge_reference_field`. "
        "All ROUGE values are F1 percentages.*\n"
    )
    out_path = reports_dir / "table2_assessment.md"
    out_path.write_text(table_md + "\n" + "\n".join(legend_lines) + "\n" + samples_md + footer,
                        encoding="utf-8")
    log.info("wrote %s", out_path)


# ----------------------------------------------------------------------

def emit_tables(cfg, artifacts: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports = _load_reports(artifacts)
    if not reports:
        log.warning("No reports found in %s. Train some models first.", artifacts)
        return

    # ---- Table I: baseline + EKNet (mode=both) ----
    table1_rows = []
    table1_keys = ["fasttext", "textrnn", "textrcnn", "transformer_small", "eknet_both"]
    label_map = {
        "fasttext": "FastText",
        "textrnn": "TextRNN",
        "textrcnn": "TextRCNN",
        "transformer_small": "Transformer",
        "eknet_both": "EKNet (ours)",
    }
    for k in table1_keys:
        r = reports.get(k)
        if not r:
            continue
        table1_rows.append(
            {
                "Method": label_map[k],
                "Precision": r.get("precision"),
                "Recall": r.get("recall"),
                "F1": r.get("f1"),
                "Miss-rate": r.get("miss_rate"),
            }
        )
    t1 = table_markdown(
        table1_rows,
        cols=["Method", "Precision", "Recall", "F1", "Miss-rate"],
        title="Table I -- Fake News Detection (Real or Fake dataset)",
    )
    (reports_dir / "table1_detection.md").write_text(t1, encoding="utf-8")
    log.info("wrote %s", reports_dir / "table1_detection.md")

    # ---- Table III: ablation ----
    ablation_keys = [("eknet_both", "Text+Entity-Ontology"), ("eknet_text_only", "Text"), ("eknet_ontology_only", "Entity-Ontology")]
    rows = []
    for k, label in ablation_keys:
        r = reports.get(k)
        if not r:
            continue
        rows.append(
            {
                "Features": label,
                "Confusion [TP,FP,FN,TN]": str(r.get("confusion_matrix")),
                "Precision": r.get("precision"),
                "Recall": r.get("recall"),
                "Miss-rate": r.get("miss_rate"),
                "F1": r.get("f1"),
            }
        )
    t3 = table_markdown(
        rows,
        cols=["Features", "Confusion [TP,FP,FN,TN]", "Precision", "Recall", "Miss-rate", "F1"],
        title="Table III -- EKNet ablation (Real or Fake dataset)",
    )
    (reports_dir / "table3_ablation.md").write_text(t3, encoding="utf-8")
    log.info("wrote %s", reports_dir / "table3_ablation.md")

    # ---- Fig. 5/6 analogue: val-loss curves ----
    history_files = {}
    for name in ["fasttext", "textrnn", "textrcnn", "transformer_small", "eknet_both"]:
        candidate = artifacts / f"{name}_history.json"
        if candidate.exists():
            history_files[name] = candidate
    if history_files:
        plot_f1_curves(history_files, reports_dir / "figures" / "val_loss.png")
        log.info("wrote %s", reports_dir / "figures" / "val_loss.png")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    p.add_argument("--emit-tables", action="store_true",
                   help="emit Table I, Table II, Table III, and the loss-curve figure")
    p.add_argument("--emit-table2", action="store_true",
                   help="emit only Table II (comment-generation ROUGE)")
    p.add_argument("--max-articles", type=int, default=200,
                   help="cap on test articles used for Table II comment eval "
                        "(default 200; 0 = use the full test split)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)
    if args.emit_tables:
        emit_tables(cfg, Path(args.artifacts), Path(args.reports))
        emit_table2(cfg, Path(args.artifacts), Path(args.reports),
                    max_articles=args.max_articles)
    elif args.emit_table2:
        emit_table2(cfg, Path(args.artifacts), Path(args.reports),
                    max_articles=args.max_articles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
