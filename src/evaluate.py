"""Aggregate all per-model reports into paper-style tables + plots.

After running ``src.train`` for every model + ablation, every report
lives at ``artifacts/<name>_report.json``. This script collects them and
emits:

* ``reports/table1_detection.md``   -- paper Table I analogue
* ``reports/table3_ablation.md``    -- paper Table III analogue
* ``reports/figures/val_loss.png``  -- Fig. 5/6 analogue
"""
from __future__ import annotations

import argparse
import json
import logging
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
    p.add_argument("--emit-tables", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)
    if args.emit_tables:
        emit_tables(cfg, Path(args.artifacts), Path(args.reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
