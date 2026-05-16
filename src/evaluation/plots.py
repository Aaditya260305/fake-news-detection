"""Figure 5/6 analogue plots + Table I/III renderers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_f1_curves(history_files: Mapping[str, str | Path], out_path: str | Path) -> None:
    """Plot val-loss curves for the supplied models (Fig. 5/6 analogue)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, path in history_files.items():
        with open(path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        if isinstance(hist, dict) and "history" in hist:
            hist = hist["history"]
        if not hist:
            continue
        xs = [h["epoch"] for h in hist]
        ys = [h["val_loss"] for h in hist]
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss per epoch (Fig. 5/6 analogue)")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def table_markdown(rows: Iterable[Mapping], cols: list[str], title: str | None = None) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = []
    if title:
        lines.append(f"### {title}\n")
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
