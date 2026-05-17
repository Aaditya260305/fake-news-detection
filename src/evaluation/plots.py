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


def table_markdown(
    rows: Iterable[Mapping],
    cols: list[str],
    title: str | None = None,
    *,
    float_fmt: str = "{:.3f}",
) -> str:
    """Render ``rows`` as a GitHub-flavoured markdown table.

    Floats are formatted with ``float_fmt`` and every cell is padded to
    the column width so the raw markdown is human-readable. Padding also
    prevents editors from auto-collapsing columns when two cells differ
    wildly in width.
    """
    rendered: list[list[str]] = []
    for r in rows:
        row_cells: list[str] = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = float_fmt.format(v)
            row_cells.append(str(v))
        rendered.append(row_cells)

    # column widths
    widths = [max(len(str(c)), *(len(row[i]) for row in rendered) if rendered else (0,))
              for i, c in enumerate(cols)]

    def _pad(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")
    lines.append(_pad(list(cols)))
    lines.append(sep)
    for row_cells in rendered:
        lines.append(_pad(row_cells))
    return "\n".join(lines)
