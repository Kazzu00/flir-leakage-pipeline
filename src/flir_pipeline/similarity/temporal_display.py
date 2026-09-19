"""Presentation of recorded pair summaries; no similarity computation or fitting."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

GAP_BINS = ("1", "2-5", "6-10", "11-25", "26-50", "51-100", ">100")


def temporal_display_table(summary: pd.DataFrame, encoder: str) -> pd.DataFrame:
    """Reuse linear quartiles/counts already computed from same-sequence pairs.

    Keep the pre-existing bins, including empty ones, rather than choosing bins
    by the appearance of the trend. A populated delta=0 must remain visible.
    """
    rows = summary.loc[summary.extractor.eq(encoder)].copy()
    if rows.frame_delta_bin.duplicated().any():
        raise ValueError("Duplicate temporal bins")
    rows = rows.set_index("frame_delta_bin")
    if not set(GAP_BINS) <= set(rows.index) or set(rows.index) - {*GAP_BINS, "0"}:
        raise ValueError("Unexpected temporal bins; review presentation explicitly")
    bins = (["0"] if "0" in rows.index and rows.loc["0", "count"] > 0 else []) + list(GAP_BINS)
    rows = rows.loc[bins, ["count", "median", "Q1", "Q3"]].rename(columns={"count": "pair_count"})
    counts = rows.pair_count.to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Invalid pair counts")
    populated = rows.loc[rows.pair_count.gt(0)]
    values = populated[["Q1", "median", "Q3"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any() or (np.abs(values) > 1.000001).any():
        raise ValueError("Invalid recorded quartiles")
    rows.loc[rows.pair_count.eq(0), ["median", "Q1", "Q3"]] = np.nan
    rows["pair_count"] = rows.pair_count.astype(int)
    return rows.reset_index()


def temporal_summary_png(summary: pd.DataFrame, encoder: str) -> bytes:
    """Median and IQR, with categorical real-index labels and an independent Y."""
    rows = temporal_display_table(summary, encoder)
    figure = Figure(figsize=(10, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    x = np.arange(len(rows))
    color = "#176b87" if encoder == "DINOv2" else "#9a542d"
    ax.fill_between(x, rows.Q1, rows.Q3, color=color, alpha=.16, label="Q1–Q3 · 50% central de pares")
    ax.plot(x, rows["median"], "o-", color=color, linewidth=2, label="Mediana")
    labels = [f"{r.frame_delta_bin.replace('-', '–')}\nn={r.pair_count:,}" for r in rows.itertuples()]
    ax.set_xticks(x, labels, fontsize=10)
    ax.set(xlabel="Diferencia de índice inferido · pares de la misma secuencia",
           ylabel="Similitud coseno", title=f"{encoder} · similitud por separación de índices")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    figure.text(.5, -.025, "Índices inferidos de nombres; no son timestamps. Banda descriptiva, no intervalo de confianza.",
                ha="center", fontsize=9, color="#526573")
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=150, bbox_inches="tight")
    return output.getvalue()
