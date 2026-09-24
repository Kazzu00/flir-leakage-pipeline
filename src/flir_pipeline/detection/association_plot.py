"""Descriptive scatter panels; no fitted trends, correlations or significance tests."""

from __future__ import annotations

from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from flir_pipeline.detection.association import (
    COLORS,
    MARKERS,
    METRIC_LABELS,
    NOTICE,
    PENDING,
    RESIDUAL_LABELS,
    STRATEGIES,
    STRATEGY_LABELS,
    AssociationEvidence,
    select_view,
)


def association_figure(evidence: AssociationEvidence, registry):
    # Panel identity is stable if the YAML rows are reordered.
    order = ["primary_dinov2", "primary_clip", "extreme_dinov2", "extreme_clip", "temporal_at5", "domain_dinov2", "domain_clip"]
    if set(registry.association_id) != set(order):
        raise ValueError("Association figure requires the prespecified v1 panel set")
    registry = registry.set_index("association_id").loc[order].reset_index()
    if not evidence.complete:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.axis("off")
        ax.text(.04, .94, "Detector ↔ residual similarity", fontsize=21, weight="bold", transform=ax.transAxes)
        state = "PENDING COMPUTE" if evidence.state == PENDING else "PARTIAL · scientific association withheld"
        ax.text(.04, .84, state, fontsize=16, color="#137c8b", transform=ax.transAxes)
        ax.text(.04, .75, f"{evidence.fairness['completed_runs']}/{evidence.fairness['expected_runs']} controlled runs", fontsize=14, transform=ax.transAxes)
        # Use column access for the display label; 'class' is not a Python tuple field.
        lines = [f"{r.category}: {r.population_label} {METRIC_LABELS[r.detector_metric]} vs {RESIDUAL_LABELS[r.residual_metric]}"
                 for r in registry.rename(columns={"class": "population_label"}).itertuples()]
        ax.text(.04, .65, "PRESPECIFIED ASSOCIATIONS\n\n" + "\n".join(lines), fontsize=11, va="top", linespacing=1.5, transform=ax.transAxes)
        ax.text(.04, .11, "Small pilots: infrastructure only. No scientific performance values shown.", fontsize=10, transform=ax.transAxes)
        ax.text(.04, .01, fill(NOTICE.split(" Cada punto")[0], width=125), fontsize=9, va="bottom", transform=ax.transAxes)
        fig.tight_layout()
        return fig
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3, bottom=.23, top=.91, hspace=.45, wspace=.35)
    for i, row in enumerate(registry.iloc[:5].itertuples()):
        ax = fig.add_subplot(grid[i//3, i%3])
        scatter(ax, evidence, row.detector_metric, row.residual_metric, row.class_id)
        ax.set_title(f"{'ABCDE'[i]}. {METRIC_LABELS[row.detector_metric]}", loc="left")
    last = grid[1, 2].subgridspec(1, 2, wspace=.35)
    for i, row in enumerate(registry.iloc[5:].itertuples()):
        ax = fig.add_subplot(last[0, i])
        scatter(ax, evidence, row.detector_metric, row.residual_metric, row.class_id)
        encoder = "DINOv2" if i == 0 else "CLIP"
        ax.set_title(f"F{'12'[i]}. Heavy Machinery\nRecall · {encoder}", fontsize=9, loc="left")
    fig.suptitle("Detector ↔ residual similarity · PRESPECIFIED ASSOCIATIONS", fontsize=17, weight="bold")
    handles = [Line2D([], [], color=c, marker=m, linestyle="", label=STRATEGY_LABELS[s])
               for s, c, m in zip(STRATEGIES, COLORS, MARKERS, strict=True)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .095), ncol=2, frameon=False)
    fig.text(.06, .075, "Cada punto = un run real. Sin promedio de seeds, jitter, regresión ni ranking; puntos coincidentes pueden superponerse.", fontsize=10)
    fig.text(.06, .045, NOTICE[:NOTICE.index(" Cada punto")], fontsize=9, wrap=True)
    return fig


def scatter(ax, evidence: AssociationEvidence, metric: str, residual: str, class_id: int) -> None:
    selected = select_view(evidence, metric, residual, class_id)
    for strategy, color, marker in zip(STRATEGIES, COLORS, MARKERS, strict=True):
        part = selected.loc[selected.strategy.eq(strategy)].dropna(subset=["x", "y"])
        ax.scatter(part.x, part.y, color=color, marker=marker, alpha=.75, s=34, edgecolors="white", linewidths=.4)
    ax.set(xlabel=RESIDUAL_LABELS[residual], ylabel=METRIC_LABELS[metric], ylim=(-.02, 1.02))
    ax.grid(alpha=.15)
