"""Inferred index ordering and gaps for display only, never new cluster labels."""

from __future__ import annotations

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from flir_pipeline.explorer.models import PreviewPlan

COLORS = {"train": "#176b87", "val": "#bc6b13", "test": "#854cb0", "multiple": "#aa3945", "unassigned": "#74828c"}


def ordered_contents(contents: pd.DataFrame) -> pd.DataFrame:
    return contents.sort_values(["sequence_key", "frame_index", "content_id"], kind="stable", na_position="last").reset_index(drop=True)


def index_gaps(contents: pd.DataFrame) -> pd.DataFrame:
    """Unknown/conflicting provenance cannot become ordered playback."""
    known = contents.loc[contents.sequence_provenance_valid & contents.frame_index_valid].copy()
    known = ordered_contents(known)
    known["previous_index"] = known.groupby("sequence_key").frame_index.shift()
    known["index_gap"] = known.frame_index - known.previous_index
    return known


def preview_plans(contents: pd.DataFrame, clustering_space_id: str, cluster_id: int,
                  *, fps: int = 4, gap_threshold: int = 25, max_frames: int = 120,
                  width: int = 640) -> list[PreviewPlan]:
    """Separate sequences first, then optional gap cuts and bounded playback pages.

    These are display segments, never clusters. One representative occurrence
    per exact content prevents repeated historical copies from slowing playback.
    """
    if fps not in (2, 4, 8, 12) or gap_threshold < 0 or max_frames < 1 or not 64 <= width <= 1280:
        raise ValueError("Invalid visualization settings")
    if not contents.content_id.is_unique or not contents.cluster_id.eq(cluster_id).all():
        raise ValueError("A preview requires unique contents from one selected cluster")
    plans = []
    for key, group in index_gaps(contents).groupby("sequence_key", sort=True):
        cuts = group.index_gap.gt(gap_threshold).fillna(False).cumsum() if gap_threshold else pd.Series(0, index=group.index)
        for _, segment in group.groupby(cuts):
            for start in range(0, len(segment), max_frames):
                page = segment.iloc[start:start + max_frames]
                plans.append(PreviewPlan(clustering_space_id, int(cluster_id), str(key),
                                         tuple(page.representative_frame_id), tuple(page.content_id),
                                         tuple(map(int, page.frame_index)), gap_threshold, fps, width))
    return plans


def timeline_figure(contents: pd.DataFrame) -> Figure:
    """Facets share no sequence line; multi-split content has its own category."""
    known = index_gaps(contents)
    sequences = list(known.groupby("sequence_key", sort=True))
    figure = Figure(figsize=(10, max(2.8, 2 * len(sequences))), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(max(1, len(sequences)), 1, squeeze=False).ravel()
    for ax, (_, group) in zip(axes, sequences, strict=False):
        memberships = group.new_splits.map(lambda s: s[0] if len(s) == 1 else "multiple" if len(s) > 1 else "unassigned")
        for label in COLORS:
            selected = group.loc[memberships.eq(label)]
            if len(selected):
                ax.scatter(selected.frame_index, [0] * len(selected), s=28, color=COLORS[label], label=label, alpha=.8)
        title = f"{group.source_archive.iloc[0]} / {group.sequence_id.iloc[0]}"
        ax.set(title=title, xlabel="Índice de frame inferido", yticks=[])
        ax.legend(loc="upper right", ncols=4, fontsize=8)
        ax.spines[["left", "top", "right"]].set_visible(False)
    if not sequences:
        axes[0].text(.5, .5, "Sin índices inferidos utilizables", ha="center")
        axes[0].set_axis_off()
    return figure


def comparison_figure(rows: pd.DataFrame, left_label: str, right_label: str) -> Figure:
    """Occurrence lanes preserve duplicates at the same inferred index."""
    figure = Figure(figsize=(11, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    known = rows.dropna(subset=["possible_frame_index"])
    for side, offset in (("left", 1), ("right", 0)):
        for name in ("train", "val", "test"):
            selected = known.loc[known[side].eq(name)]
            ax.scatter(selected.possible_frame_index, [offset + {"train": .12, "val": 0, "test": -.12}[name]] * len(selected),
                       color=COLORS[name], s=22, alpha=.75, label=name if side == "left" else None)
    ax.set(yticks=[0, 1], yticklabels=[right_label, left_label], xlabel="Índice de frame inferido", ylim=(-.4, 1.4))
    ax.legend(ncols=3, loc="upper right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    return figure
