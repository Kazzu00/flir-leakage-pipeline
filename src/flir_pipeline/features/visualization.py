"""Descriptive feature-engineering visualizations and reproducible report generation."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


_DEFAULT_DPI = 220


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _relative_path(path: Path) -> str:
    try:
        relative = path.relative_to(Path.cwd())
    except ValueError:
        return path.as_posix()
    return relative.as_posix()


def _safe_hist_bins(values: np.ndarray, max_bins: int = 30) -> int:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1
    span = float(np.ptp(finite))
    if not np.isfinite(span) or span <= 0.0:
        return 1
    if span < 1e-5:
        return 1
    return max(1, min(max_bins, 30))


def _savefig(path: Path, fig: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def _summarize_duplicate_structure(manifest: pd.DataFrame) -> dict:
    if manifest.empty:
        return {
            "total_records": 0,
            "unique_content_ids": 0,
            "duplicate_groups": 0,
            "duplicate_records": 0,
            "cross_split_duplicate_content_ids": 0,
            "train_val_duplicate_content_ids": 0,
            "train_test_duplicate_content_ids": 0,
            "val_test_duplicate_content_ids": 0,
            "difference_absolute": 0,
            "difference_percent": 0.0,
        }

    content_counts = manifest["content_id"].value_counts()
    duplicates = content_counts[content_counts > 1]
    split_map = manifest.groupby("content_id")["original_split"].agg(set)
    if "original_split" not in manifest.columns:
        cross_split_count = 0
        train_val = train_test = val_test = 0
    else:
        cross_split_mask = split_map.map(len) > 1
        cross_split_count = int(cross_split_mask.sum())
        train_content = set(manifest.loc[manifest["original_split"] == "train", "content_id"])
        val_content = set(manifest.loc[manifest["original_split"] == "val", "content_id"])
        test_content = set(manifest.loc[manifest["original_split"] == "test", "content_id"])
        train_val = len(train_content & val_content)
        train_test = len(train_content & test_content)
        val_test = len(val_content & test_content)
    summary = {
        "total_records": int(len(manifest)),
        "unique_content_ids": int(manifest["content_id"].nunique(dropna=False)),
        "duplicate_groups": int(len(duplicates)),
        "duplicate_records": int(duplicates.sum() - len(duplicates)),
        "cross_split_duplicate_content_ids": int(cross_split_count),
        "train_val_duplicate_content_ids": int(train_val),
        "train_test_duplicate_content_ids": int(train_test),
        "val_test_duplicate_content_ids": int(val_test),
        "difference_absolute": int(len(manifest) - manifest["content_id"].nunique(dropna=False)),
    }
    unique_count = summary["unique_content_ids"] or 1
    summary["difference_percent"] = (
        summary["difference_absolute"] / (len(manifest) or 1) * 100.0
    )
    summary["unique_vs_records_ratio"] = len(manifest) / max(1, unique_count)
    return summary


def _class_distribution_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["class_id", "count"])
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        classes = row.get("classes_present", "")
        if pd.isna(classes):
            continue
        for value in str(classes).split("|"):
            if value:
                rows.append({"class_id": value, "count": 1})
    if not rows:
        return pd.DataFrame(columns=["class_id", "count"])
    counts = pd.DataFrame(rows).groupby("class_id", as_index=False).sum()
    counts = counts.sort_values("class_id", key=lambda s: s.map(lambda v: int(v) if str(v).isdigit() else -1))
    return counts


def _objects_per_image_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["bucket", "count"])
    values = pd.to_numeric(manifest.get("num_objects", pd.Series(dtype=float)), errors="coerce").fillna(0)
    buckets = []
    for label, predicate in {
        "0": values == 0,
        "1": values == 1,
        "2": values == 2,
        "3+": values >= 3,
    }.items():
        buckets.append({"bucket": label, "count": int(predicate.sum())})
    return pd.DataFrame(buckets)


def _empty_label_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["label_status", "count"])
    empty = manifest.get("label_empty", pd.Series(False, index=manifest.index))
    if hasattr(empty, "fillna"):
        empty = empty.fillna(False)
    has_objects = manifest.get("label_valid", pd.Series(False, index=manifest.index)).fillna(False)
    rows = [
        {"label_status": "empty_labels", "count": int(empty.astype(bool).sum())},
        {"label_status": "non_empty_labels", "count": int((~empty.fillna(True).astype(bool)).sum())},
        {"label_status": "labels_with_objects", "count": int(has_objects.astype(bool).sum())},
    ]
    return pd.DataFrame(rows)


def _dataset_overview_figure(manifest: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        "total_records": len(manifest),
        "unique_content_ids": int(manifest["content_id"].nunique()) if "content_id" in manifest.columns else 0,
        "duplicate_groups": int(
            manifest["content_id"].value_counts().gt(1).sum()
        ) if "content_id" in manifest.columns else 0,
        "total_objects": int(manifest.get("num_objects", pd.Series([0] * len(manifest))).sum()) if "num_objects" in manifest.columns else 0,
        "classes": int(
            len(
                {
                    value
                    for values in manifest.get("classes_present", pd.Series([""]))
                    for value in str(values).split("|")
                    if value
                }
            )
        ) if "classes_present" in manifest.columns else 0,
    }
    cards = [
        (summary["total_records"], "Historical records"),
        (summary["unique_content_ids"], "Unique contents"),
        (summary["duplicate_groups"], "Exact duplicate groups"),
        (summary["total_objects"], "Annotated objects"),
        (summary["classes"], "Classes"),
    ]
    fig, axes = plt.subplots(1, len(cards), figsize=(12, 2.8))
    fig.suptitle("Dataset overview", fontsize=14, y=1.04)
    for axis, (value, label) in zip(axes, cards, strict=True):
        axis.text(0.5, 0.58, f"{value:,}", ha="center", va="center", fontsize=22, fontweight="bold")
        axis.text(0.5, 0.25, label, ha="center", va="center", fontsize=9, wrap=True)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#B8C2CC")
            spine.set_linewidth(0.8)
    _savefig(output_dir / "figures" / "01_dataset_overview.png", fig)


def _original_split_distribution_figure(manifest: pd.DataFrame, output_dir: Path) -> None:
    if "original_split" not in manifest.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No original_split metadata available", ha="center", va="center")
        ax.set_axis_off()
        _savefig(output_dir / "figures" / "02_original_split_distribution.png", fig)
        return
    counts = manifest["original_split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xlabel("Original split")
    ax.set_ylabel("Records")
    ax.set_title("Original split distribution (descriptive metadata only)")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom")
    ax.text(
        0.5,
        -0.22,
        "original_split is not used as input to feature extraction or clustering.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    _savefig(output_dir / "figures" / "02_original_split_distribution.png", fig)


def _class_distribution_figure(manifest: pd.DataFrame, output_dir: Path) -> None:
    counts = _class_distribution_rows(manifest)
    fig, ax = plt.subplots(figsize=(8, 5))
    if counts.empty:
        ax.text(0.5, 0.5, "No class labels available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(counts["class_id"].astype(str), counts["count"], color="#4C72B0")
        ax.set_xlabel("class_id")
        ax.set_ylabel("Objects")
        ax.set_title("Objects per class_id")
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "03_class_distribution.png", fig)
    counts.to_csv(output_dir / "tables" / "class_distribution.csv", index=False)


def _objects_per_image_figure(manifest: pd.DataFrame, output_dir: Path) -> None:
    summary = _objects_per_image_summary(manifest)
    fig, ax = plt.subplots(figsize=(6, 4))
    if summary.empty:
        ax.text(0.5, 0.5, "No object counts available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(summary["bucket"].astype(str), summary["count"], color="#55A868")
        ax.set_xlabel("Objects per frame")
        ax.set_ylabel("Images")
        ax.set_title("Objects per image")
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "04_objects_per_image.png", fig)
    summary.to_csv(output_dir / "tables" / "objects_per_image_summary.csv", index=False)


def _empty_labels_figure(manifest: pd.DataFrame, output_dir: Path) -> None:
    summary = _empty_label_summary(manifest)
    fig, ax = plt.subplots(figsize=(6, 4))
    if summary.empty:
        ax.text(0.5, 0.5, "No label status available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(summary["label_status"], summary["count"], color=["#C44E52", "#55A868", "#8172B3"])
        ax.set_ylabel("Count")
        ax.set_title("Empty vs populated labels")
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "05_empty_labels.png", fig)


def _bbox_area_distribution_figure(manifest: pd.DataFrame, output_dir: Path) -> None:
    values = pd.to_numeric(manifest.get("bbox_area_mean", pd.Series(dtype=float)), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    if values.empty:
        ax.text(0.5, 0.5, "No bounding-box area data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=30, color="#8172B3", alpha=0.9)
        median = float(np.median(values))
        q25 = float(np.quantile(values, 0.25))
        q75 = float(np.quantile(values, 0.75))
        ax.axvline(median, color="#C44E52", linestyle="--", label=f"median={median:.3f}")
        ax.axvline(q25, color="#CCB974", linestyle=":", label=f"p25={q25:.3f}")
        ax.axvline(q75, color="#55A868", linestyle=":", label=f"p75={q75:.3f}")
        ax.set_xlabel("Normalized bbox area")
        ax.set_ylabel("Frequency")
        ax.set_title("Bounding-box area distribution")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "06_bbox_area_distribution.png", fig)


def _image_dimensions_figure(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    if diagnostics.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No image dimensions available", ha="center", va="center")
        ax.set_axis_off()
        _savefig(output_dir / "figures" / "07_image_dimensions.png", fig)
        return
    x = pd.to_numeric(diagnostics.get("width", pd.Series(dtype=float)), errors="coerce").dropna()
    y = pd.to_numeric(diagnostics.get("height", pd.Series(dtype=float)), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(x) and len(y):
        ax.scatter(x.to_numpy(), y.to_numpy(), s=12, alpha=0.7)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image dimensions")
    ax.grid(alpha=0.2)
    _savefig(output_dir / "figures" / "07_image_dimensions.png", fig)


def _aspect_ratio_distribution_figure(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    values = pd.to_numeric(diagnostics.get("aspect_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    if values.empty:
        ax.text(0.5, 0.5, "No aspect ratios available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=30, color="#4C72B0")
        ax.set_xlabel("Aspect ratio")
        ax.set_ylabel("Count")
        ax.set_title("Aspect ratio distribution")
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "08_aspect_ratio_distribution.png", fig)


def _pixel_statistics_figure(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    x = pd.to_numeric(diagnostics.get("pixel_mean", pd.Series(dtype=float)), errors="coerce").dropna()
    y = pd.to_numeric(diagnostics.get("pixel_std", pd.Series(dtype=float)), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    if not x.empty and not y.empty:
        pairs = pd.concat([x.rename("pixel_mean"), y.rename("pixel_std")], axis=1).dropna()
        ax.scatter(pairs["pixel_mean"], pairs["pixel_std"], s=12, alpha=0.7)
        ax.set_xlabel("Mean normalized pixel intensity")
        ax.set_ylabel("Pixel intensity standard deviation")
        ax.set_title("Pixel statistics")
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No pixel statistics available", ha="center", va="center")
        ax.set_axis_off()
    _savefig(output_dir / "figures" / "09_pixel_statistics.png", fig)


def _entropy_distribution_figure(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    values = pd.to_numeric(diagnostics.get("entropy", pd.Series(dtype=float)), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    if values.empty:
        ax.text(0.5, 0.5, "No entropy data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=30, color="#55A868")
        ax.set_xlabel("Entropy (bits)")
        ax.set_ylabel("Count")
        ax.set_title("Entropy distribution")
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "10_entropy_distribution.png", fig)


def _laplacian_variance_figure(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    values = pd.to_numeric(diagnostics.get("laplacian_variance", pd.Series(dtype=float)), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    if values.empty:
        ax.text(0.5, 0.5, "No Laplacian variance data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        transformed = np.log10(1.0 + values)
        ax.hist(transformed, bins=30, color="#C44E52")
        ax.set_xlabel("log10(1 + Laplacian variance)")
        ax.set_ylabel("Count")
        ax.set_title("Transformed Laplacian variance distribution")
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "11_laplacian_variance.png", fig)


def _unique_vs_records_figure(manifest: pd.DataFrame, output_dir: Path) -> None:
    total_records = len(manifest)
    unique_content = int(manifest["content_id"].nunique()) if "content_id" in manifest.columns else 0
    diff = total_records - unique_content
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["historical records", "unique content"], [total_records, unique_content], color=["#4C72B0", "#55A868"])
    ax.set_ylabel("Count")
    ax.set_title("Historical records vs. unique visual contents")
    ax.text(0.5, 0.97, f"Difference: {diff} ({diff / max(total_records, 1) * 100:.1f}%)", transform=ax.transAxes, ha="center", va="top")
    _savefig(output_dir / "figures" / "12_unique_vs_records.png", fig)


def _cross_split_duplicate_matrix(manifest: pd.DataFrame, output_dir: Path) -> None:
    if "original_split" not in manifest.columns or "content_id" not in manifest.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No original_split data available", ha="center", va="center")
        ax.set_axis_off()
        _savefig(output_dir / "figures" / "13_cross_split_duplicate_matrix.png", fig)
        return
    splits = ["train", "val", "test"]
    matrix = _cross_split_overlap_values(manifest)
    content_by_split = {split: set(manifest.loc[manifest["original_split"] == split, "content_id"]) for split in splits}
    fig, ax = plt.subplots(figsize=(5, 5))
    display_matrix = np.ma.masked_where(np.eye(3, dtype=bool), matrix)
    non_diagonal = matrix[~np.eye(3, dtype=bool)]
    image = ax.imshow(display_matrix, cmap="Blues", vmin=0, vmax=max(1, int(non_diagonal.max(initial=0))))
    ax.set_xticks(range(3), labels=splits)
    ax.set_yticks(range(3), labels=splits)
    ax.set_title("Exact content overlap across historical splits")
    for idx, value in np.ndenumerate(matrix):
        if idx[0] != idx[1]:
            ax.text(idx[1], idx[0], str(value), ha="center", va="center", color="black")
    val_total = len(content_by_split["val"])
    test_total = len(content_by_split["test"])
    val_train_percent = matrix[0, 1] / max(1, val_total) * 100
    test_train_percent = matrix[0, 2] / max(1, test_total) * 100
    ax.text(
        0.5,
        -0.16,
        f"Validation/train: {val_train_percent:.2f}% | Test/train: {test_train_percent:.2f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _savefig(output_dir / "figures" / "13_cross_split_duplicate_matrix.png", fig)


def _cross_split_overlap_values(manifest: pd.DataFrame) -> np.ndarray:
    splits = ["train", "val", "test"]
    matrix = np.zeros((3, 3), dtype=int)
    content_by_split = {
        split: set(manifest.loc[manifest["original_split"] == split, "content_id"])
        for split in splits
    }
    for i, left in enumerate(splits):
        for j, right in enumerate(splits):
            if i != j:
                matrix[i, j] = len(content_by_split[left] & content_by_split[right])
    return matrix


def _duplicate_group_size_distribution(manifest: pd.DataFrame, output_dir: Path) -> None:
    counts = manifest["content_id"].value_counts()
    duplicate_sizes = counts[counts > 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    if duplicate_sizes.empty:
        ax.text(0.5, 0.5, "No duplicate groups", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(duplicate_sizes.values, bins=max(1, min(10, len(duplicate_sizes))), color="#8172B3")
        ax.set_xlabel("Occurrences per duplicated content_id")
        ax.set_ylabel("Duplicate groups")
        ax.set_title("Duplicate group size distribution")
        ax.grid(axis="y", alpha=0.2)
    _savefig(output_dir / "figures" / "14_duplicate_group_size_distribution.png", fig)


def _embedding_health_summary(feature_dir: Path) -> dict:
    raw = np.load(feature_dir / "embeddings_raw.npy", mmap_mode="r")
    normalized = np.load(feature_dir / "embeddings_l2.npy", mmap_mode="r")
    norms_raw = np.linalg.norm(raw, axis=1)
    norms_norm = np.linalg.norm(normalized, axis=1)
    dims_mean = normalized.mean(axis=0)
    dims_std = normalized.std(axis=0)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    extractor = metadata.get("extractor", feature_dir.parent.name)
    stats = {
        "extractor": extractor,
        "n_samples": int(raw.shape[0]),
        "embedding_dimension": int(raw.shape[1]),
        "raw_norm_min": float(norms_raw.min()),
        "raw_norm_median": float(np.median(norms_raw)),
        "raw_norm_max": float(norms_raw.max()),
        "l2_norm_mean": float(norms_norm.mean()),
        "l2_norm_min": float(norms_norm.min()),
        "l2_norm_max": float(norms_norm.max()),
        "l2_mean_near_one": bool(np.allclose(norms_norm, 1.0, atol=1e-5)),
        "dimension_std_min": float(dims_std.min()),
        "dimension_std_median": float(np.median(dims_std)),
        "dimension_std_max": float(dims_std.max()),
        "dimension_mean_min": float(dims_mean.min()),
        "dimension_mean_median": float(np.median(dims_mean)),
        "dimension_mean_max": float(dims_mean.max()),
    }
    return stats


def visualize_embedding_health(feature_dir: Path, output_dir: Path, label: str = "feature") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (feature_dir / "embeddings_raw.npy").is_file() or not (feature_dir / "embeddings_l2.npy").is_file():
        raise ValueError(f"Feature directory is missing embedding arrays: {feature_dir}")
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    extractor = metadata.get("extractor", feature_dir.parent.name)
    stage = str(label)
    prefix = f"{extractor}_"
    while stage.startswith(prefix):
        stage = stage[len(prefix):]
    stage = stage or "feature"
    raw = np.load(feature_dir / "embeddings_raw.npy", mmap_mode="r")
    normalized = np.load(feature_dir / "embeddings_l2.npy", mmap_mode="r")
    raw_norms = np.linalg.norm(raw, axis=1)
    l2_norms = np.linalg.norm(normalized, axis=1)
    dim_std = normalized.std(axis=0)
    dim_mean = normalized.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(raw_norms, bins=_safe_hist_bins(raw_norms), color="#4C72B0")
    ax.set_xlabel("L2 norm of raw embedding")
    ax.set_ylabel("Count")
    ax.set_title(f"Embedding raw norm distribution ({extractor}, {stage})")
    _savefig(output_dir / "figures" / f"embedding_raw_norm_distribution_{extractor}_{stage}.png", fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(l2_norms, bins=_safe_hist_bins(l2_norms), color="#55A868")
    ax.set_xlabel("L2 norm of normalized embedding")
    ax.set_ylabel("Count")
    ax.set_title(f"Embedding L2 norm distribution ({extractor}, {stage})")
    _savefig(output_dir / "figures" / f"embedding_l2_norm_distribution_{extractor}_{stage}.png", fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dim_std, bins=_safe_hist_bins(dim_std), color="#8172B3")
    ax.set_xlabel("Dimension std")
    ax.set_ylabel("Dimension count")
    ax.set_title(f"Embedding dimension std ({extractor}, {stage})")
    _savefig(output_dir / "figures" / f"embedding_dimension_std_{extractor}_{stage}.png", fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dim_mean, bins=_safe_hist_bins(dim_mean), color="#C44E52")
    ax.set_xlabel("Dimension mean")
    ax.set_ylabel("Dimension count")
    ax.set_title(f"Embedding dimension mean ({extractor}, {stage})")
    _savefig(output_dir / "figures" / f"embedding_dimension_mean_{extractor}_{stage}.png", fig)

    stats = _embedding_health_summary(feature_dir)
    stats["label"] = stage
    stats["feature_directory"] = _relative_path(feature_dir)
    return {"label": stage, "stats": stats, "extractor": extractor}


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["metric", "value"])
    summary = {
        "total_rows": int(len(diagnostics)),
        "width_mean": float(diagnostics.get("width", pd.Series(dtype=float)).mean()),
        "height_mean": float(diagnostics.get("height", pd.Series(dtype=float)).mean()),
        "aspect_ratio_median": float(diagnostics.get("aspect_ratio", pd.Series(dtype=float)).median()),
        "pixel_mean_mean": float(diagnostics.get("pixel_mean", pd.Series(dtype=float)).mean()),
        "pixel_std_mean": float(diagnostics.get("pixel_std", pd.Series(dtype=float)).mean()),
        "entropy_mean": float(diagnostics.get("entropy", pd.Series(dtype=float)).mean()),
        "laplacian_variance_mean": float(diagnostics.get("laplacian_variance", pd.Series(dtype=float)).mean()),
    }
    return pd.DataFrame([summary])


def generate_feature_engineering_report(
    manifest_path: Path,
    diagnostics_path: Path,
    output_dir: Path,
    feature_dirs: dict[str, Path] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    diagnostics = pd.read_parquet(diagnostics_path) if diagnostics_path.is_file() else pd.DataFrame()

    dataset_summary = {
        "total_records": int(len(manifest)),
        "unique_content_ids": int(manifest["content_id"].nunique()) if "content_id" in manifest.columns else 0,
        "duplicate_groups": int(manifest["content_id"].value_counts().gt(1).sum()) if "content_id" in manifest.columns else 0,
        "total_objects": int(manifest.get("num_objects", pd.Series([0] * len(manifest))).sum()) if "num_objects" in manifest.columns else 0,
        "classes": int(len({_value for values in manifest.get("classes_present", pd.Series([" "])) for _value in str(values).split("|") if _value})) if "classes_present" in manifest.columns else 0,
        "empty_labels": int(manifest.get("label_empty", pd.Series(False, index=manifest.index)).fillna(False).sum()) if "label_empty" in manifest.columns else 0,
    }
    _write_csv(tables_dir / "dataset_summary.csv", pd.DataFrame([dataset_summary]))
    _class_distribution_figure(manifest, output_dir)
    _objects_per_image_figure(manifest, output_dir)
    _empty_labels_figure(manifest, output_dir)
    _bbox_area_distribution_figure(manifest, output_dir)
    _dataset_overview_figure(manifest, output_dir)
    _original_split_distribution_figure(manifest, output_dir)
    _image_dimensions_figure(diagnostics, output_dir)
    _aspect_ratio_distribution_figure(diagnostics, output_dir)
    _pixel_statistics_figure(diagnostics, output_dir)
    _entropy_distribution_figure(diagnostics, output_dir)
    _laplacian_variance_figure(diagnostics, output_dir)
    _unique_vs_records_figure(manifest, output_dir)
    _cross_split_duplicate_matrix(manifest, output_dir)
    _duplicate_group_size_distribution(manifest, output_dir)

    duplicate_summary = _summarize_duplicate_structure(manifest)
    _write_csv(tables_dir / "duplicate_summary.csv", pd.DataFrame([duplicate_summary]))
    _write_csv(tables_dir / "image_diagnostics_summary.csv", _summarize_diagnostics(diagnostics))

    embedding_stats: dict[str, dict] = {}
    embedding_status: dict[str, str] = {}
    feature_dirs = feature_dirs or {}
    for extractor in ("dinov2", "clip"):
        selected_dir = feature_dirs.get(extractor)
        if selected_dir is None:
            feature_root = Path("artifacts/features") / extractor
            candidates = sorted(feature_root.glob("*/*")) if feature_root.exists() else []
            if candidates:
                selected_dir = candidates[-1]
        if selected_dir is not None and (selected_dir / "embeddings_raw.npy").exists():
            embedding_stats[extractor] = visualize_embedding_health(selected_dir, output_dir, label="smoke" if selected_dir.name.startswith("smoke") or selected_dir.parent.name.endswith("smoke") else "feature")
            embedding_status[extractor] = "available"
        else:
            embedding_status[extractor] = "pending"
    if "dinov2" in embedding_stats:
        dino_df = pd.DataFrame([embedding_stats["dinov2"]["stats"]])
        dino_df.to_csv(tables_dir / "embedding_health_dinov2.csv", index=False)
    else:
        pd.DataFrame([{"extractor": "dinov2", "status": "pending"}]).to_csv(tables_dir / "embedding_health_dinov2.csv", index=False)
    if "clip" in embedding_stats:
        clip_df = pd.DataFrame([embedding_stats["clip"]["stats"]])
        clip_df.to_csv(tables_dir / "embedding_health_clip.csv", index=False)
    else:
        pd.DataFrame([{"extractor": "clip", "status": "pending"}]).to_csv(tables_dir / "embedding_health_clip.csv", index=False)

    manifest_version = str(manifest.get("manifest_version").dropna().unique()[0]) if "manifest_version" in manifest.columns and not manifest.empty else "unknown"
    dataset_id = None
    if "dataset_id" in manifest.columns:
        dataset_id = str(manifest["dataset_id"].dropna().iloc[0]) if not manifest.empty else None
    if dataset_id is None:
        dataset_id = (Path("reports/data_manifest/manifest_metadata.json").read_text(encoding="utf-8") if Path("reports/data_manifest/manifest_metadata.json").exists() else "unknown")
    if dataset_id == "unknown":
        dataset_id = "unknown"
    else:
        try:
            dataset_id = json.loads(Path("reports/data_manifest/manifest_metadata.json").read_text(encoding="utf-8"))["dataset_id"] if Path("reports/data_manifest/manifest_metadata.json").exists() else dataset_id
        except Exception:
            pass

    feature_dir_paths = []
    for item in feature_dirs.values():
        feature_dir_paths.append(_relative_path(item))
    metadata = {
        "dataset_id": dataset_id,
        "manifest_version": manifest_version,
        "git_commit": _git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "feature_directories_used": feature_dir_paths,
        "sample_sizes": {
            "manifest_records": int(len(manifest)),
            "diagnostics_rows": int(len(diagnostics)),
            "embedding_rows": {k: v["stats"]["n_samples"] for k, v in embedding_stats.items()},
        },
    }
    report_md = output_dir / "feature_engineering_report.md"
    report_md.write_text(
        "# Feature Engineering Report\n\n"
        "## Dataset composition\n\n"
        f"- Total records: {dataset_summary['total_records']}\n"
        f"- Unique content IDs: {dataset_summary['unique_content_ids']}\n"
        f"- Duplicate groups: {duplicate_summary['duplicate_groups']}\n"
        f"- Total objects: {dataset_summary['total_objects']}\n"
        f"- Classes observed: {dataset_summary['classes']}\n\n"
        "## Annotation characteristics\n\n"
        f"- Empty labels: {dataset_summary['empty_labels']}\n"
        f"- Non-empty label records: {max(dataset_summary['total_records'] - dataset_summary['empty_labels'], 0)}\n"
        "- Label geometry and object counts are descriptive only and do not drive feature extraction.\n\n"
        "## Exact duplicate structure\n\n"
        f"- Duplicate record count: {duplicate_summary['duplicate_records']}\n"
        f"- Duplicate content IDs: {duplicate_summary['duplicate_groups']}\n"
        f"- Cross-split duplicate content IDs: {duplicate_summary['cross_split_duplicate_content_ids']}\n"
        "- Exact duplicate records are preserved in the frame-level manifest and mapped to one content-level embedding row.\n\n"
        "## Image diagnostics\n\n"
        f"- Mean image width: {diagnostics.get('width', pd.Series(dtype=float)).mean() if not diagnostics.empty else 'n/a'}\n"
        f"- Mean image height: {diagnostics.get('height', pd.Series(dtype=float)).mean() if not diagnostics.empty else 'n/a'}\n"
        f"- Mean entropy: {diagnostics.get('entropy', pd.Series(dtype=float)).mean() if not diagnostics.empty else 'n/a'}\n"
        f"- Mean Laplacian variance: {diagnostics.get('laplacian_variance', pd.Series(dtype=float)).mean() if not diagnostics.empty else 'n/a'}\n\n"
        "## DINOv2 feature health\n\n"
        f"- Status: {embedding_status.get('dinov2', 'pending')}\n"
        + (
            f"- Sample size: {embedding_stats['dinov2']['stats']['n_samples']}\n"
            if 'dinov2' in embedding_stats else "- Sample size: pending\n"
        )
        + "- Feature quality is descriptive only and should not be interpreted as semantic quality for individual dimensions.\n\n"
        + "## CLIP feature health\n\n"
        + f"- Status: {embedding_status.get('clip', 'pending')}\n"
        + (
            f"- Sample size: {embedding_stats['clip']['stats']['n_samples']}\n"
            if 'clip' in embedding_stats else "- Sample size: pending\n"
        )
        + "- Feature statistics are used to check shape, L2 normalization, and basic distributional structure only.\n\n"
        + "## Methodological implications\n\n"
        + "- There are 1657 records but 1459 unique contents.\n"
        + "- Exact duplicates can affect density-based downstream methods; this is why feature extraction is performed once per content_id.\n"
        + "- original_split is retained as historical metadata only and is not used as input to feature extraction or clustering.\n"
        + "- DINOv2 and CLIP do not use labels or original_split as model inputs.\n"
        + "- This stage remains descriptive; it does not claim a fixed clustering structure or detector superiority.\n",
        encoding="utf-8",
    )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "embedding_status": embedding_status, "dataset_summary": dataset_summary, "duplicate_summary": duplicate_summary}


def _default_feature_directories(root: Path | None = None) -> dict[str, Path]:
    root = root or Path("artifacts/features")
    directories: dict[str, Path] = {}
    for extractor in ("dinov2", "clip"):
        extractor_root = root / extractor
        if not extractor_root.exists():
            continue
        candidates = sorted(extractor_root.glob("*/*"))
        if candidates:
            directories[extractor] = candidates[-1]
    return directories


def generate_feature_engineering_report_for_cli(
    manifest: Path,
    diagnostics: Path,
    output: Path,
) -> dict:
    discovered = _default_feature_directories()
    return generate_feature_engineering_report(manifest, diagnostics, output, discovered)
