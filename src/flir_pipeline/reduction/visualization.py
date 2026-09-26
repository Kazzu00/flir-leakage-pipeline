"""Posterior interpretation of verified reductions; no fitting or label inference."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flir_pipeline.reduction.benchmark import verify_benchmark  # noqa: E402
from flir_pipeline.reduction.storage import reduction_provenance  # noqa: E402
from flir_pipeline.similarity.storage import (  # noqa: E402
    file_sha256,
    read_json,
    verify_similarity_directory,
    write_json,
)

FIGURE_NAMES = (
    "01_reduction_quality_dinov2.png", "02_reduction_quality_clip.png",
    "03_pacmap_dinov2_reference.png", "04_pacmap_clip_reference.png",
    "05_tsne_dinov2_reference.png", "06_tsne_clip_reference.png",
    "07_reduction_stability_dinov2.png", "08_reduction_stability_clip.png",
    "09_temporal_overlay_dinov2_pacmap.png", "10_temporal_overlay_clip_pacmap.png",
    "11_historical_split_overlay.png",
)
METHOD_COLORS = {"tsne": "#246b8e", "pacmap": "#b4552d"}
METHOD_NAMES = {"tsne": "t-SNE", "pacmap": "PaCMAP"}
ENCODER_NAMES = {"dinov2": "DINOv2", "clip": "CLIP"}
SPLIT_COLORS = {"train only": "#a0aeb4", "val only": "#4f82c0", "test only": "#66a887",
                "train+val": "#9b51aa", "train+test": "#d47929", "val+test": "#755542",
                "train+val+test": "#be3b56", "unknown": "#303030"}


def split_category(value: str) -> str:
    """Preserve all historical memberships instead of forcing one split per content."""
    members = set(json.loads(value))
    if not members:
        return "unknown"
    if not members <= {"train", "val", "test"}:
        raise ValueError("Unrecognized historical split membership")
    label = "+".join(s for s in ("train", "val", "test") if s in members)
    return label + " only" if len(members) == 1 else label


def temporal_window(provenance: pd.DataFrame, size: int = 30) -> pd.DataFrame:
    """Center a window on the longest consecutive run of the first sorted sequence.

    Selection uses posterior provenance only and is independent of coordinates.
    Duplicate nominal indices are ambiguous and excluded, never arbitrarily joined.
    """
    if size < 2 or not provenance.content_id.is_unique:
        raise ValueError("Temporal windows require unique contents and size >= 2")
    known = provenance.loc[provenance.sequence_key.notna() & provenance.frame_index.notna()].copy()
    for sequence in sorted(known.sequence_key.unique()):
        rows = known.loc[known.sequence_key == sequence].sort_values(["frame_index", "content_id"])
        rows = rows.loc[~rows.frame_index.duplicated(keep=False)]
        if rows.empty:
            continue
        blocks = rows.frame_index.diff().ne(1).fillna(True).cumsum()
        stretches = [part for _, part in rows.groupby(blocks, sort=True)]
        longest = sorted(stretches, key=lambda x: (-len(x), int(x.frame_index.iloc[0])))[0]
        if len(longest) < 2:
            continue
        count = min(size, len(longest))
        start = (len(longest)-count)//2
        return longest.iloc[start:start+count].copy()
    raise ValueError("No unambiguous consecutive inferred indices for a temporal example")


def _save(fig, output: Path) -> None:
    fig.tight_layout(h_pad=2.5)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _axis(ax) -> None:
    ax.set(xlabel="Coordenada 1 (escala propia)", ylabel="Coordenada 2 (escala propia)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)


def _quality_plot(runs: pd.DataFrame, encoder: str, output: Path) -> None:
    groups = list(runs.groupby(["method", "configuration_id"], sort=False))
    labels = [f"{METHOD_NAMES[m]}\n{g.label.iloc[0]}" for (m, _), g in groups]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for axis, metric, title in zip(axes[:2], ("trustworthiness", "continuity"),
                                   ("¿Cuántos vecinos intrusos se penalizan?", "¿Qué relaciones locales se omiten?"), strict=True):
        for k, color, shift in ((5, "#246b8e", -.19), (10, "#b4552d", 0), (20, "#659047", .19)):
            values = [g[f"{metric}@{k}"] for _, g in groups]
            axis.errorbar(np.arange(len(groups))+shift, [v.mean() for v in values],
                          yerr=[v.std(ddof=0) for v in values], fmt="o", capsize=3, color=color, label=f"k={k}")
        axis.set(title=title, ylabel=metric.title())
        axis.legend(fontsize=8)
    for metric, title, color, shift in (("jaccard@20_mean_jaccard", "Jaccard@20", "#246b8e", -.10),
                                        ("spearman_distance", "Spearman (100k pares)", "#b4552d", .10)):
        values = [g[metric] for _, g in groups]
        axes[2].errorbar(np.arange(len(groups))+shift, [v.mean() for v in values],
                        yerr=[v.std(ddof=0) for v in values], fmt="o", capsize=3, color=color, label=title)
    axes[2].set(title="Preservación de conjuntos y orden de distancias", ylabel="Métrica (distintas definiciones)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(np.arange(len(groups)), labels, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle(f"{ENCODER_NAMES[encoder]} · media ± DE entre tres semillas; ejes Y con escala propia", fontsize=13)
    _save(fig, output)


def _stability_plot(runs: pd.DataFrame, stability: pd.DataFrame, encoder: str, output: Path) -> None:
    configs = runs.drop_duplicates("configuration_id")
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, color, shift in ((5, "#246b8e", -.2), (10, "#b4552d", 0), (20, "#659047", .2)):
        selected = stability.loc[stability.k == k].set_index("configuration_id").loc[configs.configuration_id]
        positions = np.arange(len(configs))+shift
        ax.vlines(positions, selected.Q1, selected.Q3, color=color, linewidth=3, alpha=.55)
        ax.scatter(positions, selected["mean"], color=color, s=35, label=f"k={k}")
    labels = [f"{METHOD_NAMES[r.method]}\n{r.label}" for r in configs.itertuples()]
    ax.set_xticks(np.arange(len(configs)), labels, fontsize=9)
    ax.set(ylim=(0, 1.04), ylabel="Jaccard entre conjuntos de vecinos",
           title=f"{ENCODER_NAMES[encoder]} · estabilidad entre semillas 0/1, 0/2 y 1/2")
    ax.text(.5, -.2, "Punto: media; línea: Q1–Q3 por contenido y par de semillas. No son intervalos de confianza.",
            ha="center", transform=ax.transAxes, fontsize=9)
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=.18)
    _save(fig, output)


def generate_reduction_report(benchmarks: dict[str, Path], similarities: dict[str, Path],
                              output: Path = Path("reports/reduction")) -> dict:
    """Build derived figures/tables only after validating all explicit sources."""
    if set(benchmarks) != {"dinov2", "clip"} or set(similarities) != set(benchmarks):
        raise ValueError("The review requires both explicitly selected encoders")
    if any(read_json(path/"metadata.json").get("provenance_mode") == "sampled_video_grid" for path in similarities.values()):
        raise ValueError("This historical sequence/split review is unavailable for sampled video; numerical reduction remains supported")
    figures, tables = output/"figures", output/"tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    combined = {name: [] for name in ("runs", "configuration_summary", "configuration_stability", "candidates")}
    feature_rows, parameter_rows, candidate_rows, split_counts, temporal_rows = [], [], [], [], []
    projections, source_metadata = {}, {}
    shared_content_ids, shared_window_ids = None, None
    for encoder in ("dinov2", "clip"):
        benchmark, similarity = benchmarks[encoder], similarities[encoder]
        if not verify_benchmark(benchmark)["quality_valid"] or not verify_similarity_directory(similarity)["quality_valid"]:
            raise ValueError("Report sources failed verification")
        meta, sim = read_json(benchmark/"metadata.json"), read_json(similarity/"metadata.json")
        if (meta["extractor"] != encoder or any(meta[k] != sim[k] for k in ("dataset_id", "feature_space_id", "extractor"))
                or meta["reference_similarity_space_id"] != sim["similarity_space_id"]
                or meta["input_signatures"]["similarity"]["metadata.json"] != file_sha256(similarity/"metadata.json")):
            raise ValueError("Reduction and posterior similarity provenance do not correspond")
        data = {name: pd.read_csv(benchmark/f"{name}.csv") for name in combined}
        if set(data["candidates"].method) != {"tsne", "pacmap"} or len(data["candidates"]) != 2:
            raise ValueError("One verified reference per method is required")
        for name, frame in data.items():
            combined[name].append(frame.assign(encoder=encoder))
        provenance = pd.read_parquet(similarity/"content_provenance.parquet")
        if shared_content_ids is not None and set(provenance.content_id) != shared_content_ids:
            raise ValueError("Encoders do not describe the same content population")
        shared_content_ids = set(provenance.content_id)
        window = temporal_window(provenance)
        if shared_window_ids is not None and window.content_id.tolist() != shared_window_ids:
            raise ValueError("Temporal examples must refer to the same contents in both encoders")
        shared_window_ids = window.content_id.tolist()
        temporal_rows.append(window.assign(encoder=encoder))
        split_counts.extend({"encoder": encoder, "membership": name, "contents": int(count)}
                            for name, count in provenance.split_membership_set.map(split_category).value_counts().items())
        source_metadata[encoder] = {"benchmark_id": meta["benchmark_id"], "benchmark_metadata_sha256": file_sha256(benchmark/"metadata.json"),
                                    "similarity_space_id": sim["similarity_space_id"], "similarity_metadata_sha256": file_sha256(similarity/"metadata.json")}
        for row in data["runs"].itertuples():
            run = benchmark.parents[1]/row.method/row.reduction_space_id
            metadata = read_json(run/"metadata.json")
            parameter_rows.append({"encoder": encoder, "method": row.method, "label": row.label, "seed": row.seed,
                                   "reduction_space_id": row.reduction_space_id,
                                   "hyperparameters": json.dumps(metadata["hyperparameters"], sort_keys=True),
                                   "library": metadata["library"], "library_version": metadata["library_version"],
                                   "learning_rate_effective": metadata.get("learning_rate_effective"),
                                   "N": metadata["N"], "input_dimension": metadata["input_dimension"], "output_dimension": metadata["output_dimension"]})
        sample = read_json(benchmark.parents[1]/data["runs"].iloc[0].method/data["runs"].iloc[0].reduction_space_id/"metadata.json")
        feature_rows.append({"encoder": encoder, "model_id": sample["model_id"], "feature_space_id": sample["feature_space_id"],
                             "similarity_space_id": sim["similarity_space_id"], "N": sample["N"], "input_dimension": sample["input_dimension"],
                             "input_representation": sample["input_representation"], "pooling": sample["pooling_strategy"],
                             "runs": len(data["runs"]), "python_version": sample["python_version"]})
        offset = 0 if encoder == "dinov2" else 1
        _quality_plot(data["runs"], encoder, figures/FIGURE_NAMES[offset])
        _stability_plot(data["runs"], data["configuration_stability"], encoder, figures/FIGURE_NAMES[6+offset])
        for selected in data["candidates"].itertuples():
            run = benchmark.parents[1]/selected.method/selected.reduction_space_id
            coords = np.load(run/"coordinates.npy", allow_pickle=False)
            index = pd.read_parquet(run/"content_index.parquet")
            if coords.shape[1] != 2:
                raise ValueError("This principal report renders 2D reductions only")
            aligned = index[["content_id", "embedding_row"]].merge(provenance.drop(columns="embedding_row", errors="ignore"), on="content_id", how="left", validate="one_to_one").sort_values("embedding_row")
            if len(aligned) != len(provenance) or aligned.sequence_key.isna().all():
                raise ValueError("Missing posterior provenance for report coordinates")
            projections[(encoder, selected.method)] = (coords, aligned, selected)
            metrics = read_json(run/"metrics.json")
            stability = data["configuration_stability"].loc[(data["configuration_stability"].configuration_id == selected.configuration_id) & (data["configuration_stability"].k == 20)].iloc[0]
            candidate_rows.append({"encoder": encoder, "method": selected.method, "label": selected.label, "seed": selected.seed,
                                   "reduction_space_id": selected.reduction_space_id, "trustworthiness@20": metrics["trustworthiness@20"],
                                   "continuity@20": metrics["continuity@20"], "jaccard@20": metrics["jaccard@20_mean_jaccard"],
                                   "stability@20": stability["mean"], "spearman": metrics["spearman_distance"],
                                   "fit_seconds": read_json(run/"metadata.json")["fit_seconds"], "mean_criterion_rank": selected.mean_criterion_rank})
            fig, ax = plt.subplots(figsize=(9, 6.7))
            ax.scatter(*coords.T, s=7, alpha=.65, color=METHOD_COLORS[selected.method], edgecolors="none")
            _axis(ax)
            ax.set_title(f"{ENCODER_NAMES[encoder]} · {METHOD_NAMES[selected.method]} · {selected.label} · semilla 0\n"
                         f"N={len(coords)} · T@20={metrics['trustworthiness@20']:.4f} · Jaccard@20={metrics['jaccard@20_mean_jaccard']:.4f}")
            fig.text(.5, .01, "Referencia exploratoria seleccionada por métricas; no son etiquetas de agrupamiento.", ha="center", fontsize=9)
            figure_index = (2 if selected.method == "pacmap" else 4)+offset
            _save(fig, figures/FIGURE_NAMES[figure_index])
        coords, aligned, selected = projections[(encoder, "pacmap")]
        example = aligned.set_index("content_id").loc[window.content_id]
        xy = coords[example.embedding_row.to_numpy()]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(*coords.T, s=5, color="#b9c2c7", alpha=.32)
        ax.plot(*xy.T, color="#454e57", alpha=.7, lw=1)
        points = ax.scatter(*xy.T, c=example.frame_index.to_numpy(dtype=int), cmap="viridis", s=36, zorder=3)
        ax.annotate("Inicio", xy[0], xytext=(6, 7), textcoords="offset points", fontsize=9)
        ax.annotate("Fin", xy[-1], xytext=(6, -12), textcoords="offset points", fontsize=9)
        fig.colorbar(points, ax=ax, label="Índice de frame inferido; no representa segundos")
        _axis(ax)
        ax.set_title(f"{ENCODER_NAMES[encoder]} · PaCMAP · interpretación posterior\n"
                     f"{window.sequence_id.iloc[0]} · índices {int(window.frame_index.iloc[0])}–{int(window.frame_index.iloc[-1])}")
        fig.text(.5, .01, "Ventana central del tramo consecutivo más largo de la primera secuencia; selección independiente de la proyección.", ha="center", fontsize=8)
        _save(fig, figures/FIGURE_NAMES[8+offset])
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    legend_handles = {}
    for ax, ((encoder, method), (coords, aligned, selected)) in zip(axes.ravel(), projections.items(), strict=True):
        categories = aligned.split_membership_set.map(split_category)
        for category, color in SPLIT_COLORS.items():
            mask = categories.eq(category).to_numpy()
            if mask.any():
                legend_handles[category] = ax.scatter(*coords[mask].T, s=10 if "+" in category else 6,
                                                     color=color, alpha=.8, label=category, edgecolors="none")
        _axis(ax)
        ax.set_title(f"{ENCODER_NAMES[encoder]} · {METHOD_NAMES[method]} · {selected.label}")
    fig.suptitle("Pertenencia histórica por contenido · categorías de múltiples splits conservadas", fontsize=14)
    fig.legend(legend_handles.values(), legend_handles.keys(), loc="lower center", ncol=len(legend_handles), fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .97))
    fig.savefig(figures/FIGURE_NAMES[10], dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for name, frames in combined.items():
        pd.concat(frames, ignore_index=True).to_csv(tables/f"{name}.csv", index=False)
    for name, rows in (("input_spaces", feature_rows), ("effective_parameters", parameter_rows),
                       ("candidate_reference_metrics", candidate_rows), ("historical_memberships", split_counts)):
        pd.DataFrame(rows).to_csv(tables/f"{name}.csv", index=False)
    pd.concat(temporal_rows, ignore_index=True).to_parquet(tables/"temporal_example.parquet", index=False)
    runs = pd.concat(combined["runs"], ignore_index=True)
    timing = runs.groupby(["encoder", "method"]).agg(runs=("seed", "size"), fit_total_seconds=("fit_seconds", "sum"),
                                                      fit_min_seconds=("fit_seconds", "min"), fit_median_seconds=("fit_seconds", "median"),
                                                      fit_max_seconds=("fit_seconds", "max"), backend_total_seconds=("backend_total_seconds", "sum"),
                                                      evaluation_total_seconds=("evaluation_seconds", "sum")).reset_index()
    timing.to_csv(tables/"timings.csv", index=False)
    metadata = {**reduction_provenance(), "artifact_kind": "reduction_review", "sources": source_metadata,
                "N": len(shared_content_ids), "run_count": len(runs), "figure_count": len(FIGURE_NAMES),
                "temporal_example_size": len(shared_window_ids), "temporal_selection": "first_sorted_sequence_longest_consecutive_run_center_30_v1",
                "output_sha256": {p.relative_to(output).as_posix(): file_sha256(p) for p in [*(figures/n for n in FIGURE_NAMES), *tables.glob("*")]}}
    write_json(output/"report_metadata.json", metadata)
    return metadata
