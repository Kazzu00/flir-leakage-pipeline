"""Descriptive tables and local-only figures from verified similarity artifacts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flir_pipeline.features.preprocessing import decode_zip_image  # noqa: E402
from flir_pipeline.similarity.comparison import compare_neighbor_spaces  # noqa: E402
from flir_pipeline.similarity.storage import (  # noqa: E402
    TABLE_NAMES,
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    verify_similarity_directory,
    write_json,
)

FIGURE_NAMES = (
    "01_global_similarity_distribution_dinov2.png",
    "02_global_similarity_distribution_clip.png",
    "03_rank1_similarity_comparison.png",
    "04_same_vs_different_sequence_similarity.png",
    "05_similarity_vs_frame_delta_dinov2.png",
    "06_similarity_vs_frame_delta_clip.png",
    "07_cross_split_high_similarity_quantiles.png",
    "08_neighbor_agreement_dinov2_clip.png",
    "nearest_neighbors_dinov2.png",
    "nearest_neighbors_clip.png",
    "maximum_similarity_pairs.png",
)
COLORS = {"dinov2": "#25767b", "clip": "#b76134"}
DISPLAY = {"dinov2": "DINOv2", "clip": "CLIP"}


def _save(fig, path: Path) -> None:
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _read_verified_image(archive: zipfile.ZipFile, row: pd.Series) -> Image.Image:
    """Read only selected original bytes, verify identity, decode in memory."""
    if Path(archive.filename).name != row.source_archive:
        raise ValueError("Selected image belongs to a different source archive")
    data = archive.read(row.source_member_path)
    if hashlib.sha256(data).hexdigest() != row.image_sha256:
        raise ValueError("Selected image no longer matches its content index")
    return decode_zip_image(archive, row.source_member_path).convert("RGB")


def _image_caption(row: pd.Series, prefix: str, delta=None) -> str:
    sequence = row.sequence_id or "desconocida"
    index = "?" if pd.isna(row.frame_index) else str(int(row.frame_index))
    gap = "n/a" if delta is None or pd.isna(delta) else str(int(delta))
    splits = ",".join(json.loads(row.split_membership_set)) or "?"
    return f"{prefix}\n{sequence} · índice {index}\nΔ={gap} · {{{splits}}}"


def render_neighbor_grid(contents: pd.DataFrame, neighbors: pd.DataFrame, queries: list[str],
                         archive: zipfile.ZipFile, output: Path, encoder: str) -> pd.DataFrame:
    """Render shared seeded queries + top five, retaining private selection IDs locally."""
    indexed = contents.set_index("content_id", drop=False)
    fig, axes = plt.subplots(len(queries), 6, figsize=(19, 3.2*len(queries)), squeeze=False)
    selection = []
    for i, query in enumerate(queries):
        group = neighbors.loc[neighbors.query_content_id == query].sort_values("neighbor_rank").head(5)
        if len(group) != 5:
            raise ValueError("Visual review requires at least five neighbors")
        rows = [(query, 0, None, None)] + [(r.neighbor_content_id, r.neighbor_rank, r.cosine_similarity, r.frame_delta) for r in group.itertuples()]
        for ax, (content_id, rank, cosine, delta) in zip(axes[i], rows, strict=True):
            row = indexed.loc[content_id]
            ax.imshow(_read_verified_image(archive, row))
            prefix = f"Consulta {i+1}" if rank == 0 else f"Vecino {rank} · cos={cosine:.4f}"
            ax.set_title(_image_caption(row, prefix, delta), fontsize=8)
            ax.axis("off")
            selection.append({"extractor": encoder, "query_ordinal": i+1, "query_content_id": query,
                              "neighbor_rank": rank, "displayed_content_id": content_id})
    fig.suptitle(f"{DISPLAY[encoder]} · consultas compartidas, muestreo reproducible\nSecuencia/índice inferidos; splits como metadata posterior", fontsize=13)
    _save(fig, output)
    return pd.DataFrame(selection)


def _verify_comparison(directory: Path, spaces: dict) -> pd.DataFrame:
    """Bind comparison to both selected spaces and recompute agreement before reporting."""
    meta = read_json(directory/"metadata.json")
    signature = meta["signature"]
    for side, name in (("left", "dinov2"), ("right", "clip")):
        space = spaces[name]
        if signature[f"{side}_similarity_space_id"] != space["meta"]["similarity_space_id"] or signature[f"{side}_neighbors_sha256"] != file_sha256(space["path"]/"nearest_neighbors.parquet"):
            raise ValueError("Comparison is not bound to the selected similarity spaces")
    if signature["dataset_id"] != spaces["dinov2"]["meta"]["dataset_id"] or meta["comparison_id"] != stable_id(signature) or signature["ks"] != [1, 5, 10, 20]:
        raise ValueError("Comparison identity or k configuration is inconsistent")
    for name in ("neighbor_agreement.parquet", "neighbor_agreement_summary.csv"):
        if file_sha256(directory/name) != meta["output_sha256"][name]:
            raise ValueError("Comparison output checksum mismatch")
    expected, summary = compare_neighbor_spaces(spaces["dinov2"]["neighbors"], spaces["clip"]["neighbors"])
    pd.testing.assert_frame_equal(pd.read_parquet(directory/"neighbor_agreement.parquet"), expected)
    pd.testing.assert_frame_equal(pd.read_csv(directory/"neighbor_agreement_summary.csv"), summary)
    return summary


def generate_similarity_report(dinov2: Path, clip: Path, comparison: Path, images_archive: Path,
                               output: Path = Path("reports/similarity"), seed: int = 0,
                               examples: int = 3) -> dict:
    """Create bounded visual evidence; no raw images or public execution outputs.

    All statistics use full unordered pairs or full directed neighborhoods.
    Only the galleries are sampled, with the same queries for both encoders.
    Private row IDs/hashes remain in ignored audit tables, never display tables.
    """
    spaces = {}
    for name, path in (("dinov2", dinov2), ("clip", clip)):
        quality = verify_similarity_directory(path)
        if not quality["quality_valid"]:
            raise ValueError("Report requires verified complete similarity artifacts")
        meta = read_json(path/"metadata.json")
        if meta["extractor"] != name or meta["top_k"] != 20:
            raise ValueError("Select the stated encoders with top_k=20 for this review")
        spaces[name] = {"path": path, "meta": meta, "quality": quality,
                        "summary": read_json(path/"similarity_summary.json"),
                        "pairs": pd.read_parquet(path/"pair_analysis.parquet"),
                        "contents": pd.read_parquet(path/"content_provenance.parquet"),
                        "neighbors": pd.read_parquet(path/"nearest_neighbors.parquet"),
                        "topk": pd.read_parquet(path/"topk_content_summary.parquet")}
    if spaces["dinov2"]["meta"]["dataset_id"] != spaces["clip"]["meta"]["dataset_id"]:
        raise ValueError("Both encoders must describe the same canonical dataset")
    agreement = _verify_comparison(comparison, spaces)
    n = spaces["dinov2"]["meta"]["content_count"]
    if not 1 <= examples <= min(6, n):
        raise ValueError("Use between one and six deterministic gallery queries")
    figures, tables = output/"figures", output/"tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    combined = {}
    for table in TABLE_NAMES:
        combined[table] = pd.concat([pd.read_csv(s["path"]/f"{table}.csv").assign(extractor=DISPLAY[name]) for name, s in spaces.items()], ignore_index=True)
    combined["global_similarity"] = pd.DataFrame([{"extractor": DISPLAY[name], **s["summary"]["global_similarity"]} for name, s in spaces.items()])
    combined["feature_spaces"] = pd.DataFrame([{"extractor": DISPLAY[name], **{key: s["meta"][key] for key in ("model_id", "feature_space_id", "similarity_space_id", "embedding_dimension", "pooling_strategy", "content_count")},
                                               "matrix_shape": f"{n} × {n}", "dtype": "float32", "quality_valid": s["quality"]["quality_valid"]} for name, s in spaces.items()])
    combined["temporal_neighbors"] = pd.DataFrame([{"extractor": DISPLAY[name], "neighborhood": kind, **values} for name, s in spaces.items() for kind, values in s["summary"]["temporal_neighbors"].items()])
    combined["temporal_coverage"] = pd.DataFrame([{"extractor": DISPLAY[name], **s["summary"]["temporal_coverage"]} for name, s in spaces.items()])
    combined["historical_membership"] = pd.DataFrame([{"split_membership_set": key, "content_count": value} for key, value in spaces["dinov2"]["summary"]["historical_membership"].items()])
    combined["neighbor_agreement_summary"] = agreement

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for i, (name, space) in enumerate(spaces.items()):
            values = space["pairs"].cosine_similarity.to_numpy()
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.hist(values, bins=80, color=COLORS[name], alpha=.9)
            ax.axvline(np.median(values), color="#252525", ls="--", label=f"Mediana {np.median(values):.4f}")
            ax.set(title=f"{DISPLAY[name]} · todos los pares de contenidos distintos (N={len(values):,})", xlabel="Similitud coseno", ylabel="Número de pares únicos")
            ax.legend()
            _save(fig, figures/FIGURE_NAMES[i])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, (name, space) in zip(axes, spaces.items(), strict=True):
            columns = ["rank1_similarity", "top5_mean_similarity", "top10_mean_similarity", "top20_mean_similarity"]
            ax.boxplot([space["topk"][c] for c in columns], tick_labels=["Rank 1", "Media 5", "Media 10", "Media 20"], showfliers=False)
            ax.set(title=DISPLAY[name], ylabel="Coseno por contenido")
        fig.suptitle("Vecindarios completos · cajas Q1–Q3, mediana; bigotes 1,5 IQR")
        _save(fig, figures/FIGURE_NAMES[2])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, (name, space) in zip(axes, spaces.items(), strict=True):
            pairs = space["pairs"]
            ax.boxplot([pairs.loc[pairs.same_sequence.eq(value).fillna(False), "cosine_similarity"] for value in (True, False)], tick_labels=["Misma secuencia", "Otra secuencia"], showfliers=False)
            ax.set(title=DISPLAY[name], ylabel="Similitud coseno")
        fig.suptitle("Secuencia inferida · cajas Q1–Q3, mediana; bigotes 1,5 IQR")
        _save(fig, figures/FIGURE_NAMES[3])
        for i, (name, space) in enumerate(spaces.items()):
            valid = space["pairs"].dropna(subset=["frame_delta"])
            fig, ax = plt.subplots(figsize=(9, 4.8))
            if not valid.empty:
                collection = ax.hexbin(np.log1p(valid.frame_delta.to_numpy(dtype=float)), valid.cosine_similarity, gridsize=60, bins="log", mincnt=1, cmap="viridis")
                fig.colorbar(collection, ax=ax, label="Pares por celda (escala logarítmica)")
            ax.set(title=f"{DISPLAY[name]} · misma secuencia, índice conocido", xlabel="log(1 + diferencia absoluta de índices inferidos)", ylabel="Similitud coseno")
            _save(fig, figures/FIGURE_NAMES[4+i])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for i, (name, _) in enumerate(spaces.items()):
            candidates = combined["quantile_candidates"]
            q = candidates.loc[candidates.extractor == DISPLAY[name]]
            x = np.arange(len(q)) + (i-.5)*.35
            axes[0].bar(x, q.cross_split_count, width=.35, label=DISPLAY[name], color=COLORS[name])
            axes[1].bar(x, 100*q.cross_split_count/q.pair_count, width=.35, label=DISPLAY[name], color=COLORS[name])
        for ax in axes:
            ax.set_xticks(np.arange(len(q)), [f"{p:g}%" for p in q.top_percentage])
            ax.set_xlabel("Fracción superior de cada encoder (cohortes anidadas)")
            ax.legend()
        axes[0].set_ylabel("Pares candidatos entre splits")
        axes[1].set_ylabel("% de pares de cada cohorte")
        fig.suptitle("Alta similitud y relación histórica entre splits · análisis descriptivo")
        _save(fig, figures/FIGURE_NAMES[6])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(agreement))
        ax.fill_between(x, agreement.Q1, agreement.Q3, color="#b9cecd", label="Q1–Q3 entre contenidos")
        ax.plot(x, agreement.mean_jaccard, "o-", color="#25767b", label="Media")
        ax.plot(x, agreement.median_jaccard, "s--", color="#303030", label="Mediana")
        ax.set(xticks=x, xticklabels=[str(k) for k in agreement.k], xlabel="k", ylabel="Jaccard de los conjuntos de vecinos", ylim=(-.02, 1.02), title="Acuerdo DINOv2–CLIP · mismas consultas, espacios independientes")
        ax.legend()
        _save(fig, figures/FIGURE_NAMES[7])

    query_ids = sorted(spaces["dinov2"]["contents"].content_id.tolist())
    queries = np.random.default_rng(seed).choice(query_ids, size=examples, replace=False).tolist()
    selections, inspected = [], []
    max_fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    with zipfile.ZipFile(images_archive, "r") as archive:
        for i, (name, space) in enumerate(spaces.items()):
            selections.append(render_neighbor_grid(space["contents"], space["neighbors"], queries, archive, figures/f"nearest_neighbors_{name}.png", name))
            # Also inspect the maximum pair even when no cosine falls within the
            # numerical near-unit tolerance. This avoids treating .996 as uninteresting.
            maximum = space["pairs"].nlargest(1, "cosine_similarity")
            near_unit = pd.read_parquet(space["path"]/"near_unit_pairs.parquet")
            review = pd.concat([maximum, near_unit]).drop_duplicates(["query_row", "neighbor_row"])
            for ordinal, pair in enumerate(review.itertuples()):
                a, b = (space["contents"].iloc[index] for index in (pair.query_row, pair.neighbor_row))
                left, right = _read_verified_image(archive, a), _read_verified_image(archive, b)
                distinct_ids = a.content_id != b.content_id
                distinct_hashes = a.image_sha256 != b.image_sha256
                if not distinct_ids or not distinct_hashes:
                    raise ValueError("Distinct-content review unexpectedly includes exact bytes")
                inspected.append({"extractor": DISPLAY[name], "reason": "maximum" if ordinal == 0 else "near_unit", "cosine_similarity": pair.cosine_similarity,
                                  "query_content_id": a.content_id, "neighbor_content_id": b.content_id,
                                  "query_image_sha256": a.image_sha256, "neighbor_image_sha256": b.image_sha256,
                                  "distinct_content_ids": bool(distinct_ids), "distinct_image_hashes": bool(distinct_hashes),
                                  "decoded_rgb_pixels_equal": bool(left.size == right.size and np.array_equal(np.asarray(left), np.asarray(right))),
                                  "same_sequence": pair.same_sequence, "frame_delta": pair.frame_delta,
                                  "historical_cross_split": pair.historical_cross_split})
                if ordinal == 0:
                    for ax, img, row in zip(axes[i], (left, right), (a, b), strict=True):
                        ax.imshow(img)
                        ax.set_title(_image_caption(row, f"{DISPLAY[name]} · máximo cos={pair.cosine_similarity:.6f}", pair.frame_delta), fontsize=9)
                        ax.axis("off")
    max_fig.suptitle("Pares de máxima similitud · contenidos y hashes de bytes distintos")
    _save(max_fig, figures/"maximum_similarity_pairs.png")
    pd.concat(selections, ignore_index=True).to_parquet(tables/"visual_selection.parquet", index=False)
    inspection = pd.DataFrame(inspected)
    inspection.to_parquet(tables/"near_unit_inspection.parquet", index=False)
    combined["maximum_pair_review"] = inspection.drop(columns=[c for c in inspection if "content_id" in c and c != "distinct_content_ids" or "sha256" in c])
    for name, frame in combined.items():
        frame.to_csv(tables/f"{name}.csv", index=False)
    receipt = {**execution_provenance(), "gallery_seed": seed, "gallery_query_count": examples,
               "gallery_selection": "numpy default_rng choice without replacement over sorted content IDs; shared queries",
               "near_unit_counts": {name: s["summary"]["near_unit_pair_count"] for name, s in spaces.items()},
               "inputs": {name: {"similarity_space_id": s["meta"]["similarity_space_id"], "metadata_sha256": file_sha256(s["path"]/"metadata.json")} for name, s in spaces.items()},
               "comparison_id": read_json(comparison/"metadata.json")["comparison_id"],
               "figures": list(FIGURE_NAMES), "output_sha256": {p.relative_to(output).as_posix(): file_sha256(p) for folder in (tables, figures) for p in sorted(folder.iterdir()) if p.is_file()}}
    write_json(output/"report_metadata.json", receipt)
    return receipt
