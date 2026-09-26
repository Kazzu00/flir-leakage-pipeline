"""Local occurrence-preserving inspection of verified sampled-video clustering.

This presentation layer never fits, changes assignments, identifies sequences or
publishes scientific artifacts. Every source occurrence survives the content join.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flir_pipeline.clustering.distances import retain_distances
from flir_pipeline.clustering.experiments import load_families, verify_collection
from flir_pipeline.clustering.metrics import validate_labels
from flir_pipeline.clustering.storage import clustering_provenance, verify_run
from flir_pipeline.clustering.visualization import exemplar_members
from flir_pipeline.data.local_images import declared_file, relative_posix_path
from flir_pipeline.data.video_temporal import (
    VIDEO_VERSION,
    audit_video_temporal_lineage,
)
from flir_pipeline.features.image_source import ImageSource
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json
from flir_pipeline.similarity.video_temporal import build_video_content_provenance

SEMANTICS = {
    "sequence_identity": "unknown",
    "scene_annotations": "unavailable",
    "dataset_splits": "unassigned",
    "object_annotations": "unavailable",
    "clusters_are_sequences": False,
    "video_id": "source membership only; never sequence identity",
    "timestamp_seconds": "relative sampling-grid seconds; not capture time",
    "transition": "assignment change between observed samples; not a scene boundary",
    "representative_occurrence": "display-only; first by video_id, sample_index, frame_id",
    "noise": "assignment -1; not a cluster; no medoid or within-noise coherence",
}
NOTICE = (
    "Secuencias desconocidas. Un clúster es un agrupamiento visual, no una secuencia. "
    "video_id identifica el video fuente. No se asignan escenas, particiones ni "
    "anotaciones de objetos. Los tiempos son relativos a la grilla de muestreo, "
    "no tiempos de captura verificados."
)
STYLE = """
body{font:16px/1.5 system-ui,sans-serif;color:#202d38;background:#f6f8fa;margin:0}
main{max-width:1280px;margin:auto;padding:28px}h1,h2{line-height:1.25}
.notice{background:#fff4d6;border-left:5px solid #b77900;padding:16px}
.scroll{overflow:auto;max-height:560px}table{border-collapse:collapse;background:white;
font-size:13px;width:100%}th,td{padding:7px;border:1px solid #d6dfe6;text-align:left}
th{position:sticky;top:0;background:#e8eef3}a{color:#075d85}
.gallery{display:flex;gap:14px;flex-wrap:wrap}figure{margin:0;width:270px;
background:white;border:1px solid #d6dfe6;padding:12px}figure img{width:100%;
height:180px;object-fit:contain}figcaption{font-size:13px;overflow-wrap:anywhere}
.timeline{width:100%;background:white}section{margin:26px 0}summary{cursor:pointer}
code{overflow-wrap:anywhere}small{color:#43596b}
"""


def _page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="es"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{escape(title)}</title><style>{STYLE}</style><body><main>"
            f'<h1>{escape(title)}</h1><p class="notice">{NOTICE}</p>{body}</main></body></html>')


def _table(frame: pd.DataFrame) -> str:
    return '<div class="scroll">'+frame.to_html(index=False, escape=True, na_rep="No disponible")+'</div>'


def aligned_video_records(manifest: pd.DataFrame, family) -> pd.DataFrame:
    """Bind every occurrence to both posterior provenance and feature indexing."""
    if family.source.similarity.get("provenance_mode") != "sampled_video_grid":
        raise ValueError("Video review requires sampled-video similarity v2")
    contents, records = build_video_content_provenance(manifest, family.source.content_index)
    try:
        pd.testing.assert_frame_equal(contents, family.context.provenance, check_dtype=False, check_exact=True)
        pd.testing.assert_frame_equal(records, pd.read_parquet(family.source.similarity_directory/"record_provenance.parquet"),
                                      check_dtype=False, check_exact=True)
        columns = ["frame_id", "content_id", "embedding_row"]
        mapping = pd.read_parquet(family.source.feature_directory/"record_index.parquet")
        pd.testing.assert_frame_equal(records[columns], mapping[columns].sort_values("frame_id").reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("Video occurrence provenance or feature mapping is misaligned") from error
    return records


def inspection_tables(labels, index, records, summary, neighbors) -> dict[str, pd.DataFrame]:
    """Count contents and occurrences separately; video membership is descriptive."""
    validate_labels(labels, len(index))
    assignments = index[["content_id", "embedding_row"]].assign(cluster_id=labels)
    occurrences = records.merge(assignments, on=["content_id", "embedding_row"], how="left", validate="many_to_one")
    if len(occurrences) != len(records) or occurrences.cluster_id.isna().any():
        raise ValueError("Assignments must cover every video occurrence")
    occurrences = occurrences.sort_values(["video_id", "sample_index", "frame_id"]).reset_index(drop=True)
    occurrences["is_noise"] = occurrences.cluster_id.eq(-1)
    occurrences["sequence_status"] = "unknown"
    previous = occurrences.groupby("video_id", sort=False)
    occurrences["previous_sample_index"] = previous.sample_index.shift().astype("Int64")
    occurrences["previous_cluster_id"] = previous.cluster_id.shift().astype("Int64")
    occurrences["previous_timestamp_seconds"] = previous.timestamp_seconds.shift()
    occurrences["sample_index_gap"] = occurrences.sample_index-occurrences.previous_sample_index
    occurrences["timestamp_gap_seconds"] = occurrences.timestamp_seconds-occurrences.previous_timestamp_seconds
    occurrences["missing_grid_positions"] = occurrences.sample_index_gap-1
    occurrences["assignment_changed"] = occurrences.cluster_id.ne(occurrences.previous_cluster_id).astype("boolean")
    occurrences["transition_kind"] = "unchanged"
    changed = occurrences.assignment_changed.fillna(False)
    occurrences.loc[changed, "transition_kind"] = "cluster_to_cluster"
    occurrences.loc[changed & occurrences.is_noise, "transition_kind"] = "to_noise"
    occurrences.loc[changed & occurrences.previous_cluster_id.eq(-1), "transition_kind"] = "from_noise"
    occurrences.loc[occurrences.previous_cluster_id.isna(), "transition_kind"] = "start"

    summaries = summary.set_index("cluster_id")
    retained_neighbors = {}
    for k in (5, 10, 20):
        edges = neighbors.loc[neighbors.neighbor_rank.le(k)]
        a, b = edges.query_row.to_numpy(), edges.neighbor_row.to_numpy()
        same = (labels[a] >= 0) & (labels[a] == labels[b])
        groups, counts = np.unique(labels[a[same]], return_counts=True)
        retained_neighbors[k] = dict(zip(groups, counts, strict=True))
    clusters = []
    for cluster in [*sorted(set(labels)-{-1}), -1]:
        members = np.flatnonzero(labels == cluster)
        group = occurrences.loc[occurrences.cluster_id.eq(cluster)]
        row = {"cluster_id": int(cluster), "is_noise": cluster == -1, "unique_contents": len(members),
               "content_fraction": len(members)/len(index), "occurrences": len(group),
               "occurrence_fraction": len(group)/len(occurrences), "source_video_count": group.video_id.nunique(),
               "sequence_status": "unknown"}
        for key in ("medoid_content_id", "intra_pair_count", "mean_intra_cosine", "median_intra_cosine", "Q1_intra_cosine", "Q3_intra_cosine"):
            row[key] = summaries.loc[cluster, key] if cluster != -1 else None
        for k in (5, 10, 20):
            row[f"visual_neighbor_coherence@{k}"] = (
                int(retained_neighbors[k].get(cluster, 0))/(len(members)*k)
                if cluster != -1 and len(members) else None)
        clusters.append(row)
    memberships = occurrences.groupby(["cluster_id", "is_noise", "video_id", "source_video"], sort=True).agg(
        occurrences=("frame_id", "size"), unique_contents=("content_id", "nunique")).reset_index()
    memberships["fraction_of_group_occurrences"] = memberships.occurrences/memberships.groupby("cluster_id").occurrences.transform("sum")
    memberships["fraction_of_video_occurrences"] = memberships.occurrences/memberships.groupby("video_id").occurrences.transform("sum")
    video_summary = occurrences.groupby(["video_id", "source_video"], sort=True).agg(
        occurrences=("frame_id", "size"), unique_contents=("content_id", "nunique"),
        noise_occurrences=("is_noise", "sum"), observed_assignment_changes=("assignment_changed", "sum"),
        missing_grid_positions=("missing_grid_positions", "sum")).reset_index()
    video_summary["sequence_status"] = "unknown"
    return {"clusters": pd.DataFrame(clusters), "occurrences": occurrences,
            "source_video_membership": memberships, "videos": video_summary,
            "transitions": occurrences.loc[changed].copy()}


def gallery_selections(labels, summary, family, occurrences) -> pd.DataFrame:
    """Select content in original distance; select one occurrence for display only."""
    ids = family.context.content_ids
    positions = {identity: i for i, identity in enumerate(ids)}
    displays = occurrences.sort_values(["video_id", "sample_index", "frame_id"]).drop_duplicates("content_id").set_index("content_id")
    occurrence_counts = occurrences.content_id.value_counts()
    chosen = []
    with retain_distances(family.context.original_distances):
        for row in summary.sort_values("cluster_id").itertuples():
            medoid = positions[row.medoid_content_id]
            members = exemplar_members(labels, row.cluster_id, medoid, family.context)
            chosen.extend((row.cluster_id, member, "medoid" if member == medoid else "distance_representative") for member in members)
    noise = sorted(np.flatnonzero(labels == -1), key=lambda i: ids[i])[:4]
    chosen.extend((-1, member, "noise_sample_content_id_order") for member in noise)
    rows = []
    for cluster, member, role in chosen:
        content = ids[member]
        row = displays.loc[content]
        rows.append({"cluster_id": int(cluster), "content_id": content, "role": role,
                     "display_only": True, "display_frame_id": row.frame_id,
                     "display_video_id": row.video_id, "display_sample_index": row.sample_index,
                     "display_timestamp_seconds": row.timestamp_seconds, "image_path": row.image_path,
                     "image_sha256": row.image_sha256,
                     "all_occurrence_count": int(occurrence_counts[content])})
    return pd.DataFrame(rows)


def _safe_output(output: Path, protected: list[Path]) -> None:
    resolved = output.resolve()
    repository = Path(__file__).resolve().parents[3]
    if resolved.is_relative_to(repository) and not resolved.is_relative_to(repository/"reports"):
        raise ValueError("In-repository video reports must stay under ignored reports/")
    if any(resolved.is_relative_to(p.resolve()) or p.resolve().is_relative_to(resolved) for p in protected):
        raise ValueError("Report output must be separate from read-only sources and scientific artifacts")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Existing report preserved; choose a new empty output directory")


def _run_directories(directory: Path, families) -> tuple[list[Path], dict, list[Path]]:
    meta = read_json(directory/"metadata.json")
    protected = [directory]
    if meta.get("artifact_kind") == "clustering_run":
        locations = [directory]
        checks = verify_run(directory, families[meta["extractor"]])
    elif meta.get("artifact_kind") == "clustering_comparison":
        root = directory.parents[1]
        # Validate referenced paths before the existing collection verifier reads them.
        screening = declared_file(root, relative_posix_path(meta["screening_path"])+"/metadata.json").parent
        protected.append(screening)
        for collection in (meta, read_json(screening/"metadata.json")):
            for ref in collection["runs"]:
                declared_file(root, relative_posix_path(ref["path"])+"/metadata.json")
        locations = [declared_file(root, ref["path"]+"/metadata.json").parent for ref in meta["runs"]]
        checks = {"screening": verify_collection(screening, families), "comparison": verify_collection(directory, families)}
        checks["quality_valid"] = all(c["quality_valid"] for c in checks.values())
    else:
        raise ValueError("Video review accepts a clustering run or comparison")
    if not checks["quality_valid"]:
        raise ValueError("Clustering source-bound verification failed")
    for location in locations:
        run = read_json(location/"metadata.json")
        family = families[run["extractor"]]
        if (run.get("provenance_mode") != "sampled_video_grid"
                or any(run[k] != family.source.feature[k] for k in ("dataset_id", "feature_space_id", "extractor"))):
            raise ValueError("Run must match sampled-video source identities")
    return locations, checks, [*protected, *locations]


def _timeline(group: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    categories = sorted(group.cluster_id.unique())
    positions = {cluster: i for i, cluster in enumerate(categories)}
    colors = ["#777777" if cluster == -1 else plt.get_cmap("tab20")(int(cluster) % 20) for cluster in group.cluster_id]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.scatter(group.timestamp_seconds, group.cluster_id.map(positions), c=colors, s=15)
    ticks = np.unique(np.linspace(0, len(categories)-1, min(12, len(categories)), dtype=int))
    ax.set_yticks(ticks, ["Ruido (-1)" if categories[i] == -1 else f"Clúster {categories[i]}" for i in ticks])
    ax.set(xlabel="timestamp_seconds · segundos relativos de muestreo", ylabel="Asignación observada",
           title="Muestras observadas del video fuente · secuencias desconocidas")
    fps = float(group.sample_fps.iloc[0])
    top = ax.secondary_xaxis("top", functions=(lambda seconds: seconds*fps, lambda sample: sample/fps))
    top.set_xlabel("sample_index · posición en la grilla")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=120)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def _render_run(output, meta, tables, selections, images, thumbnails, root) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output/f"{name}.csv", index=False)
        table.to_parquet(output/f"{name}.parquet", index=False)
    body = '<p><a href="../../index.html">Volver a los runs</a></p>'
    body += "<h2>Tamaño, ruido y coherencia visual</h2><p>La unidad de clustering es content_id único. "
    body += "Las ocurrencias incluyen todas las repeticiones. La coherencia usa coseno/vecinos originales; "
    body += "no valida escenas. Ruido (-1) no tiene medoide ni coherencia interna.</p>"
    body += _table(tables["clusters"])
    body += "<h2>Pertenencia a videos fuente</h2><p>Las fracciones se calculan sobre ocurrencias. "
    body += "Un contenido puede pertenecer a varios videos: sus conteos únicos por video no son aditivos.</p>"
    body += _table(tables["source_video_membership"])
    body += "<h2>Galerías de inspección</h2><p>Medoide y hasta tres representantes por distancia euclídea original. "
    body += "Para ruido se muestran hasta cuatro contenidos por orden de ID. Cada ocurrencia elegida es "
    body += "<strong>solo para visualización / display-only</strong>; todas sus posiciones se conservan en occurrences.</p>"
    selections = selections.copy()
    selections["thumbnail"] = ""
    for cluster, group in selections.groupby("cluster_id", sort=True):
        title = "Ruido (-1), sin medoide" if cluster == -1 else f"Clúster {cluster} · secuencia desconocida"
        body += f'<section><h3>{title}</h3><div class="gallery">'
        for i, row in group.iterrows():
            if row.content_id not in thumbnails:
                relative = f"images/thumb-{len(thumbnails):06d}.jpg"
                image = images.decode(row.image_path, row.image_sha256).convert("RGB")
                image.thumbnail((320, 240))
                image.save(root/relative, format="JPEG", quality=85)
                thumbnails[row.content_id] = relative
            relative = thumbnails[row.content_id]
            selections.loc[i, "thumbnail"] = relative
            caption = (f"{row.role} · display-only · frame {row.display_frame_id} · video {row.display_video_id} · "
                       f"sample_index={row.display_sample_index} · timestamp_seconds={row.display_timestamp_seconds} · "
                       f"{row.all_occurrence_count} ocurrencias conservadas")
            body += f'<figure><img loading="lazy" src="../../{relative}" alt="{escape(caption, quote=True)}"><figcaption>{escape(caption)}</figcaption></figure>'
        body += "</div></section>"
    selections.to_csv(output/"display_selections.csv", index=False)
    selections.to_parquet(output/"display_selections.parquet", index=False)
    body += "<h2>Líneas de tiempo y cambios de asignación</h2><p>Orden por sample_index. "
    body += "Solo se dibujan puntos observados, sin interpolar huecos. Los cambios no son límites de escena. "
    body += "Los colores se reutilizan; la asignación exacta está en el eje y en las tablas completas.</p>"
    body += _table(tables["videos"])
    for number, (video, group) in enumerate(tables["occurrences"].groupby("video_id", sort=True)):
        stem = f"video-{number:03d}"
        _timeline(group, output/stem)
        group.to_csv(output/f"{stem}.csv", index=False)
        body += (f'<section><h3>{escape(video)} · {escape(str(group.source_video.iloc[0]))}</h3>'
                 f'<img class="timeline" src="{stem}.png" alt="Asignaciones por tiempo relativo del video fuente">'
                 f'<p><a href="{stem}.csv">Todas las posiciones (CSV)</a> · <a href="{stem}.svg">Figura SVG</a></p></section>')
    body += '<h2>Tablas completas</h2><ul>'
    for name in [*tables, "display_selections"]:
        body += f'<li><a href="{name}.csv">{name}.csv</a> · <a href="{name}.parquet">Parquet</a></li>'
    body += "</ul>"
    title = f"{meta['extractor']} · {meta['representation']} · {meta['algorithm']} · {meta['clustering_space_id']}"
    (output/"index.html").write_text(_page(title, body), encoding="utf-8")


def generate_video_clustering_review(directory: Path, inputs: Path, images_root: Path,
                                     output: Path = Path("reports/clustering_video")) -> dict:
    """Verify all sources before publishing a separate, local, immutable review.

    Comparison reviews cover every referenced run, with run-local cluster IDs.
    Sources must remain unchanged during verification/rendering (no writer lock).
    An interrupted output is preserved and has no completion metadata.
    """
    # Match ImageSource's root normalization before checking write destinations.
    images_root = images_root.expanduser().resolve()
    inputs_sha256 = file_sha256(inputs)
    clustering_sha256 = file_sha256(directory/"metadata.json")
    specification = yaml.safe_load(inputs.read_text(encoding="utf-8"))
    manifest_path = Path(specification["manifest"])
    manifest_sha256 = file_sha256(manifest_path)
    manifest = pd.read_parquet(manifest_path)
    audit_video_temporal_lineage(manifest)  # Reject historical/mixed/annotated manifests early.
    families = load_families(inputs)
    records = {encoder: aligned_video_records(manifest, family) for encoder, family in families.items()}
    locations, checks, protected = _run_directories(directory, families)
    run_hashes = {location: file_sha256(location/"metadata.json") for location in locations}
    protected = [images_root, manifest_path, inputs, *protected]
    for family in families.values():
        protected += [family.source.feature_directory, family.source.similarity_directory, family.benchmark_directory.parents[1]]
    _safe_output(output, protected)
    with ImageSource(None, images_root) as images:
        images.validate(manifest)  # Includes duplicates and occurrences never displayed.
        output.mkdir(parents=True, exist_ok=True)
        (output/"images").mkdir()
        thumbnails, runs = {}, []
        for number, location in enumerate(locations):
            meta = read_json(location/"metadata.json")
            family = families[meta["extractor"]]
            labels = np.load(location/"cluster_labels.npy", allow_pickle=False)
            summary = pd.read_parquet(location/"cluster_summary.parquet")
            tables = inspection_tables(labels, family.source.content_index, records[meta["extractor"]], summary, family.context.neighbors)
            selected = gallery_selections(labels, summary, family, tables["occurrences"])
            relative = f"runs/run-{number:04d}"
            _render_run(output/relative, meta, tables, selected, images, thumbnails, output)
            metrics = read_json(location/"metrics.json")
            runs.append({"clustering_space_id": meta["clustering_space_id"], "encoder": meta["extractor"],
                         "representation": meta["representation"], "reduction_seed": meta["reduction_seed"],
                         "algorithm": meta["algorithm"], "clusters": len(summary), "noise_contents": int((labels == -1).sum()),
                         "noise_fraction": metrics["noise_fraction"], "silhouette_original_space": metrics["silhouette_original_space"],
                         "sequence_status": "unknown", "report": f"{relative}/index.html",
                         "source_metadata_sha256": file_sha256(location/"metadata.json")})
    pd.DataFrame(runs).to_csv(output/"runs.csv", index=False)
    body = "<p>Revisión local de todos los runs suministrados. Los IDs de clúster son locales a cada run: "
    body += "un mismo número entre runs no establece correspondencia. No se ajustó ni seleccionó un modelo.</p>"
    body += _table(pd.DataFrame(runs).drop(columns=["report", "source_metadata_sha256"]))+"<ul>"
    body += "".join(f'<li><a href="{r["report"]}">{escape(r["encoder"])} · {escape(r["clustering_space_id"])}</a></li>' for r in runs)+"</ul>"
    (output/"index.html").write_text(_page("Revisión de clustering · videos muestreados", body), encoding="utf-8")
    if (file_sha256(inputs) != inputs_sha256 or file_sha256(manifest_path) != manifest_sha256
            or file_sha256(directory/"metadata.json") != clustering_sha256
            or any(file_sha256(location/"metadata.json") != digest for location, digest in run_hashes.items())):
        raise ValueError("Source metadata changed during review; incomplete report preserved")
    metadata = {**clustering_provenance(), "artifact_kind": "sampled_video_clustering_review", "review_version": "v1",
                "manifest_version": VIDEO_VERSION, "semantics": SEMANTICS, "source_verification": checks,
                "manifest_sha256": manifest_sha256, "inputs_yaml_sha256": inputs_sha256,
                "clustering_metadata_sha256": clustering_sha256,
                "source_signatures": {encoder: family.source.signatures for encoder, family in families.items()},
                "run_count": len(runs), "occurrences_per_run": len(manifest), "unique_contents": manifest.content_id.nunique(),
                "source_images_read_only": True, "validated_image_occurrences": len(manifest),
                "display_thumbnail_count": len(thumbnails),
                "output_sha256": {p.relative_to(output).as_posix(): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output/"report_metadata.json", metadata)  # Completion marker, written last.
    return metadata
