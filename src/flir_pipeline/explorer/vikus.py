"""Content-level VIKUS export from existing assignments and reduced coordinates.

All generated assets are disposable local presentation, never scientific outputs.
No model, distance computation, reducer, clustering or split optimizer is used.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import __version__ as pillow_version

from flir_pipeline.data.classes import DETECTION_CLASSES
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.explorer.data import with_split
from flir_pipeline.explorer.discovery import read_json, sha256
from flir_pipeline.explorer.frames import FrameReader
from flir_pipeline.explorer.models import ClusterData, SplitData
from flir_pipeline.explorer.timeline import ordered_contents
from flir_pipeline.explorer.vikus_upstream import (
    ADAPTER_VERSION,
    COMMIT,
    install_runtime,
)

SCHEMA_VERSION = "content-collection-v1"
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}
CLASS_COLUMNS = ("vehicles", "buildings", "roads", "rivers", "heavy_machinery")


def annotation_metadata(group: pd.DataFrame) -> dict:
    """Union is post-hoc presence; retain unknowns, mixed empties and conflicts."""
    classes, empties = set(), []
    for row in group.to_dict("records"):
        known = "classes_present" in row and isinstance(row["classes_present"], str)
        known &= bool(row.get("label_valid", True)) and bool(row.get("label_exists", True))
        if not known:
            empties.append(None)
            continue
        found = {int(x) for x in row["classes_present"].split("|") if x}
        if found - set(DETECTION_CLASSES):
            raise ValueError("Unknown class ID in manifest annotations")
        classes.update(found)
        empties.append(not found)
    complete = None not in empties
    empty = ("true" if all(empties) else "mixed" if any(empties) else "false") if complete else "unknown"
    presence = [DETECTION_CLASSES[i].class_name for i in sorted(classes)]
    if True in empties:
        presence.append("Empty annotation")
    if not complete:
        presence.append("Unknown annotation")
    return {**{f"_contains_{name}": "true" if i in classes else "false" if complete else "unknown" for i, name in enumerate(CLASS_COLUMNS)},
            "_class_presence": json.dumps(presence), "_empty_annotation": empty,
            "_annotation_conflict": str(group.label_sha256.nunique() > 1).lower()}


def metadata_table(cluster: ClusterData, split: SplitData | None, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per content, all occurrence identities and memberships retained."""
    if dataset_id_from_manifest(manifest) != cluster.run.dataset_id:
        raise ValueError("Clustering and manifest dataset identities differ")
    expected = cluster.records.set_index("frame_id").sort_index()
    actual = manifest.set_index("frame_id").sort_index()
    mapping_columns = ["content_id", "source_archive", "possible_sequence", "possible_frame_index", "original_split"]
    try:
        # Temporal lineage normalizes integer columns to nullable Int64.
        pd.testing.assert_frame_equal(actual[mapping_columns], expected[mapping_columns], check_dtype=False)
    except AssertionError as error:
        raise ValueError("Manifest occurrence/provenance mapping differs from loaded clustering") from error
    contents, records = with_split(cluster, split)
    annotation_groups = {key: group for key, group in manifest.groupby("content_id", sort=False)}
    occurrence_groups = {key: group for key, group in records.groupby("content_id", sort=False)}
    rows = []
    for position, row in enumerate(ordered_contents(contents).itertuples()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row.content_id):
            raise ValueError("Content ID is not a safe image filename")
        original = json.loads(row.split_membership_set)
        assigned = [SPLIT_NAMES[s] for s in row.new_splits]
        group = "Noise (-1)" if row.cluster_id == -1 else f"Cluster {row.cluster_id}"
        sequence = row.sequence_id if row.sequence_provenance_valid else "Unknown / ambiguous"
        sequence_group = f"{row.source_archive} / {sequence}" if row.sequence_provenance_valid else sequence
        annotations = annotation_metadata(annotation_groups[row.content_id])
        occurrence_fields = ["frame_id", "original_split", "new_split", "source_archive", "possible_sequence", "possible_frame_index"]
        occurrences = occurrence_groups[row.content_id].sort_values("frame_id")[occurrence_fields]
        keywords = [f"cluster:{row.cluster_id}", f"sequence:{sequence}", f"noise:{str(row.cluster_id == -1).lower()}"]
        keywords += [f"split:{s}" for s in assigned] + [f"class:{c}" for c in json.loads(annotations["_class_presence"])]
        rows.append({"id": row.content_id, "year": 0, "keywords": "|".join(keywords),
                     "_cluster_id": row.cluster_id, "_cluster_group": group,
                     "_sequence_id": sequence, "_sequence_group": sequence_group,
                     "_sequence_candidates": row.sequence_candidates,
                     "_frame_index": int(row.frame_index) if row.frame_index_valid else "",
                     "_new_split": " + ".join(assigned) or "Unassigned",
                     "_split_group": " + ".join(assigned) or "Unassigned",
                     "_split_memberships": json.dumps(assigned),
                     "_original_split_membership": json.dumps(original),
                     "_noise": str(row.cluster_id == -1).lower(),
                     "_encoder": cluster.run.encoder, "_representation": cluster.run.representation,
                     "_algorithm": cluster.run.algorithm, "_clustering_space_id": cluster.run.space_id,
                     "_split_space_id": split.run.space_id if split else "",
                     "_group_id": getattr(row, "group_id", ""),
                     "_occurrence_count": len(occurrences),
                     "_frame_occurrences": occurrences.to_json(orient="records", force_ascii=False),
                     "_representative_frame_id": row.representative_frame_id,
                     "_image_url": f"data/images/{row.content_id}.jpg", "_display_order": position,
                     **annotations})
    table = pd.DataFrame(rows)
    if not table.id.is_unique or len(table) != len(contents):
        raise ValueError("VIKUS requires exactly one item per unique content")
    return table


def find_source_reduction(cluster: ClusterData, root: Path) -> Path | None:
    identifier = cluster.run.metadata.get("reduction_space_id")
    if identifier is None:
        return None
    matches = []
    for path in sorted(root.rglob("metadata.json")):
        if any(part.endswith(".partial") for part in path.parts):
            continue
        meta = read_json(path)
        if meta.get("artifact_kind") == "reduction_run" and meta.get("reduction_space_id") == identifier:
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError("The exact source reduction must exist uniquely; no substitute is selected")
    return matches[0]


def layout_table(cluster: ClusterData, directory: Path, *, source: bool = True) -> tuple[pd.DataFrame, dict]:
    """Join saved coordinates by content ID; preserve numerical values exactly."""
    meta = read_json(directory / "metadata.json")
    if (meta["artifact_kind"] != "reduction_run" or meta["method"] not in {"pacmap", "tsne"}
            or meta["dataset_id"] != cluster.run.dataset_id
            or meta["feature_space_id"] != cluster.run.metadata["feature_space_id"]
            or meta["extractor"] != cluster.run.encoder or meta["output_dimension"] != 2):
        raise ValueError("Reduction is not aligned with the clustering feature space")
    if source and meta["reduction_space_id"] != cluster.run.metadata.get("reduction_space_id"):
        raise ValueError("Reduction is not the exact source of this clustering")
    fingerprints = {}
    for name in ("coordinates.npy", "content_index.parquet"):
        fingerprints[name] = sha256(directory / name)
        if fingerprints[name] != meta["output_sha256"][name]:
            raise ValueError("Reduction checksum differs")
    if source:
        expected = cluster.run.metadata["input_signatures"]["reduction"]
        for name in ("coordinates.npy", "content_index.parquet", "metadata.json"):
            if sha256(directory / name) != expected[name]:
                raise ValueError("Reduction differs from clustering source fingerprint")
    index = pd.read_parquet(directory / "content_index.parquet")
    coordinates = np.load(directory / "coordinates.npy", allow_pickle=False)
    if (not index.content_id.is_unique or set(index.content_id) != set(cluster.contents.content_id)
            or not np.array_equal(index.embedding_row, np.arange(len(index)))
            or coordinates.shape != (len(index), 2) or not np.isfinite(coordinates).all()
            or len(index) != meta["N"]):
        raise ValueError("Invalid reduction coordinate/index coverage")
    rows = pd.DataFrame({"id": index.content_id, "x": coordinates[:, 0], "y": coordinates[:, 1]})
    rows = rows.set_index("id").loc[cluster.contents.content_id].reset_index()
    # Pandas retains the requested index name from the indexing Series.
    rows.columns = ["id", "x", "y"]
    return rows, {"method": meta["method"], "reduction_space_id": meta["reduction_space_id"],
                  "source_of_clustering": source, "metadata_sha256": sha256(directory / "metadata.json"),
                  "coordinates_sha256": fingerprints["coordinates.npy"],
                  "csv_transform": "none; original values joined by content_id"}


def viewer_config(table: pd.DataFrame, cluster: ClusterData, split: SplitData | None, layouts: list[dict]) -> dict:
    display_layouts = [{"title": "Clusters", "type": "group", "groupKey": "_cluster_group", "columns": 6},
                       {"title": "Sequences", "type": "group", "groupKey": "_sequence_group", "columns": 12}]
    if split:
        title = "Cluster-aware split" if split.run.strategy == "cluster_aware" else f"{split.run.strategy} split"
        display_layouts.append({"title": title, "type": "group", "groupKey": "_split_group", "columns": 12})
    for layout in layouts:
        title = {"pacmap": "PaCMAP", "tsne": "t-SNE"}[layout["method"]] + " visual similarity"
        display_layouts.append({"title": title, "url": layout["url"], "scale": .5})
    fields = [("Scene / Cluster · candidato", "_cluster_group"), ("Secuencia inferida", "_sequence_group"),
              ("Índice inferido · no timestamp", "_frame_index"), ("Split", "_new_split"),
              ("Membership histórico", "_original_split_membership"), ("Noise", "_noise"),
              ("Clases · presencia post-hoc", "_class_presence"), ("Anotación vacía · mixed conserva conflictos", "_empty_annotation"),
              ("Conflicto de anotación", "_annotation_conflict"), ("Ocurrencias históricas", "_occurrence_count"),
              ("Encoder / representación", "_encoder"), ("Representación", "_representation"), ("Algoritmo", "_algorithm"),
              ("clustering_space_id", "_clustering_space_id"), ("split_space_id", "_split_space_id"),
              ("content_id", "id"), ("group_id", "_group_id"), ("Mapping de ocurrencias", "_frame_occurrences")]
    cluster_order = table.sort_values(["_noise", "_cluster_id"])._cluster_group.drop_duplicates().tolist()
    split_order = [s for s in ("train", "validation", "test") if s in set(table._split_group)]
    split_order += sorted(set(table._split_group) - set(split_order))
    experiment = split.run.space_id if split else cluster.run.space_id
    alias = (split.run.candidate if split else cluster.run.candidate) or "FLIR"
    return {"project": {"name": f"FLIR · {alias} · Scene / Cluster", "quality": 1}, "searchEnabled": True,
            "delimiter": "|", "projection": {"columns": 6},
            "loader": {"items": "data/data.csv", "timeline": "data/timeline.csv", "info": "data/info.md",
                       "layouts": display_layouts,
                       "textures": {"medium": {"size": 128, "url": "data/sprites/manifest.json"},
                                    "detail": {"size": 1024, "csv": "_image_url"}}},
            "filter": {"type": "crossfilter", "dimensions": [
                {"label": "Cluster · candidate scene", "source": "_cluster_group"},
                {"label": "Sequence", "source": "_sequence_group"},
                {"label": "Split", "source": "_split_memberships"},
                {"label": "Noise", "source": "_noise"},
                {"label": "Class presence · post-hoc", "source": "_class_presence"}]},
            "sortArrays": {"_cluster_group": cluster_order, "_sequence_group": sorted(table._sequence_group.unique()),
                           "_split_group": split_order, "_split_memberships": ["train", "validation", "test"], "_noise": ["false", "true"]},
            "style": {"fontColor": "#203846", "fontColorActive": "#ffffff", "fontBackground": "#176b87",
                      "textShadow": "#f0f4f6", "canvasBackground": "#edf2f4", "timelineBackground": "#ffffff",
                      "timelineFontColor": "#203846", "detailBackground": "#ffffff", "infoBackground": "#173b4a",
                      "infoFontColor": "#ffffff", "searchbarBackground": "#ffffff"},
            "detail": {"structure": [{"name": label, "source": key, "type": "array" if key == "_class_presence" else "text", "display": "wide"} for label, key in fields]},
            "flir": {"schema": SCHEMA_VERSION, "experiment_label": f"{alias} · {experiment}",
                     "year": "0 is a constant technical placeholder, never a calendar year or frame index; no year layout or detail field",
                     "timing": "filename-inferred indices; no verified timestamps or source FPS",
                     "layout_policy": "saved coordinates, uniform viewport scale; no spacing, collision removal or refit"}}


def safe_bundle_path(workspace: Path, name: str, data_root: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", name):
        raise ValueError("Bundle name must be a plain portable identifier")
    workspace = workspace.resolve()
    intended = workspace / "reports" / "explorer" / "vikus"
    base = intended.resolve()
    destination = (base / name).resolve()
    if base != intended or not destination.is_relative_to(base) or destination.is_relative_to(data_root.resolve()):
        raise ValueError("VIKUS output must stay under reports/explorer/vikus, outside source data")
    return destination


def create_images(destination: Path, table: pd.DataFrame, manifest: pd.DataFrame, data_root: Path) -> dict:
    """Aspect-preserving resizing/encoding only; 2048 px atlases, 128 px cells."""
    image_dir, sprite_dir = destination / "data/images", destination / "data/sprites"
    image_dir.mkdir(parents=True)
    sprite_dir.mkdir(parents=True)
    sheets = []
    with FrameReader(data_root, manifest) as reader:
        for offset in range(0, len(table), 256):
            page = table.iloc[offset:offset + 256]
            atlas = Image.new("RGB", (2048, 2048), "#edf2f4")
            sprites = []
            for position, row in enumerate(page.itertuples(index=False, name=None)):
                item = dict(zip(table.columns, row, strict=True))
                image = reader.image(item["_representative_frame_id"], width=1024)
                ratio = 1024 / max(image.size)
                detail = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS)
                detail.save(image_dir / f"{item['id']}.jpg", quality=92, subsampling=0)
                image.thumbnail((128, 128), Image.Resampling.LANCZOS)
                x, y = (position % 16) * 128, (position // 16) * 128
                atlas.paste(image, (x, y))
                sprites.append({"name": item["id"], "position": {"x": x, "y": y}, "dimension": {"w": image.width, "h": image.height}})
            filename = f"sheet-{offset // 256:03d}.jpg"
            atlas.save(sprite_dir / filename, quality=92, subsampling=0)
            sheets.append({"image": filename, "sprites": sprites})
    sprite_manifest = {"meta": {"type": "pixi-packer", "version": 1}, "spritesheets": sheets}
    (sprite_dir / "manifest.json").write_text(json.dumps(sprite_manifest, indent=2), encoding="utf-8")
    return {"images": len(table), "sprite_sheets": len(sheets), "sprite_cell_size": 128, "detail_max_side": 1024,
            "encoding": "JPEG quality=92 subsampling=0", "resampling": "Pillow LANCZOS; preserve aspect ratio; no crop/enhancement"}


def verify_bundle(directory: Path) -> dict:
    """Verify generated-file integrity; not a new scientific verification."""
    receipt = read_json(directory / "bundle_receipt.json")
    if receipt.get("kind") != "flir_vikus_local_bundle" or not receipt.get("complete"):
        raise ValueError("Incomplete local VIKUS bundle")
    for name, checksum in receipt["output_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != checksum:
            raise ValueError("Generated VIKUS bundle differs from its receipt")
    return receipt


def build_bundle(workspace: Path, manifest_path: Path, data_root: Path, cluster: ClusterData,
                 split: SplitData | None, reductions: list[tuple[Path, bool]], runtime_archive: Path,
                 *, name: str | None = None) -> Path:
    manifest = pd.read_parquet(manifest_path)
    table = metadata_table(cluster, split, manifest)
    layouts, layout_rows = [], []
    for directory, is_source in reductions:
        rows, info = layout_table(cluster, directory, source=is_source)
        if info["method"] in {layout["method"] for layout in layouts}:
            raise ValueError("Choose at most one explicitly identified layout per reduction method")
        info["url"] = f"data/layouts/{info['method']}.csv"
        layouts.append(info)
        layout_rows.append(rows)
    if cluster.run.metadata.get("reduction_space_id") and not any(layout["source_of_clustering"] for layout in layouts):
        raise ValueError("The source reduced coordinates must be included")
    sources = [manifest_path]
    for run in [cluster.run, *([split.run] if split else [])]:
        sources.extend([run.directory / "metadata.json", *(run.directory / n for n in run.metadata["output_sha256"])])
    for directory, _ in reductions:
        sources.extend(directory / name for name in ("metadata.json", "content_index.parquet", "coordinates.npy"))
    sources = list(dict.fromkeys(sources))
    before = {path: sha256(path) for path in sources}
    implementation = [Path(__file__), Path(__file__).with_name("vikus_upstream.py"),
                      *sorted(Path(__file__).with_name("vikus_assets").iterdir())]
    identity = {"schema": SCHEMA_VERSION, "upstream_commit": COMMIT, "adapter_version": ADAPTER_VERSION,
                "exporter_sha256": {path.name: sha256(path) for path in implementation},
                "dataset_id": cluster.run.dataset_id, "clustering_space_id": cluster.run.space_id,
                "split_space_id": split.run.space_id if split else None,
                "source_hashes": list(before.values()), "pillow_version": pillow_version, "layouts": layouts}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identifier = split.run.space_id if split else cluster.run.space_id
    destination = safe_bundle_path(workspace, name or f"{identifier}-{fingerprint[:8]}", data_root)
    if destination.exists():
        receipt = verify_bundle(destination)
        if receipt["export_identity"] != identity:
            raise ValueError("Existing bundle belongs to a different export; choose a new name")
        return destination
    archives = [(data_root / a).resolve() for a in sorted(manifest.source_archive.unique())]
    if any(not a.is_relative_to(data_root.resolve()) for a in archives):
        raise ValueError("Source ZIP outside FLIR_DATA_ROOT")
    archive_hashes = {a: sha256(a) for a in archives}
    destination.mkdir(parents=True, exist_ok=False)
    runtime = install_runtime(runtime_archive, destination)
    (destination / "data/layouts").mkdir(parents=True)
    table.to_csv(destination / "data/data.csv", index=False, lineterminator="\n")
    for rows, layout in zip(layout_rows, layouts, strict=True):
        rows.to_csv(destination / layout["url"], index=False, float_format="%.17g", lineterminator="\n")
    config = viewer_config(table, cluster, split, layouts)
    (destination / "data/config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "data/timeline.csv").write_text("year,title,text\n", encoding="utf-8")
    info = (f"# FLIR · Scene / Cluster\n\n{len(table)} contenidos únicos; {len(manifest)} ocurrencias históricas.\n\n"
            f"Experimento: {identifier}\n\nClustering: {cluster.run.space_id}\n\n"
            "## Inspección local\nCada imagen representa un content_id. Un cluster es una **candidate scene**, no una escena validada. "
            "Noise conserva -1 y sus asignaciones singleton existentes.\n\n"
            "## Navegación\nRueda para zoom; arrastra para desplazar; clic en imagen para metadata. "
            "Filtros independientes a la izquierda. Clic en el título de un filtro lo restablece. "
            "Varios valores de clases o splits requieren su presencia conjunta (AND); el resto usa OR dentro de cada dimensión.\n\n"
            "## Límites\nSecuencia e índice son inferidos, no timestamps. year=0 es sólo un requisito técnico oculto. "
            "Las clases son presencia post-hoc de cualquiera de las ocurrencias; se conservan anotaciones conflictivas. "
            "Las proyecciones reutilizan coordenadas existentes; sólo cambia la escala uniforme del viewport. "
            "No se aplica separación artificial de imágenes ni se validan escenas por su apariencia. "
            "Los puntos filtrados conservan sus coordenadas y se atenúan.\n\n"
            "Streamlit conserva la inspección detallada, playback, timelines, gaps y comparación de particiones. "
            "Usa los mismos IDs de experimento y cluster en ambas herramientas.\n\n"
            f"## Attribution\n[VIKUS Viewer](https://github.com/cpietsch/vikus-viewer), Christopher Pietsch y colaboradores. "
            f"MIT; commit {COMMIT}. Licencia íntegra en LICENSE.md. Adaptaciones locales documentadas en docs/visualization/vikus.md.\n")
    (destination / "data/info.md").write_text(info, encoding="utf-8")
    assets = create_images(destination, table, manifest, data_root)
    if any(sha256(p) != checksum for p, checksum in {**before, **archive_hashes}.items()):
        raise ValueError("A source changed during export; bundle left incomplete")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=workspace, capture_output=True, text=True, check=False).stdout.strip()
    receipt = {"kind": "flir_vikus_local_bundle", "complete": True, "created_at": datetime.now(UTC).isoformat(),
               "source_git_commit": commit, "source_git_dirty": bool(dirty), "export_identity": identity, "runtime": runtime, "assets": assets,
               "content_count": len(table), "occurrence_count": len(manifest),
               "source_zip_sha256": {p.name: value for p, value in archive_hashes.items()},
               "sources_unchanged": True, "scientific_experiments_executed": False,
               "output_sha256": {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}}
    (destination / "bundle_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verify_bundle(destination)
    return destination
