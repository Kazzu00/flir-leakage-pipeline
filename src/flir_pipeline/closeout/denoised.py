"""Strict streaming TAR importer for Laura's external denoised dataset.

Numeric class 4 is preserved. Its new name is provider metadata, not a semantic
adjudication. Historical splits never become protected-test anchors here.
"""

from __future__ import annotations

import csv
import io
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image

from flir_pipeline.closeout.io import (
    digest,
    environment,
    safe_relative,
    seal,
    transaction,
    write_csv,
    write_json,
)
from flir_pipeline.similarity.storage import file_sha256

PREFIX = "dataset_denoised"
META = ("clases.yaml", "correspondencia.csv", "resumen.json", "faltantes.csv")
NAMES = ["vehicle", "building", "road", "river", "maquinaria"]
STEM = re.compile(r"^(video\d+)_(\d+)_(\d+(?:\.\d+)?)$")


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently selecting the last value."""


def _unique_mapping(loader, node):
    pairs = loader.construct_pairs(node, deep=True)
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate YAML key")
        result[key] = value
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def parse_csv(raw: bytes) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields, rows = reader.fieldnames or [], list(reader)
    if (
        not fields
        or len(fields) != len(set(fields))
        or any(None in r or None in r.values() for r in rows)
    ):
        raise ValueError("Malformed CSV header or row width")
    return fields, rows


def labels(raw: bytes) -> tuple[list[int], int]:
    counts, boundary = [0] * 5, 0
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid YOLO row or nonfinite value")
        c, x, y, w, h = values
        if (
            c != int(c)
            or int(c) not in range(5)
            or not all(0 <= v <= 1 for v in values[1:])
            or min(w, h) <= 0
        ):
            raise ValueError("Invalid YOLO class or normalized box")
        # Preserve original bytes; border overflow is reported, never clipped.
        boundary += int(
            min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6
        )
        counts[int(c)] += 1
    return counts, boundary


def validate_metadata(metadata: dict) -> tuple[list[dict], dict]:
    import json

    classes = [
        yaml.load(raw.decode("utf-8-sig"), Loader=UniqueKeyLoader)
        for raw in metadata["clases.yaml"]
    ]
    expected = {"nc": 5, "names": dict(enumerate(NAMES))}
    if (
        len(classes) != 2
        or classes[1] != expected
        or classes[0] != {"nc": 5, "names": {**expected["names"], 4: "SDZI"}}
    ):
        raise ValueError(
            "Expected exactly class versions SDZI -> maquinaria, with no other changes"
        )
    csvs = metadata["correspondencia.csv"]
    if len(csvs) != 2:
        raise ValueError("Expected two correspondence versions")
    h0, r0 = parse_csv(csvs[0])
    h1, rows = parse_csv(csvs[1])

    def renamed(k):
        return "maquinaria" if k == "SDZI" else k

    if [renamed(k) for k in h0] != h1 or [
        {renamed(k): v for k, v in row.items()} for row in r0
    ] != rows:
        raise ValueError("Correspondence versions differ beyond SDZI header rename")
    required = {
        "imagen",
        "video",
        "archivo_video",
        "fps",
        "frame",
        "segundo",
        "nombre_original",
        "dataset_origen",
        "split_original",
        "estado",
        "regimen",
        "familia_denoising",
        "ruta_imagen_origen_en_este_dataset",
        "n_cajas",
        *NAMES,
    }
    if not required <= set(h1):
        raise ValueError(
            f"Missing correspondence columns: {sorted(required - set(h1))}"
        )
    if any(len(metadata[k]) != 1 for k in ("resumen.json", "faltantes.csv")):
        raise ValueError("Summary/missing metadata must occur exactly once")
    summary = json.loads(metadata["resumen.json"][0])
    fields, missing = parse_csv(metadata["faltantes.csv"][0])
    if fields != ["imagen", "familia", "esperado"]:
        raise ValueError("Unknown faltantes schema")
    if not isinstance(summary, dict) or set(summary) != {
        "imagenes",
        "etiquetas",
        "faltantes",
        "por_familia",
        "ganadores",
    }:
        raise ValueError("Unknown resumen schema")
    families = dict(Counter(r["familia_denoising"] or "(ninguna)" for r in rows))
    if (
        summary["imagenes"] != len(rows)
        or summary["etiquetas"] != len(rows)
        or summary["faltantes"] != len(missing)
        or summary["por_familia"] != families
    ):
        raise ValueError("resumen/faltantes disagree with correspondence")
    if not isinstance(summary["ganadores"], dict):
        raise ValueError("Invalid denoising winners")
    for family, winner in summary["ganadores"].items():
        if (
            set(winner)
            != {"metodo", "experimento", "checkpoint", "paso_checkpoint", "parada"}
            or family not in families
        ):
            raise ValueError("Unknown winner schema/family")
        for row in rows:
            if row["familia_denoising"] == family and any(
                str(row.get(k)) != str(v) for k, v in winner.items()
            ):
                raise ValueError("Winner provenance disagrees with correspondence")
    if len({r["imagen"] for r in missing}) != len(missing) or {
        r["imagen"] for r in rows
    } & {r["imagen"] for r in missing}:
        raise ValueError("Conflicting missing-image declarations")
    return rows, {"summary": summary, "missing_rows": missing, "missing_schema": fields}


def prepare(
    source: Path, output: Path, expected_count: int = 1116, deep_verify: bool = False
) -> dict:
    source_sha = file_sha256(source)
    with transaction(output, source) as stage:
        metadata = defaultdict(list)
        inventory, images, annotations, folded = [], {}, {}, set()
        with tarfile.open(source, "r:*") as tar:
            for order, member in enumerate(tar):
                name = safe_relative(
                    member.name.rstrip("/") if member.isdir() else member.name
                )
                parts = PurePosixPath(name).parts
                if member.isdir():
                    if name not in (PREFIX, f"{PREFIX}/images", f"{PREFIX}/labels"):
                        raise ValueError(f"Unknown TAR directory: {name}")
                    continue
                if not member.isfile():
                    raise ValueError(f"Nonregular TAR member: {name}")
                is_meta = len(parts) == 2 and parts[0] == PREFIX and parts[1] in META
                if not is_meta and not (
                    len(parts) == 3
                    and parts[0] == PREFIX
                    and (
                        (
                            parts[1] == "images"
                            and PurePosixPath(name).suffix.lower()
                            in (".png", ".jpg", ".jpeg")
                        )
                        or (
                            parts[1] == "labels"
                            and PurePosixPath(name).suffix == ".txt"
                        )
                    )
                ):
                    raise ValueError(f"Unrecognized TAR member: {name}")
                if not is_meta and name.casefold() in folded:
                    raise ValueError(f"Duplicate image/label: {name}")
                folded.add(name.casefold())
                # Bounds protect against malformed metadata and unexpectedly huge members.
                if member.size > (16 * 1024**2 if is_meta else 128 * 1024**2):
                    raise ValueError(f"Oversized TAR member: {name}")
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable member: {name}")
                raw = stream.read()
                if len(raw) != member.size:
                    raise ValueError("Truncated TAR member")
                if is_meta:
                    metadata[parts[1]].append(raw)
                    version = len(metadata[parts[1]])
                    target = (
                        stage
                        / "metadata_original"
                        / f"{PurePosixPath(name).stem}_version{version}{PurePosixPath(name).suffix}"
                    )
                else:
                    target = stage / parts[1] / parts[2]
                    collection = images if parts[1] == "images" else annotations
                    stem = PurePosixPath(name).stem
                    if stem.casefold() in {k.casefold() for k in collection}:
                        raise ValueError("Ambiguous image/label stem")
                    entry = {
                        "path": target.relative_to(stage).as_posix(),
                        "sha256": digest(raw),
                    }
                    if parts[1] == "images" and deep_verify:
                        with Image.open(io.BytesIO(raw)) as im:
                            im.load()
                            entry.update(width=im.width, height=im.height, mode=im.mode)
                    if parts[1] == "labels":
                        entry["counts"], entry["boundary_overflow"] = labels(raw)
                    collection[stem] = entry
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                inventory.append(
                    {
                        "tar_order": order,
                        "member": name,
                        "size": len(raw),
                        "sha256": digest(raw),
                        "output": target.relative_to(stage).as_posix(),
                    }
                )
        if set(metadata) != set(META):
            raise ValueError("Missing required metadata")
        rows, meta_report = validate_metadata(metadata)
        if (
            len(images) != expected_count
            or set(images) != set(annotations)
            or len(rows) != len(images)
            or len({r["imagen"] for r in rows}) != len(rows)
        ):
            raise ValueError(
                "Expected unique paired images/labels/correspondence counts"
            )
        if {r["imagen"] for r in rows} != {
            Path(v["path"]).name for v in images.values()
        }:
            raise ValueError("CSV/image names differ")
        manifest, sources, content_labels = [], {}, defaultdict(set)
        time_discrepancies = []
        for row in rows:
            stem = Path(row["imagen"]).stem
            match = STEM.fullmatch(stem)
            fps, second = float(row["fps"]), float(row["segundo"])
            if (
                match is None
                or row["estado"] != "OK"
                or row["video"] != match[1]
                or int(row["frame"]) != int(match[2])
                or not math.isfinite(fps)
                or fps <= 0
                or not math.isfinite(second)
                or second < 0
                or abs(second - float(match[3])) > 0.001001
            ):
                raise ValueError(
                    f"Invalid video/frame/time/status correspondence: {stem}"
                )
            video, frame = row["video"], int(row["frame"])
            if abs(second - float(match[3])) > 0.000001:
                time_discrepancies.append(
                    {
                        "image": row["imagen"],
                        "csv_second": second,
                        "filename_second": float(match[3]),
                        "absolute_difference": abs(second - float(match[3])),
                    }
                )
            source_video = row["archivo_video"]
            if not source_video or sources.setdefault(video, (source_video, fps)) != (
                source_video,
                fps,
            ):
                raise ValueError("Conflicting video source/fps")
            image, label = images[stem], annotations[stem]
            if label["counts"] != [int(row[k]) for k in NAMES] or sum(
                label["counts"]
            ) != int(row["n_cajas"]):
                raise ValueError(f"YOLO/CSV object count mismatch: {stem}")
            if not row["nombre_original"] or not row["dataset_origen"]:
                raise ValueError("Missing original occurrence provenance")
            content_labels[image["sha256"]].add(label["sha256"])
            manifest.append(
                {
                    **row,
                    "frame_id": "denoised_" + digest(row["imagen"].encode())[:24],
                    "content_id": image["sha256"],
                    "image_path": image["path"],
                    "label_path": label["path"],
                    "label_sha256": label["sha256"],
                    "source_frame_key": f"{source_video}:{frame}",
                    "boundary_overflow": label["boundary_overflow"],
                    **{f"class_{i}": n for i, n in enumerate(label["counts"])},
                }
            )
        # Both full metadata versions survive; selected versions are explicit copies.
        for name in META:
            (stage / name).write_bytes(metadata[name][-1])
        write_csv(stage / "members_inventory.csv", inventory)
        write_csv(stage / "manifest.csv", manifest)
        report = {
            "schema": "flir_denoised_tar_v1",
            "state": "AUDITED_NO_SPLIT",
            "source_tar_sha256": source_sha,
            "source_tar_bytes": source.stat().st_size,
            "images": len(images),
            "labels": len(annotations),
            "correspondence_rows": len(rows),
            "deep_verified_images": len(images) if deep_verify else 0,
            "unique_contents": len(content_labels),
            "annotation_conflict_contents": sorted(
                k for k, v in content_labels.items() if len(v) > 1
            ),
            "boxes_per_class": {
                str(i): sum(r[f"class_{i}"] for r in manifest) for i in range(5)
            },
            "boundary_overflow_boxes": sum(r["boundary_overflow"] for r in manifest),
            "time_discrepancies": time_discrepancies,
            "time_tolerance_seconds": 0.001001,
            "time_tolerance_rationale": "Two millisecond-precision representations may differ by one rounding unit; retain both without correcting or asserting native timestamps",
            "original_split_counts": dict(Counter(r["split_original"] for r in rows)),
            "video_counts": dict(Counter(r["video"] for r in rows)),
            "metadata": meta_report,
            "metadata_versions": {
                k: [
                    {"version": i + 1, "sha256": digest(v)}
                    for i, v in enumerate(values)
                ]
                for k, values in metadata.items()
            },
            "selected_versions": {k: len(v) for k, v in metadata.items()},
            "class_4": {
                "id": 4,
                "name": "maquinaria",
                "historical_alias": "SDZI",
                "attribution": "Laura, as reported in supplied brief; semantic equivalence not independently adjudicated",
            },
            "limits": [
                "Historical membership is not protected-test membership",
                "CSV provenance is provider-reported, not independently verified against source video",
                "No clustering, split or semantic annotation approval created",
            ],
            **environment(),
        }
        if file_sha256(source) != source_sha:
            raise ValueError("Source TAR changed during import")
        write_json(stage / "audit.json", report)
        seal(stage, {"state": "AUDITED_NO_SPLIT", "source_tar_sha256": source_sha})
    return report
