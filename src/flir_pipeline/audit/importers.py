"""Bounded read-only adapters for observed legacy formats, retaining original bytes.

Companion identity is resolved transactionally by the store. Different
experiments/windows are never merged just because nominal times coincide.
"""

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import ZipFile

from .models import CandidateEvent, SourceAsset, TemporalObservation

PRIORITY = [
    "video13_original_split_index_audit.json",
    "video13_full_temporal_profile.json",
    "video13_full_temporal_profile.csv",
    "video13_global_5fps_candidates.json",
    "video13_global_5fps_candidates.csv",
    "video13_candidate_runs.csv",
    "video13_alert_episode_groups_gap3.json",
    "video13_review_batch01.zip",
    "video13_extended_context_review.zip",
    "video13_review_batch01_classification.json",
    "video13_extended_context_review_findings.json",
]
MAX_FILE = 64 * 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def safe_member(name: str) -> str:
    if (
        not name
        or "\\" in name
        or ":" in name
        or "\x00" in name
        or PureWindowsPath(name).drive
        or PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
    ):
        raise ValueError(f"Unsafe ZIP member: {name!r}")
    return name


def decode(data: bytes, suffix: str):
    text = data.decode("utf-8-sig")
    if suffix == ".json":

        def pairs(items):
            result = {}
            for k, v in items:
                if k in result:
                    raise ValueError(f"Duplicate JSON key: {k}")
                result[k] = v
            return result

        return json.loads(text, object_pairs_hook=pairs)
    reader = csv.DictReader(io.StringIO(text), strict=True)
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Missing or duplicate CSV headers")
    rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError("Empty or malformed CSV")
    return rows


def unpack(data: bytes, suffix: str):
    media = {}
    if suffix != ".zip":
        return decode(data, suffix), "", media
    with ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if len(names) > 10000 or len(set(names)) != len(names):
            raise ValueError("Duplicate or excessive ZIP members")
        for info in z.infolist():
            safe_member(info.filename)
            if (
                info.file_size > MAX_MEMBER
                or (info.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Oversized member or symlink")
        manifests = [
            n
            for n in names
            if PurePosixPath(n).name in {"manifest.json", "manifest.csv"}
        ]
        if not manifests:
            raise ValueError("ZIP has no supported manifest.json/manifest.csv")
        preferred = sorted(manifests, key=lambda n: (not n.endswith(".json"), n))[0]
        obj = decode(z.read(preferred), PurePosixPath(preferred).suffix)
        for name in manifests:
            other = decode(z.read(name), PurePosixPath(name).suffix)
            if name != preferred:
                compare_records(obj, other)
        for name in names:
            if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                media[name] = {"member": name, "size": z.getinfo(name).file_size}
        return obj, preferred, media


def records(obj):
    if not isinstance(obj, (list, dict)):
        raise ValueError("Expected a JSON object or record list")
    rows = obj if isinstance(obj, list) else obj.get("results", [])
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Expected a list of record objects")
    return rows


def compare_records(first, second):
    a, b = records(first), records(second)
    if len(a) != len(b):
        raise ValueError("JSON/CSV companion record counts disagree")
    for i, (left, right) in enumerate(zip(a, b, strict=True)):
        if set(left) != set(right):
            raise ValueError(f"Companion fields disagree at row {i}")
        for key, value in left.items():
            other = right[key]
            if value is None and other in (None, ""):
                continue
            if isinstance(value, bool):
                equal = str(value).lower() == str(other).lower()
            elif isinstance(value, (int, float)):
                equal = math.isclose(value, float(other), abs_tol=1e-6)
            else:
                equal = str(value) == str(other)
            if not equal:
                raise ValueError(f"Companion values disagree: row {i}, {key}")


def number(row, key, required=True):
    value = row.get(key)
    if value is None or value == "":
        if required:
            raise ValueError(f"Missing required field: {key}")
        return None
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a number: {key}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {key}")
    return result


def flag(value):
    if value is True or isinstance(value, str) and value in ("True", "true"):
        return True
    if value is False or isinstance(value, str) and value in ("False", "false"):
        return False
    raise ValueError("Invalid boolean flag")


def integer(row, key):
    value = number(row, key)
    if value < 0 or not value.is_integer():
        raise ValueError(f"Expected nonnegative integer: {key}")
    return int(value)


def normalize(data: bytes, filename: str):
    suffix = Path(filename).suffix.lower()
    sha = digest(data)
    asset_id = digest((filename + sha).encode())
    obj, manifest, media = unpack(data, suffix)
    meta = (
        {k: v for k, v in obj.items() if k != "results"}
        if isinstance(obj, dict)
        else {}
    )
    asset = SourceAsset(
        asset_id=asset_id, filename=filename, sha256=sha, kind=suffix[1:], metadata=meta
    )
    source = meta.get("source_video")
    if source is not None and not isinstance(source, str):
        raise ValueError("source_video must be a string")
    events, observations = [], []
    warnings = []
    if source is None:
        warnings.append(
            "Source video not declared; retained as unknown (no filename inference)."
        )
    if isinstance(obj, dict) and "splits" in obj:
        if not isinstance(obj["splits"], dict):
            raise ValueError("Historical splits must be an object")
        all_indices = set()
        for split, payload in obj["splits"].items():
            indices = payload["indices"]
            if split not in {"train", "val", "test"} or payload["count"] != len(
                indices
            ):
                raise ValueError("Invalid historical split/count")
            if len(indices) != len(set(indices)):
                raise ValueError("Duplicate historical index within split")
            for idx in indices:
                if type(idx) is not int or idx < 0:
                    raise ValueError("Invalid historical index")
                all_indices.add(idx)
                observations.append(
                    TemporalObservation(
                        observation_id=digest(f"{asset_id}:{split}:{idx}".encode()),
                        asset_id=asset_id,
                        dataset_index=idx,
                        original_split=split,
                        alignment_method="historical_index_only",
                    ).model_dump()
                )
        if obj["indices_unique"] != len(all_indices):
            raise ValueError("Unique historical count disagrees")
        for anomaly in historical_anomalies(observations):
            warnings.append("historical_index_overlap: " + encode(anomaly))
        return asset.model_dump(), events, observations, media, warnings
    rows = records(obj)
    if not rows:
        raise ValueError(
            "Unsupported or empty record schema; original remains unmodified"
        )
    for i, row in enumerate(rows):
        locator = f"{manifest or filename}#/results/{i}"
        eid = digest(f"{asset_id}:{locator}".encode())
        start = end = idx = end_idx = None
        start_sample = end_sample = representative = fps = None
        temporal_basis = "reported_nominal"
        evidence = []
        candidate = True
        if "left_index" in row:
            idx, end_idx = integer(row, "left_index"), integer(row, "right_index")
            start, end = number(row, "left_second"), number(row, "right_second")
            candidate = flag(row.get("review_candidate"))
            method, alert = "SIFT_RANSAC_and_blurred_correlation", "low_similarity"
        elif "approx_video_second" in row:
            representative = number(row, "approx_video_second")
            idx = end_idx = number(row, "approx_video13_index")
            for key in (
                "event_start_sample",
                "event_end_sample",
                "low_similarity_samples",
            ):
                integer(row, key)
            if number(row, "event_start_sample") > number(row, "event_end_sample"):
                raise ValueError("Reversed sample interval")
            start_sample = integer(row, "event_start_sample")
            end_sample = integer(row, "event_end_sample")
            if "frames_processed" in meta and end_sample >= integer(
                meta, "frames_processed"
            ):
                raise ValueError(
                    "Event sample interval exceeds declared sampled-frame domain"
                )
            fps = number(meta, "sample_rate_fps", required=False)
            origin = number(meta, "start_second_nominal", required=False)
            temporal_basis = "sample_interval_only; grid metadata unavailable"
            if fps is not None and fps <= 0:
                raise ValueError("sample_rate_fps must be positive")
            if fps is not None and origin is not None:
                start, end = origin + start_sample / fps, origin + end_sample / fps
                # The legacy producer omits index-base metadata. Require every
                # reported start to agree with this grid rather than guessing.
                if not math.isclose(start, representative, rel_tol=0, abs_tol=1e-6):
                    raise ValueError(
                        "Ambiguous sample grid: reported start disagrees with origin + sample/fps"
                    )
                temporal_basis = "nominal_grid; zero-based convention consistent with reported start; alignment unverified"
            method, alert = (
                "5fps_exploratory_correlation",
                str(row.get("candidate_class", "unclassified")),
            )
        elif "review_id" in row:
            idx = end_idx = number(row, "approx_index")
            key = "center_second" if "center_second" in row else "approx_source_second"
            start = end = number(row, key)
            method, alert = (
                "exploratory_review_window",
                str(row.get("kind", "context_window")),
            )
            image = row.get("image_file", row.get("image"))
            if image:
                member = str(PurePosixPath(manifest).parent / safe_member(image))
                if member not in media:
                    warnings.append(f"{locator}: missing image {member}")
                else:
                    evidence.append(member)
        elif {"start", "end", "length"} <= row.keys():
            idx, end_idx = integer(row, "start"), integer(row, "end")
            if integer(row, "length") != end_idx - idx + 1:
                raise ValueError("Run length disagrees with inclusive interval")
            method, alert = "consecutive_alert_runs", "alert_run"
        else:
            raise ValueError(f"Unsupported record at {locator}")
        metrics = {}
        for key in (
            "correlation",
            "sift_good_matches",
            "ransac_inliers",
            "minimum_correlation",
            "previous_stability",
            "following_stability",
            "global_min_correlation",
            "low_similarity_samples",
        ):
            if key in row:
                value = number(row, key, required=False)
                if value is not None:
                    if (
                        "correlation" in key or "stability" in key
                    ) and not -1 <= value <= 1:
                        raise ValueError(f"Out-of-range correlation: {key}")
                    if key in {
                        "sift_good_matches",
                        "ransac_inliers",
                        "low_similarity_samples",
                    } and (value < 0 or not value.is_integer()):
                        raise ValueError(f"Invalid count: {key}")
                metrics[key] = value
        oid = digest(f"{eid}:observation".encode())
        observations.append(
            TemporalObservation(
                observation_id=oid,
                asset_id=asset_id,
                source_video=source,
                dataset_index=idx,
                nominal_second=start,
            ).model_dump()
        )
        event = CandidateEvent(
            event_id=eid,
            asset_id=asset_id,
            record_locator=locator,
            source_video=source,
            start_second=start,
            end_second=end,
            start_index=idx,
            end_index=end_idx,
            event_start_sample=start_sample,
            event_end_sample=end_sample,
            representative_second=representative,
            sample_rate_fps=fps,
            temporal_basis=temporal_basis,
            method=str(meta.get("method", method)),
            method_version=str(meta.get("method_version", "unreported")),
            alert_type=alert,
            metrics=metrics,
            evidence=evidence,
            observations=[oid],
        )
        if candidate:
            events.append(event.model_dump())
    if "frames_received" in meta and meta["frames_received"] != len(rows) + 1:
        raise ValueError("Profile frame/transition counts disagree")
    return asset.model_dump(), events, observations, media, warnings


def historical_anomalies(observations):
    groups = {}
    for observation in observations:
        if observation["original_split"] is not None:
            groups.setdefault(observation["dataset_index"], []).append(observation)
    return [
        {
            "kind": "historical_index_overlap",
            "asset_id": items[0]["asset_id"],
            "dataset_index": index,
            "splits": sorted({r["original_split"] for r in items}),
            "observation_ids": sorted(r["observation_id"] for r in items),
            "semantic_leakage_confirmed": False,
        }
        for index, items in sorted(groups.items())
        if len({r["original_split"] for r in items}) > 1
    ]
