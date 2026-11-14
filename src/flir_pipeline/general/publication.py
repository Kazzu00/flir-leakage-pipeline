"""Immutable publications with input revalidation and deterministic solver replay."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from flir_pipeline.general.analysis import analyze, residual_similarity
from flir_pipeline.general.contract import Manifest, digest
from flir_pipeline.general.ingestion import verify_sources
from flir_pipeline.general.timeline import render_timeline
from flir_pipeline.splitting.base import SplitConfig
from flir_pipeline.utils.hashing import sha256_file


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def implementation():
    package = Path(__file__).resolve().parents[1]
    paths = [
        *Path(__file__).parent.glob("*.py"),
        package / "splitting/construction.py",
        package / "splitting/base.py",
        package / "data/identity.py",
        package / "data/yolo_labels.py",
        package / "data/local_images.py",
        package / "utils/hashing.py",
        package / "alignment/visual_localization.py",
    ]
    return {
        str(p.relative_to(package)).replace("\\", "/"): sha256_file(p)
        for p in sorted(paths)
    }


def load_features(path):
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {"content_ids", "embeddings_l2", "feature_space_id"}:
            raise ValueError(
                "Feature NPZ requires content_ids, embeddings_l2, feature_space_id"
            )
        return (
            source["content_ids"].tolist(),
            source["embeddings_l2"].copy(),
            str(source["feature_space_id"].item()),
        )


def build(
    manifest: Manifest,
    config: SplitConfig,
    root: Path,
    output: Path,
    feature_paths: list[Path] | None = None,
) -> dict:
    if output.exists():
        raise ValueError("Publication already exists; choose a new output")
    source_receipt = verify_sources(manifest, root)
    result = analyze(manifest, config)
    features = []
    for path in feature_paths or []:
        checksum = sha256_file(path)
        ids, matrix, space = load_features(path)
        if sha256_file(path) != checksum:
            raise ValueError("Features changed during read")
        features.append((path, ids, matrix, space, checksum))
    if features:
        result["residual_similarity"] = {
            "spaces": [
                residual_similarity(manifest, result, ids, matrix, space)
                for _, ids, matrix, space, _ in features
            ]
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage alongside destination; never replace an existing publication. Failed
    # stages remain inspectable and are not declared completed publications.
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-partial-", dir=output.parent))
    write_json(stage / "manifest.json", manifest.canonical())
    write_json(
        stage / "relations.json",
        [r.model_dump(mode="json") for r in manifest.relations],
    )
    write_json(stage / "config.json", asdict(config))
    write_json(stage / "analysis.json", result)
    write_json(stage / "input_receipt.json", source_receipt)
    feature_receipt = []
    for i, (path, _ids, _matrix, _space, checksum) in enumerate(features):
        name = f"features_{i}.npz"
        # Preserve original bytes and bind them, rather than recompressing floats.
        (stage / name).write_bytes(path.read_bytes())
        if sha256_file(stage / name) != checksum:
            raise ValueError("Features changed before publication")
        feature_receipt.append(name)
    write_json(stage / "features.json", feature_receipt)
    for run in result["runs"]:
        write_json(stage / f"{run['policy']}-seed{run['seed']}.json", run)
    timeline = render_timeline(manifest, result)
    if timeline is not None:
        (stage / "timeline.html").write_text(timeline, encoding="utf-8")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write_json(
        stage / "provenance.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "versions": {
                p: version(p)
                for p in ("numpy", "scipy", "pandas", "pydantic", "pillow")
            },
            "git_commit": commit,
            "implementation_sha256": implementation(),
            "code_identity_note": "Implementation checksums bind the working tree; commit alone may not contain these changes.",
            "training_allowed": False,
        },
    )
    write_json(
        stage / "receipt.json",
        {p.name: sha256_file(p) for p in sorted(stage.iterdir())},
    )
    # Recheck that sources did not change while solving or preparing the output.
    if verify_sources(manifest, root) != source_receipt:
        raise ValueError("Input receipt changed during build")
    if output.exists():
        raise ValueError("Publication appeared during build")
    os.rename(stage, output)
    return {
        "dataset_id": result["dataset_id"],
        "runs": len(result["runs"]),
        "states": [r["state"] for r in result["runs"]],
        "timeline": timeline is not None,
        "analysis_sha256": sha256_file(output / "analysis.json"),
    }


def verify(output: Path, root: Path) -> dict:
    receipt = read_json(output / "receipt.json")
    if set(p.name for p in output.iterdir()) != set(receipt) | {"receipt.json"}:
        raise ValueError("Publication file set changed")
    for name, checksum in receipt.items():
        if Path(name).name != name or sha256_file(output / name) != checksum:
            raise ValueError(f"Publication checksum mismatch: {name}")
    if (
        read_json(output / "provenance.json")["implementation_sha256"]
        != implementation()
    ):
        raise ValueError(
            "Implementation differs; replay with the recorded source snapshot"
        )
    manifest = Manifest.model_validate(read_json(output / "manifest.json"))
    if verify_sources(manifest, root) != read_json(output / "input_receipt.json"):
        raise ValueError("Source receipt differs")
    result = analyze(manifest, SplitConfig(**read_json(output / "config.json")))
    feature_names = read_json(output / "features.json")
    if feature_names:
        result["residual_similarity"] = {
            "spaces": [
                residual_similarity(manifest, result, *load_features(output / name))
                for name in feature_names
            ]
        }
    if digest(result) != digest(read_json(output / "analysis.json")):
        raise ValueError("Independent full analysis replay differs")
    return {
        "verified": True,
        "replayed_analysis_sha256": digest(result),
        "membership_sha256": [r["membership_sha256"] for r in result["runs"]],
        "source_files_rehashed": len(read_json(output / "input_receipt.json")),
        "training_allowed": False,
    }
