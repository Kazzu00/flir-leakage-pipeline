"""Discover completed local runs without importing fit/solve/verification code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from flir_pipeline.explorer.models import Run

REQUIRED = {
    "clustering_run": ("content_index.parquet", "cluster_labels.npy", "cluster_summary.parquet", "metrics.json", "quality.json"),
    "split_run": ("record_split_assignments.parquet", "source_groups.parquet", "split_assignments.parquet", "quality.json"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_labels(table: Path | None) -> dict[str, str]:
    """Optional display aliases; an alias never creates or selects a run."""
    if table is None or not table.is_file():
        return {}
    rows = pd.read_csv(table, usecols=["clustering_space_id", "candidate_label"])
    if rows.clustering_space_id.duplicated().any() or rows.candidate_label.isna().any():
        raise ValueError("Ambiguous candidate display aliases")
    return dict(zip(rows.clustering_space_id, rows.candidate_label, strict=True))


def discover_runs(root: Path, aliases: dict[str, str] | None = None) -> tuple[list[Run], list[str]]:
    """Only completed metadata + required files; selected runs get checksum checks.

    Partial runs remain on disk and are never repaired or removed by this reader.
    Unknown artifact kinds (collections, benchmarks) are not experiments here.
    """
    runs, issues = [], []
    aliases = aliases or {}
    for path in sorted(root.rglob("metadata.json")):
        if any(part.endswith(".partial") for part in path.relative_to(root).parts):
            continue
        try:
            meta = read_json(path)
            kind = meta.get("artifact_kind")
            if kind not in REQUIRED:
                continue
            if not all((path.parent / name).is_file() and name in meta["output_sha256"] for name in REQUIRED[kind]):
                raise ValueError("incomplete")
            if not read_json(path.parent / "quality.json").get("quality_valid"):
                raise ValueError("quality")
            if kind == "clustering_run":
                identifier = meta["clustering_space_id"]
                run = Run(path.parent, kind, identifier, meta["dataset_id"], meta,
                          encoder=meta["extractor"], representation=meta["representation"],
                          algorithm=meta["algorithm"], clustering_space_id=identifier,
                          seed=meta.get("reduction_seed"), candidate=aliases.get(identifier, ""))
            else:
                identity = meta["identity_payload"]
                cluster_id = identity["clustering_space_id"]
                strategy = identity["configuration"]["strategy"]
                if strategy not in {"historical", "random_content", "cluster_aware"}:
                    raise ValueError("strategy")
                run = Run(path.parent, kind, meta["split_space_id"], identity["dataset_id"], meta,
                          strategy=strategy, clustering_space_id=cluster_id, seed=identity["seed"],
                          candidate=aliases.get(cluster_id, ""))
            runs.append(run)
        except (OSError, ValueError, KeyError, TypeError):
            issues.append(f"{path.relative_to(root).as_posix()}: incomplete / invalid; excluded")
    # Duplicate identities are ambiguous, even if one copy appears healthy.
    counts = pd.Series([r.space_id for r in runs], dtype=str).value_counts()
    duplicates = set(counts[counts.gt(1)].index)
    if duplicates:
        issues.append(f"{len(duplicates)} duplicate run identities excluded")
    return [r for r in runs if r.space_id not in duplicates], issues


def compatible_splits(cluster: Run, splits: list[Run]) -> list[Run]:
    """A cluster-aware overlay must refer to the selected clustering exactly."""
    return [s for s in splits if s.dataset_id == cluster.dataset_id and (
        s.strategy in {"historical", "random_content"} or s.clustering_space_id == cluster.space_id)]


def checked_file(run: Run, name: str) -> Path:
    """Check only bytes being read, without recomputing scientific metrics."""
    path = run.directory / name
    if not path.resolve().is_relative_to(run.directory.resolve()):
        raise ValueError("Artifact must remain within selected run")
    if sha256(path) != run.metadata["output_sha256"][name]:
        raise ValueError(f"Recorded checksum differs for {name}")
    return path
