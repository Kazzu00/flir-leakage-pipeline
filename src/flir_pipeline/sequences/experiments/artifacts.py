"""Content-addressed immutable publications using the existing JSON/SHA helpers."""

import importlib.metadata
from pathlib import Path

import pandas as pd

from flir_pipeline.data.local_images import declared_file, relative_posix_path
from flir_pipeline.sequences.experiments.config import SEMANTICS
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)

KINDS = {
    "sequence_variant_correspondence_v1",
    "sequence_variant_comparison_v1",
    "sequence_representation_v1",
    "sequence_final_summary_v1",
    "sequence_boundary_candidates_v2",
    "sequence_structure_review_v1",
    "sequence_recurrence_v1",
    "sequence_cluster_evaluation_v1",
    "sequence_cluster_transition_diagnostics_v1",
    "sequence_experiment_suite_v1",
    "sequence_clustering_experiment_v1",
    "sequence_review_package_v1",
    "sequence_manual_review_v1",
    "sequence_external_evidence_v1",
}


def versions():
    names = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "pacmap", "pillow")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    package = Path(__file__).resolve().parents[2]
    reused = [
        package / name
        for name in (
            "sequences/detection.py",
            "clustering/algorithms.py",
            "clustering/base.py",
            "clustering/distances.py",
            "clustering/metrics.py",
            "reduction/reducers.py",
            "reduction/base.py",
            "linkage/sources.py",
            "linkage/candidates.py",
            "data/variants.py",
            "features/storage.py",
        )
    ]
    result["implementation_sha256"] = {
        path.relative_to(package).as_posix(): file_sha256(path)
        for path in [*sorted(Path(__file__).parent.glob("*.py")), *reused]
    }
    return result


def table_digest(table):
    # Logical content, ordered columns and dtypes; publication SHA additionally
    # binds the exact Parquet bytes. JSON null represents undefined metrics.
    return stable_id(
        {
            "columns": list(table),
            "dtypes": [str(t) for t in table.dtypes],
            "rows": table.to_json(orient="split", index=False, double_precision=15),
        }
    )


def publish(
    root: Path,
    kind: str,
    config: dict,
    sources: dict,
    tables: dict[str, pd.DataFrame],
    summary: dict,
    extras: dict | None = None,
    media: dict[str, bytes] | None = None,
) -> Path:
    import hashlib

    if kind not in KINDS:
        raise ValueError("Unknown experimental artifact contract")
    repository = Path(__file__).resolve().parents[4]
    resolved_root = root.resolve()
    if resolved_root.is_relative_to(repository) and not any(
        resolved_root.is_relative_to(repository / name)
        for name in ("artifacts", "reports")
    ):
        raise ValueError(
            "Publications in the checkout must remain in ignored artifacts/ or reports/"
        )
    if any(
        (parent / "metadata.json").is_file()
        for parent in (resolved_root, *resolved_root.parents)
    ):
        raise ValueError("Never publish inside an existing immutable artifact")
    extras, media = extras or {}, media or {}
    summary = {**summary, **SEMANTICS}
    for name in [*tables, *extras]:
        if not name.isidentifier() or name in {"metadata", "receipt", "summary"}:
            raise ValueError("Invalid artifact table/JSON name")
    for name in media:
        relative_posix_path(name)
        if not name.startswith("media/"):
            raise ValueError("Binary review outputs must live under media/")
    identity = {
        "artifact_kind": kind,
        "config": config,
        "sources": sources,
        "software": versions(),
        "semantics": SEMANTICS,
        "tables": {k: table_digest(v) for k, v in tables.items()},
        "json": {
            "summary": stable_id(summary),
            **{k: stable_id(v) for k, v in extras.items()},
        },
        "media": {k: hashlib.sha256(v).hexdigest() for k, v in media.items()},
    }
    directory = root / kind / stable_id(identity)
    if directory.exists():
        verified = inspect(directory)
        if verified["identity"] != identity:
            raise ValueError("Existing publication differs; never overwrite")
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_parquet(directory / f"{name}.parquet", index=False)
    for name, value in {"summary": summary, **extras}.items():
        write_json(directory / f"{name}.json", value)
    for name, content in media.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    outputs = sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    )
    write_json(
        directory / "metadata.json",
        {
            "artifact_id": stable_id(identity),
            "artifact_kind": kind,
            "identity": identity,
            "semantics": SEMANTICS,
            "output_checksums": {
                name: file_sha256(directory / name) for name in outputs
            },
            "execution": execution_provenance(),
        },
    )
    write_json(
        directory / "receipt.json",
        {"metadata_sha256": file_sha256(directory / "metadata.json")},
    )
    inspect(directory)
    return directory


def inspect(directory: Path, sources: dict | None = None) -> dict:
    meta = read_json(directory / "metadata.json")
    if set(meta) != {
        "artifact_id",
        "artifact_kind",
        "identity",
        "semantics",
        "output_checksums",
        "execution",
    }:
        raise ValueError("Unexpected metadata schema")
    identity = meta["identity"]
    if (
        set(identity)
        != {
            "artifact_kind",
            "config",
            "sources",
            "software",
            "semantics",
            "tables",
            "json",
            "media",
        }
        or identity["artifact_kind"] not in KINDS
        or meta["artifact_kind"] != identity["artifact_kind"]
        or meta["semantics"] != SEMANTICS
        or identity["semantics"] != SEMANTICS
        or meta["artifact_id"] != stable_id(identity)
        or directory.name != meta["artifact_id"]
    ):
        raise ValueError("Invalid experimental identity/semantics")
    if read_json(directory / "receipt.json") != {
        "metadata_sha256": file_sha256(directory / "metadata.json")
    }:
        raise ValueError("Modified metadata receipt")
    expected = {
        *(f"{k}.parquet" for k in identity["tables"]),
        *(f"{k}.json" for k in identity["json"]),
        *identity["media"],
    }
    actual = {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }
    if set(meta["output_checksums"]) != expected or actual != expected | {
        "metadata.json",
        "receipt.json",
    }:
        raise ValueError("Artifact has missing or undeclared files")
    for name, digest in meta["output_checksums"].items():
        if file_sha256(declared_file(directory, name)) != digest:
            raise ValueError(f"Modified artifact file: {name}")
    for name, digest in identity["tables"].items():
        if table_digest(pd.read_parquet(directory / f"{name}.parquet")) != digest:
            raise ValueError(f"Logical table identity mismatch: {name}")
    for name, digest in identity["json"].items():
        if stable_id(read_json(directory / f"{name}.json")) != digest:
            raise ValueError(f"JSON identity mismatch: {name}")
    for name, digest in identity["media"].items():
        if meta["output_checksums"][name] != digest:
            raise ValueError("Media identity mismatch")
    if sources is not None and identity["sources"] != sources:
        raise ValueError("Artifact is not bound to supplied immutable sources")
    return meta


def binding(directory: Path) -> dict:
    meta = inspect(directory)
    return {
        "artifact_id": meta["artifact_id"],
        "metadata_sha256": file_sha256(directory / "metadata.json"),
    }


def tables(directory: Path) -> dict[str, pd.DataFrame]:
    meta = inspect(directory)
    return {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in meta["identity"]["tables"]
    }


def summary(directory: Path) -> dict:
    return {
        **read_json(directory / "summary.json"),
        "verification_performed": False,
        "quality_valid": None,
        "artifact_id": read_json(directory / "metadata.json")["artifact_id"],
    }
