"""Immutable, staged ingestion publications with replayable scientific identities."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.local_images import declared_file, relative_posix_path
from flir_pipeline.data.variants import make_variant, validate_variant
from flir_pipeline.data.video_variant_contract import (
    KIND,
    IngestionConfig,
    digest_document,
    load_config,
)
from flir_pipeline.data.video_variant_ingestion import (
    MANIFEST_COLUMNS,
    TABLE_DTYPES,
    derive_evidence,
    input_files,
    observe_sources,
    records,
    snapshot_sources,
)
from flir_pipeline.utils.hashing import sha256_file


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_json(directory: Path, name: str):
    return json.loads(declared_file(directory, name).read_text(encoding="utf-8"))


def _implementation() -> dict:
    package = Path(__file__).parent
    names = (
        "video_variant_contract.py",
        "video_variant_ingestion.py",
        "video_variant_storage.py",
        "zip_image_collection.py",
        "identity.py",
        "variants.py",
        "local_images.py",
        "video_frames.py",
    )
    return {
        **{name: sha256_file(package / name) for name in names},
        "utils/hashing.py": sha256_file(package.parent / "utils" / "hashing.py"),
    }


def _execution(config_sha256, ffprobe_bin):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    probe_version = None
    if ffprobe_bin is not None:
        try:
            result = subprocess.run(
                [ffprobe_bin, "-version"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                stdin=subprocess.DEVNULL,
                shell=False,
            )
            probe_version = result.stdout.splitlines()[0] if result.stdout else None
        except (OSError, subprocess.SubprocessError):
            pass  # Individual probe errors are already explicit source-table QA.
    return dict(
        created_at=datetime.now(UTC).isoformat(),
        python_version=platform.python_version(),
        library_versions={
            name: version(name)
            for name in ("pandas", "pyarrow", "pillow", "pydantic", "pyyaml")
        },
        git_commit=commit,
        git_worktree_dirty=dirty,
        config_source_sha256=config_sha256,
        ffprobe_requested=ffprobe_bin is not None,
        ffprobe_version=probe_version,
        source_read_only=True,
    )


def _preflight(config_path: Path, input_root: Path, output_root: Path):
    root, output = input_root.expanduser().resolve(), output_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("input-root must be an existing directory")
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Input and output roots must be disjoint")
    if config_path.resolve().is_relative_to(output):
        raise ValueError("Configuration must remain outside the output root")
    repository = Path(__file__).resolve().parents[3]
    if output.is_relative_to(repository) and not any(
        output.is_relative_to(repository / name) for name in ("artifacts", "reports")
    ):
        raise ValueError(
            "Publications in the checkout must remain in ignored artifacts/ or reports/"
        )
    if any((p / "metadata.json").exists() for p in (output, *output.parents)):
        raise ValueError("Never publish inside an immutable artifact")
    return root, output


def _check_observed(config, observed, sources):
    """Check raw occurrence bindings before rebuilding the derived evidence."""
    if set(sources) != set(input_files(config)):
        raise ValueError("Source snapshot has missing or unknown inputs")
    for key, declaration in input_files(config).items():
        relative = declaration if isinstance(declaration, str) else declaration.path
        source = sources[key]
        if (
            set(source) != {"path", "sha256", "size_bytes"}
            or source["path"] != relative
            or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
            or type(source["size_bytes"]) is not int
            or source["size_bytes"] < 0
        ):
            raise ValueError("Invalid source snapshot")
        expected = None if isinstance(declaration, str) else declaration.expected_sha256
        if expected is not None and source["sha256"] != expected:
            raise ValueError("Source snapshot differs from configured checksum")
    archives = records(observed["archives"])
    if [a["archive_key"] for a in archives] != sorted(
        a.archive_key for a in config.archives
    ):
        raise ValueError("Archive coverage/order differs from configuration")
    archive_map = {row["archive_key"]: row for row in archives}
    for row in archives:
        if any(
            row[key] != sources[f"archive:{row['archive_key']}"][key]
            for key in ("path", "sha256", "size_bytes")
        ):
            raise ValueError("Archive source binding differs")
        subset = observed["entries"].loc[
            observed["entries"].archive_key.eq(row["archive_key"])
        ]
        if len(subset) != row[
            "inventoried_entries"
        ] or subset.member_ordinal.tolist() != list(range(len(subset))):
            raise ValueError("ZIP entry ordinals or inventory count differ")
        if row["archive_error"] is None and len(subset) != row["declared_entries"]:
            raise ValueError("Successful archive must inventory all entries")
        if row["archive_error"] is None and (
            row["uncompressed_bytes"] != int(subset.size_bytes.sum())
            or len(subset) > config.limits.max_archive_entries
            or row["uncompressed_bytes"] > config.limits.max_archive_uncompressed_bytes
        ):
            raise ValueError("Successful archive has inconsistent size/budget")
    entries = observed["entries"]
    if entries.entry_id.isna().any() or not entries.entry_id.is_unique:
        raise ValueError("Physical entry IDs must be unique")
    for entry in records(entries):
        archive = archive_map.get(entry["archive_key"])
        if archive is None or entry["source_archive"] != archive["path"]:
            raise ValueError("Unknown physical entry archive")
        expected = digest_document(
            [
                "zip-entry-v1",
                entry["archive_key"],
                archive["sha256"],
                entry["member_ordinal"],
                entry["source_member_path"],
            ]
        )
        if expected != entry["entry_id"]:
            raise ValueError("Physical entry identity differs")
        expected_directory = entry["source_member_path"].endswith("/")
        expected_image = (
            not expected_directory
            and PurePosixPath(entry["source_member_path"]).suffix.lower()
            in config.image_extensions
        )
        if (
            entry["is_directory"] != expected_directory
            or entry["is_image"] != expected_image
        ):
            raise ValueError("Physical entry classification differs")
        if any(
            entry[key] is None or entry[key] < 0
            for key in ("size_bytes", "compressed_bytes", "crc32")
        ):
            raise ValueError("Invalid ZIP entry size/CRC")
        digest = entry["image_sha256"]
        if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid exact image checksum")
        if entry["is_image"] and entry["read_error"] is None and digest is None:
            raise ValueError("Readable image entry requires exact content identity")
        if entry["is_image"] and entry["read_error"] is None:
            relative_posix_path(entry["source_member_path"])
            if (
                entry["size_bytes"] > config.limits.max_member_bytes
                or entry["size_bytes"] / max(entry["compressed_bytes"], 1)
                > config.limits.max_compression_ratio
            ):
                raise ValueError("Readable image exceeds configured resource limits")
        if entry["image_decode_valid"] and (
            digest is None
            or entry["image_error"] is not None
            or any(
                entry[k] is None or entry[k] <= 0
                for k in ("width", "height", "channels")
            )
        ):
            raise ValueError("Invalid successful image decode")
        if entry["image_decode_valid"] and (
            entry["width"] > config.limits.max_width
            or entry["height"] > config.limits.max_height
            or entry["width"] * entry["height"] > config.limits.max_pixels
            or config.expected_image_format is not None
            and entry["image_format"] != config.expected_image_format
            or entry["read_error"] is not None
        ):
            raise ValueError("Successful image decode violates format/budget")
    videos = records(observed["video_sources"])
    if [v["video_key"] for v in videos] != sorted(v.video_key for v in config.videos):
        raise ValueError("Video coverage/order differs")
    video_config = {v.video_key: v for v in config.videos}
    for video in videos:
        if any(
            video[key] != sources[f"video:{video['video_key']}"][key]
            for key in ("path", "sha256", "size_bytes")
        ):
            raise ValueError("Video source binding differs")
        if video["source_video_id"] != digest_document(
            ["source-video-bytes-v1", video["sha256"]]
        ):
            raise ValueError("Video byte identity differs")
        if (
            json.loads(video["declared_metadata_json"])
            != video_config[video["video_key"]].declared_metadata
        ):
            raise ValueError("Video declarations differ")
        status = video["probe_status"]
        if status not in {"not_requested", "observed", "error"} or (
            (video["probe_metadata_json"] is not None) != (status == "observed")
            or (video["probe_error"] is not None) != (status == "error")
        ):
            raise ValueError("Invalid video inspection state")


def inspect_ingestion(directory: Path, *, _staging=False) -> dict:
    """Verify stored bytes and reconstruct all IDs/QA; no original data is read."""
    directory = Path(directory).resolve()
    if not _staging and directory.name.endswith(".partial"):
        raise ValueError("Incomplete staging publication is not consumable")
    meta = _read_json(directory, "metadata.json")
    if (
        set(meta)
        != {"artifact_kind", "schema_version", "artifact_id", "identity", "execution"}
        or meta["artifact_kind"] != KIND
        or meta["schema_version"] != 1
    ):
        raise ValueError("Unexpected ingestion metadata contract")
    identity = meta["identity"]
    if (
        set(identity)
        != {
            "artifact_kind",
            "config",
            "sources",
            "output_checksums",
            "implementation_sha256",
        }
        or identity["artifact_kind"] != KIND
        or meta["artifact_id"] != digest_document(identity)
        or not _staging
        and directory.name != meta["artifact_id"]
    ):
        raise ValueError("Invalid ingestion artifact identity")
    if not identity["implementation_sha256"] or any(
        not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in identity["implementation_sha256"].values()
    ):
        raise ValueError("Invalid implementation fingerprints")
    if _read_json(directory, "receipt.json") != {
        "metadata_sha256": sha256_file(directory / "metadata.json")
    }:
        raise ValueError("Metadata receipt differs")
    expected = set(identity["output_checksums"])
    actual = {p.name for p in directory.iterdir()}
    if actual != expected | {"metadata.json", "receipt.json"}:
        raise ValueError("Publication has missing or undeclared files")
    for name, digest in identity["output_checksums"].items():
        if sha256_file(declared_file(directory, name)) != digest:
            raise ValueError(f"Modified ingestion output: {name}")
    config = IngestionConfig.model_validate(identity["config"])
    if (
        _read_json(directory, "config.json") != identity["config"]
        or _read_json(directory, "sources.json") != identity["sources"]
    ):
        raise ValueError("Stored configuration/source snapshot differs")
    data = {}
    for name, schema in TABLE_DTYPES.items():
        frame = pd.read_parquet(declared_file(directory, f"{name}.parquet"))
        if list(frame) != list(schema) or any(
            str(frame[key].dtype) != dtype for key, dtype in schema.items()
        ):
            raise ValueError(f"Unexpected table columns/dtypes: {name}")
        data[name] = frame
    observed = {name: data[name] for name in ("archives", "entries", "video_sources")}
    _check_observed(config, observed, identity["sources"])
    rebuilt, manifest, summary = derive_evidence(config, observed, identity["sources"])
    try:
        for name in TABLE_DTYPES:
            pd.testing.assert_frame_equal(data[name], rebuilt[name])
    except AssertionError as error:
        raise ValueError(
            "Stored scientific identities/index/alignment evidence differs from replay"
        ) from error
    required = {
        *(f"{name}.parquet" for name in TABLE_DTYPES),
        "config.json",
        "sources.json",
        "integrity.json",
    }
    if manifest is not None:
        required |= {"manifest.parquet", "variant.json"}
        stored_manifest = pd.read_parquet(declared_file(directory, "manifest.parquet"))
        try:
            pd.testing.assert_frame_equal(stored_manifest, manifest)
        except AssertionError as error:
            raise ValueError(
                "Scientific manifest differs from occurrence replay"
            ) from error
        if list(stored_manifest) != list(MANIFEST_COLUMNS):
            raise ValueError("Unexpected scientific manifest schema")
        variant = validate_variant(
            _read_json(directory, "variant.json"),
            dataset_id_from_manifest(manifest),
            sha256_file(directory / "manifest.parquet"),
        )
        expected_variant = make_variant(
            summary["dataset_id"],
            sha256_file(directory / "manifest.parquet"),
            config.variant_name,
            definition=config.transformation,
        )
        if variant != expected_variant:
            raise ValueError("Variant declaration differs from ingestion configuration")
        summary["dataset_variant_id"] = variant["dataset_variant_id"]
    else:
        summary["dataset_variant_id"] = None
    if expected != required or _read_json(directory, "integrity.json") != summary:
        raise ValueError("Integrity summary or optional manifest coverage differs")
    return {"metadata": meta, "summary": summary}


def build_ingestion(
    config_path: Path,
    input_root: Path,
    output_root: Path,
    *,
    ffprobe_bin: str | None = None,
) -> Path:
    """Publish once, preserving failed staging dirs; only completed equals reuse.

    The output-root lock enforces a single writer. It is never broken automatically.
    Metadata and receipt are written last inside staging, verified there, then the
    directory is renamed on the same filesystem. No original file is modified.
    """
    config_path = Path(config_path)
    config_hash = sha256_file(config_path)
    config = load_config(config_path)
    root, output = _preflight(config_path, Path(input_root), Path(output_root))
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".video-variant-writer.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(
            "Single writer; inspect interrupted work before removing this lock.\n"
        )
    try:
        stage = Path(
            tempfile.mkdtemp(
                prefix=".video-variant-stage-", suffix=".partial", dir=output
            )
        )
        observed, sources = observe_sources(config, root, ffprobe_bin=ffprobe_bin)
        data, manifest, summary = derive_evidence(config, observed, sources)
        if sha256_file(config_path) != config_hash:
            raise ValueError("Configuration changed during ingestion")
        _write_json(stage / "config.json", config.model_dump(mode="json"))
        _write_json(stage / "sources.json", sources)
        for name, frame in data.items():
            frame.to_parquet(stage / f"{name}.parquet", index=False)
        if manifest is not None:
            manifest.to_parquet(stage / "manifest.parquet", index=False)
            variant = make_variant(
                summary["dataset_id"],
                sha256_file(stage / "manifest.parquet"),
                config.variant_name,
                definition=config.transformation,
            )
            _write_json(stage / "variant.json", variant)
            summary["dataset_variant_id"] = variant["dataset_variant_id"]
        else:
            summary["dataset_variant_id"] = None
        _write_json(stage / "integrity.json", summary)
        identity = dict(
            artifact_kind=KIND,
            config=config.model_dump(mode="json"),
            sources=sources,
            output_checksums={p.name: sha256_file(p) for p in sorted(stage.iterdir())},
            implementation_sha256=_implementation(),
        )
        artifact_id = digest_document(identity)
        _write_json(
            stage / "metadata.json",
            dict(
                artifact_kind=KIND,
                schema_version=1,
                artifact_id=artifact_id,
                identity=identity,
                execution=_execution(config_hash, ffprobe_bin),
            ),
        )
        _write_json(
            stage / "receipt.json",
            {"metadata_sha256": sha256_file(stage / "metadata.json")},
        )
        inspect_ingestion(stage, _staging=True)
        if (
            snapshot_sources(config, root) != sources
            or sha256_file(config_path) != config_hash
        ):
            raise ValueError("Sources/configuration changed before publication")
        destination = output / artifact_id
        if destination.exists():
            previous = inspect_ingestion(destination)
            if previous["metadata"]["identity"] != identity:
                raise ValueError("Existing publication differs; never overwrite")
            # Only this invocation's redundant staging tree is removed after a
            # verified equivalent final publication. Interrupted staging stays.
            if stage.resolve().parent != output or stage.is_symlink():
                raise ValueError("Unsafe redundant staging cleanup")
            shutil.rmtree(stage)
        else:
            stage.rename(destination)
        return destination
    finally:
        lock.unlink()


def verify_ingestion(directory: Path, input_root: Path | None = None) -> dict:
    checked = inspect_ingestion(directory)
    meta, summary = checked["metadata"], checked["summary"]
    if input_root is not None:
        config = IngestionConfig.model_validate(meta["identity"]["config"])
        observed, sources = observe_sources(
            config, Path(input_root).expanduser().resolve()
        )
        if sources != meta["identity"]["sources"]:
            raise ValueError("Original source checksums differ")
        try:
            for name in ("archives", "entries"):
                pd.testing.assert_frame_equal(
                    observed[name], pd.read_parquet(Path(directory) / f"{name}.parquet")
                )
        except AssertionError as error:
            raise ValueError(
                "Original ZIP facts differ from stored evidence"
            ) from error
    return dict(
        publication_valid=True,
        source_bound=input_root is not None,
        image_observations_replayed=input_root is not None,
        integrity_valid=summary["integrity_valid"],
        scientific_manifest_available=summary["scientific_manifest_available"],
        alignment_verified=False,
        detector_ready=False,
        artifact_id=meta["artifact_id"],
    )


def summarize_ingestion(directory: Path) -> dict:
    return inspect_ingestion(directory)["summary"]
