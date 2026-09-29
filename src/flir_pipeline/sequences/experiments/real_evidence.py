"""Content-driven discovery and immutable normalization of external reports."""

from pathlib import Path

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.sequences.experiments.artifacts import KINDS
from flir_pipeline.sequences.experiments.structure import (
    EvidenceEnvelope,
    expected_binding,
    publish_normalized,
)
from flir_pipeline.similarity.storage import file_sha256, read_json

FAMILIES = (
    "provenance_decision_v1",
    "video11_manual_transition_review_v2",
    "video11_sequence_structure_v1",
    "video11_core_recurrence_refined_v1",
)


def producer_kind(document):
    if not isinstance(document, dict):
        return None
    declared = {
        document.get(key)
        for key in (
            "artifact",
            "artifact_kind",
            "schema_version",
            "artifact_type",
            "protocol",
        )
        if isinstance(document.get(key), str)
    } & set(FAMILIES)
    if len(declared) > 1:
        raise ValueError("Conflicting producer artifact kind declarations")
    return next(iter(declared), None)


def discover(root, source=None, selections=None):
    """Directory names are locators only; candidate compatibility is schema-bound."""
    root = Path(root).resolve()
    selections = selections or {}
    if set(selections) - set(FAMILIES):
        raise ValueError("Unknown --select producer kind")
    found = {kind: [] for kind in FAMILIES}
    # Frozen copies inside our own publications are derived evidence, not new
    # external producers. Prune by the declared publication contract, not paths.
    publications = {
        path.parent
        for path in root.rglob("metadata.json")
        if isinstance(meta := read_json(path), dict)
        and meta.get("artifact_kind") in KINDS
    }
    for path in sorted(root.rglob("*.json")):
        if publications.intersection(path.parents):
            continue
        document = read_json(path)
        kind = producer_kind(document)
        if kind is None:
            continue
        if kind in selections:
            selected = Path(selections[kind]).resolve()
            if path.resolve() != selected and path.parent.resolve() != selected:
                continue
        envelope = adapt(path, document, source)
        if envelope is not None:
            found[kind].append((path, envelope))
    result = {}
    for kind, candidates in found.items():
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one compatible {kind}; found {len(candidates)}: "
                + ", ".join(str(p) for p, _ in candidates)
                + f". Use --select {kind}=<metadata-file-or-directory> to resolve ambiguity."
            )
        result[kind] = candidates[0]
    return result


def adapt(path, document, source):
    """Only explicit, inspected contracts may translate scientific semantics."""
    if document.get("artifact") in FAMILIES:
        from flir_pipeline.sequences.experiments.native_evidence import read_native

        return read_native(path, document)
    if document.get("schema_version") == "sequence_evidence_import_v1":
        if source is None:
            raise ValueError(
                "Explicit envelopes require canonical inputs; standalone inspection accepts native schemas"
            )
        envelope = EvidenceEnvelope.model_validate(document)
        if {
            k: getattr(envelope, k) for k in expected_binding(source)
        } != expected_binding(source):
            return None
        for name, checksum in envelope.producer_files.items():
            if file_sha256(declared_file(path.parent, name)) != checksum:
                raise ValueError(f"External producer changed: {path.parent / name}")
        return envelope
    raise ValueError(
        f"Uninspected producer schema in {path}; provide its metadata/CSV schema for an explicit adapter, never infer from its directory name"
    )


def inspect_real(root, family, selections=None):
    """Validate native files and cross-report consistency without loading features/writing."""
    from flir_pipeline.sequences.experiments.native_evidence import (
        native_producers,
        validate_bundle,
    )

    found = {
        kind: report
        for kind, (_, report) in discover(root, selections=selections).items()
    }
    return {
        **validate_bundle(found, family),
        "dry_run": True,
        "published": False,
        "producers": native_producers(found),
    }


def import_real(root, source, output, selections=None, *, dry_run=False):
    found = discover(root, source, selections)
    from flir_pipeline.sequences.experiments.native_evidence import (
        NativeReport,
        import_native,
    )

    if any(isinstance(report, NativeReport) for _, report in found.values()):
        return import_native(
            {kind: report for kind, (_, report) in found.items()},
            source,
            output,
            dry_run=dry_run,
        )
    if dry_run:
        return {
            "dry_run": True,
            "published": False,
            "adapter": "discovered_explicit_producers_v1",
            "dataset_id": source.signature["dataset_id"],
            "dataset_variant_id": source.signature["dataset_variant_id"],
            "producer_files": {kind: str(path) for kind, (path, _) in found.items()},
        }
    intervals, observations, producers, originals = {}, {}, {}, {}
    for kind, (path, envelope) in found.items():
        source.safe_output(output, path.parent)
        producers[f"{kind}/{path.name}"] = file_sha256(path)
        producers.update(
            {
                f"{kind}/{name}": checksum
                for name, checksum in envelope.producer_files.items()
            }
        )
        originals[kind] = {
            "metadata_path": str(path.resolve()),
            "metadata_sha256": file_sha256(path),
            "evidence": envelope.model_dump(mode="json"),
        }
        for collection, rows, attr in (
            (intervals, envelope.intervals, "element_id"),
            (observations, envelope.observations, "observation_id"),
        ):
            for row in rows:
                key = getattr(row, attr)
                if key in collection and collection[key] != row:
                    raise ValueError(f"Conflicting evidence across producers: {key}")
                collection[key] = row
    first = found["video11_sequence_structure_v1"][1]
    merged = first.model_copy(
        update=dict(
            artifact_kind="sequence_structure_external_v1",
            producer_files=producers,
            intervals=tuple(intervals[k] for k in sorted(intervals)),
            observations=tuple(observations[k] for k in sorted(observations)),
            exhaustive_boundary_review=found["video11_manual_transition_review_v2"][
                1
            ].exhaustive_boundary_review,
        )
    )
    return publish_normalized(
        merged,
        source,
        output,
        {"real_producers": originals, "producer_files": producers},
        adapter="discovered_explicit_producers_v1",
    )
