"""Frozen manual evidence, image QA, complete occurrences and source binding."""

import hashlib
import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_sequence_experiments import (
    small_config,
    synthetic_memory,
    zones,
)

from flir_pipeline.sequences.experiments.artifacts import inspect, tables
from flir_pipeline.sequences.experiments.config import RecurrenceConfig
from flir_pipeline.sequences.experiments.review import (
    create_package,
    import_decisions,
    verify_decision_history,
)
from flir_pipeline.sequences.experiments.runner import (
    run_boundary,
    run_recurrence,
    suite,
    verify,
)
from flir_pipeline.sequences.experiments.structure import (
    expected_binding,
    import_evidence,
)
from flir_pipeline.similarity.storage import file_sha256, write_json


def reviewed_source(tmp_path):
    source = synthetic_memory(duplicate=True)
    image_root = tmp_path / "images"
    image_root.mkdir()
    content_map, image_paths = {}, {}
    for i, content in enumerate(source.contents):
        buffer = io.BytesIO()
        Image.new("RGB", (36, 24), (i, i * 3 % 256, i * 7 % 256)).save(
            buffer, format="PNG"
        )
        data = buffer.getvalue()
        content_map[content] = hashlib.sha256(data).hexdigest()
        image_paths[content_map[content]] = f"synthetic-{i}.png"
        (image_root / f"synthetic-{i}.png").write_bytes(data)
    new_ids = [content_map[c] for c in source.contents]
    order = np.argsort(new_ids)
    source.contents = sorted(new_ids)
    source.embeddings = {e: x[order] for e, x in source.embeddings.items()}
    source.records["content_id"] = source.records.content_id.map(content_map)
    source.records["content_row"] = source.records.content_id.map(
        {c: i for i, c in enumerate(source.contents)}
    )
    source.records["image_sha256"] = source.records.content_id
    source.records["image_path"] = source.records.content_id.map(image_paths)
    producer = tmp_path / "producer.csv"
    producer.write_text("synthetic review\n", encoding="utf-8")
    envelope = {
        "schema_version": "sequence_evidence_import_v1",
        "artifact_kind": "sequence_structure_external_v1",
        **expected_binding(source),
        "producer_files": {producer.name: file_sha256(producer)},
        "reviewer": "synthetic",
        "reviewed_at": "2026-09-01T12:00:00Z",
        "ground_truth": False,
        "split_created": False,
        "automatic_confirmation": False,
        "intervals": zones(source).to_dict("records"),
    }
    path = tmp_path / "envelope.json"
    write_json(path, envelope)
    structure = import_evidence(path, source, tmp_path / "output")
    return source, structure, image_root


def test_full_reviewed_suite_keeps_targets_recurrence_stability_and_ablations(tmp_path):
    source, structure, _ = reviewed_source(tmp_path)
    output = suite(source, small_config(), tmp_path / "output", structure)
    assert verify(output, source, structure)["quality_valid"]
    data = tables(output)
    assert {
        "ari",
        "ami",
        "v_measure",
        "evaluated_n",
        "evaluated_content_coverage",
    } <= set(data["comparison"])
    assert {
        "combined_union",
        "combined_intersection",
        "original_l2_temporal",
        "clustering_transition",
    } <= set(data["boundary_ablation_comparison"].signal)
    assert data["boundary_stability"].shape[0] > 0
    assert "recurrence_pair_jaccard" in data["stability"]
    assert not inspect(output)["semantics"]["visual_dependency_groups_created"]
    # A modified dependency cannot hide behind a valid suite metadata checksum.
    child = data["artifacts"].query("role == 'clustering'").iloc[0]
    child_path = tmp_path / "output" / child.relative_directory / "assignments.parquet"
    child_path.write_bytes(child_path.read_bytes() + b"tamper")
    assert not verify(output, source, structure)["quality_valid"]


@pytest.mark.parametrize("kind", ["boundary", "recurrence"])
def test_review_package_images_and_immutable_decision_history(tmp_path, kind):
    source, structure, image_root = reviewed_source(tmp_path)
    evidence = (
        run_boundary(source, small_config().boundary, tmp_path / "output")
        if kind == "boundary"
        else run_recurrence(
            source, structure, RecurrenceConfig(rank_fraction=1.0), tmp_path / "output"
        )
    )
    package = create_package(
        evidence, source, tmp_path / "review", images_root=image_root, radius=2
    )
    meta = inspect(package)
    assert meta["semantics"]["ground_truth"] is False
    data = tables(package)
    assert len(data["occurrences"]) == len(source.records)
    assert set(data["context"].frame_id) <= set(source.records.frame_id)
    if kind == "recurrence":
        roles = data["context"].role
        assert (
            roles.str.startswith("clip_medoid").any()
            and roles.str.startswith("dinov2_medoid").any()
        )
        assert (
            roles.eq("strongest_clip_matches").any()
            and roles.eq("strongest_dinov2_matches").any()
        )
    pngs = list((package / "media").glob("*.png"))
    assert pngs
    with Image.open(pngs[0]) as image:
        assert image.size == (960, 800)
    query = data["queries"].review_query_id.iloc[0]
    decisions = pd.DataFrame(
        [
            dict(
                review_query_id=query,
                decision="supported",
                reviewer="synthetic",
                reviewed_at="2026-09-01T12:00:00Z",
                notes="manual evidence only",
            )
        ]
    )
    csv = tmp_path / "decisions.csv"
    decisions.to_csv(csv, index=False)
    review = import_decisions(package, csv, tmp_path / "review")
    assert verify_decision_history(review)["quality_valid"]
    assert verify(review, source, evidence=evidence, package=package)["quality_valid"]
    assert not verify(review, source)["quality_valid"]
    assert import_decisions(package, csv, tmp_path / "review", review) == review
    assert not inspect(review)["semantics"]["automatic_confirmation"]
    for field, value in (
        ("decision", "confirmed"),
        ("review_query_id", "unknown"),
        ("reviewed_at", "2026-09-01"),
    ):
        changed = decisions.copy()
        changed.loc[0, field] = value
        changed.to_csv(csv, index=False)
        with pytest.raises(ValueError):
            import_decisions(package, csv, tmp_path / "review", review)
    decisions.loc[0, "decision"] = "unsupported"
    decisions.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        import_decisions(package, csv, tmp_path / "review", review)
    # No update to a decision can merge temporal sequence identities or create VDGs.
    assert not any(
        c in tables(review)["decisions"] for c in ("sequence_id", "vdg_id", "split_id")
    )


def test_review_reads_and_verifies_image_bytes(tmp_path):
    source, _, image_root = reviewed_source(tmp_path)
    evidence = run_boundary(source, small_config().boundary, tmp_path / "output")
    for path in image_root.iterdir():
        path.write_bytes(b"invalid image")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        create_package(evidence, source, tmp_path / "review", images_root=image_root)


def test_external_unsupported_and_ambiguous_not_assigned_and_flags_strict(tmp_path):
    source, structure, _ = reviewed_source(tmp_path)
    from flir_pipeline.sequences.experiments.structure import (
        EvidenceEnvelope,
        structure_membership,
    )
    from flir_pipeline.similarity.storage import read_json

    interval = zones(source)
    interval["decision"] = ["ambiguous", "unsupported"]
    membership, cores = structure_membership(source, interval)
    assert cores.empty and not membership.evaluation_mask.any()
    payload = read_json(structure / "external_evidence.json")
    for value in (0, "false", True):
        with pytest.raises(ValueError):
            EvidenceEnvelope.model_validate({**payload, "ground_truth": value})
