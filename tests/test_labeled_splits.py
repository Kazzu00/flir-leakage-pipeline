"""Offline labeled candidate invariants; all images/labels are synthetic."""

import json
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.adendo import audit_adendo
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json
from flir_pipeline.splitting.base import SplitConfig, targets_from_manifest
from flir_pipeline.splitting.construction import aggregate_groups, milp_assignment
from flir_pipeline.splitting.labeled import (
    LabeledConfig,
    load_labeled_audit,
    population,
    sensitivity_groups,
    verify_original_sources,
)
from flir_pipeline.splitting.labeled_storage import (
    build_labeled_candidates,
    verify_labeled_candidates,
)


@pytest.fixture
def audited(tmp_path):
    def png(i):
        stream = BytesIO()
        Image.fromarray(
            np.random.default_rng(i).integers(0, 256, (32, 32), dtype=np.uint8)
        ).save(stream, format="PNG")
        return stream.getvalue()

    def label(i):
        return f"{i % 5} 0.5 0.5 0.2 0.2\n"

    images, labels, adendo = (
        tmp_path / n for n in ("images.zip", "labels.zip", "adendo.zip")
    )
    with ZipFile(images, "w") as zi, ZipFile(labels, "w") as zl:
        for i in range(30):
            split = ("train", "val", "test")[i % 3]
            zi.writestr(f"{split}/h{i}.png", png(i))
            zl.writestr(f"{split}/h{i}.txt", label(i) if i >= 5 else "")
        zi.writestr("val/h0.png", png(0))
        zl.writestr(
            "val/h0.txt", label(0)
        )  # exact duplicate, conflicting occurrence annotation
        zl.writestr("train/orphan.txt", label(2))
    with ZipFile(adendo, "w") as archive:
        for i in range(3):
            archive.writestr(f"train/a{i}.png", png(100 + i))
            archive.writestr(f"labels/a{i}.txt", label(i))
    evidence = tmp_path / "evidence.json"
    write_json(
        evidence,
        {
            "schema_version": "adendo_evidence_v1",
            "relations": [
                {
                    "state": "visual_candidate",
                    "source": "synthetic",
                    "method": "test",
                    "adendo_basename": "a0",
                    "historical_basename": f"h{i}",
                    "historical_split": ("train", "val")[i],
                }
                for i in range(2)
            ],
        },
    )
    audit = tmp_path / "audit"
    audit_adendo(images, labels, adendo, audit, evidence=evidence, phash_threshold=0)
    return audit, tmp_path, evidence


@pytest.fixture
def config():
    return LabeledConfig(balance=SplitConfig(seeds=(0,), node_limit=8))


def test_adapter_preserves_occurrences_conflicts_null_and_ratios(audited, config):
    audit, root, evidence = audited
    source = load_labeled_audit(audit)
    assert verify_original_sources(source, root, evidence)
    assert len(source.manifest) == 34 and source.manifest.content_id.nunique() == 33
    assert (
        source.manifest.loc[source.manifest.cohort == "adendo", "original_split"]
        .isna()
        .all()
    )
    duplicate = source.manifest[source.manifest.content_id.duplicated(False)]
    stats = source.statistics.set_index("frame_id").loc[duplicate.frame_id]
    assert sorted(stats.class_0) == [0, 1] and sum(stats.empty_annotation_count) == 1
    hist, _, expected = population(source, "historical", config)
    _, _, actual = population(source, "expanded", config)
    assert len(hist) == 31
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_allclose(expected, [10 / 31, 11 / 31, 10 / 31])
    # The original historical API still rejects supplemental nulls.
    with pytest.raises(ValueError, match="Complete historical"):
        targets_from_manifest(source.manifest, config.balance)


def test_policies_and_transitive_components(audited, config):
    source = load_labeled_audit(audited[0])
    for policy in ("S1", "S2"):
        groups, _, selection = sensitivity_groups(
            source.manifest, source.relations, policy, config
        )
        assert (
            groups.group_id.nunique() == 33 and selection["imposed_relation_ids"] == []
        )
    groups, labels, selection = sensitivity_groups(
        source.manifest, source.relations, "S3", config
    )
    assert (
        groups.group_id.nunique() == 31 and len(selection["imposed_relation_ids"]) == 2
    )
    assert sum(labels >= 0) == 3
    units = aggregate_groups(source.statistics, groups)
    assert units.record_count.max() == 4
    manifest, _, _ = population(source, "historical", config)
    hist, _, scope = sensitivity_groups(manifest, source.relations, "S3", config)
    assert (
        hist.group_id.nunique() == 30
        and scope["excluded_outside_population_records"] == 2
    )


def test_fresh_suite_semantics_replay_and_seed_identity(audited, config):
    audit, root, evidence = audited
    before = {p.name: file_sha256(p) for p in audit.iterdir()}
    first = root / "suite"
    receipt = build_labeled_candidates(
        audit, first, config, source_root=root, evidence=evidence
    )
    assert receipt["candidate_count"] == 6
    assert verify_labeled_candidates(first, audit=audit, replay=True)["quality_valid"]
    second = root / "repeat"
    repeat = build_labeled_candidates(audit, second, config)
    assert [r["split_id"] for r in receipt["runs"]] == [
        r["split_id"] for r in repeat["runs"]
    ]
    for run in receipt["runs"]:
        run_id = run["split_id"]
        assert (
            read_json(first / "runs" / run_id / "metadata.json")["membership_id"]
            == read_json(second / "runs" / run_id / "metadata.json")["membership_id"]
        )
        records = pd.read_parquet(
            first / "runs" / run_id / "occurrence_memberships.parquet"
        )
        assert (
            records.frame_id.is_unique
            and records.groupby("content_id").new_split.nunique().max() == 1
        )
        assert len(records) in {31, 34}
        metrics = read_json(first / "runs" / run_id / "metrics.json")
        assert metrics["quality"]["mandatory_group_fracture_count"] == 0
    assert before == {p.name: file_sha256(p) for p in audit.iterdir()}
    assert (
        read_json(first / "historical_reference.json")[
            "exact_duplicate_cross_split_count"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="fresh"):
        build_labeled_candidates(audit, first, config)


def test_balanced_classes_and_configured_publication_gate(audited, config):
    audit, root, _ = audited
    output = root / "balanced"
    build_labeled_candidates(
        audit, output, config, populations=("expanded",), policies=("S2",)
    )
    row = pd.read_csv(output / "comparison.csv").iloc[0]
    assert row.all_classes_covered and row.class_deviation_pp < 5
    assert row.max_relative_record_deviation < 0.1
    with pytest.raises(ValueError, match="publication ceiling"):
        build_labeled_candidates(
            audit,
            root / "impossible",
            replace(config, max_relative_record_deviation=0),
            populations=("expanded",),
            policies=("S2",),
        )
    assert not (root / "impossible" / "summary.json").exists()


def test_rejected_edges_and_conflicting_reviews(audited, config):
    source = load_labeled_audit(audited[0])
    candidate = next(r for r in source.relations if r["state"] == "visual_candidate")
    rejected = {**candidate, "state": "rejected", "relation_id": "review_rejected"}
    confirmed = {
        **candidate,
        "state": "confirmed_provenance",
        "relation_id": "review_confirmed",
    }
    edges = [candidate, rejected]
    _, _, selection = sensitivity_groups(source.manifest, edges, "S3", config)
    assert not selection["imposed_relation_ids"]
    for policy in ("S1", "S2", "S3"):
        with pytest.raises(ValueError, match="Conflicting reviews"):
            sensitivity_groups(source.manifest, [*edges, confirmed], policy, config)
    _, _, resolved = sensitivity_groups(
        source.manifest,
        [*edges, confirmed],
        "S3",
        replace(config, review_conflict_policy="reject_pair"),
    )
    assert not resolved["imposed_relation_ids"] and resolved["conflicting_review_pairs"]


def test_transitive_rejection_and_too_few_groups(audited, config):
    source = load_labeled_audit(audited[0])
    candidates = [e for e in source.relations if e["state"] == "visual_candidate"]
    a, b = candidates
    edge = {
        **a,
        "left_frame_id": a["right_frame_id"],
        "left_content_id": a["right_content_id"],
        "right_frame_id": b["right_frame_id"],
        "right_content_id": b["right_content_id"],
        "state": "rejected",
    }
    with pytest.raises(ValueError, match="Transitive"):
        sensitivity_groups(source.manifest, [*candidates, edge], "S3", config)
    groups, _, _ = sensitivity_groups(source.manifest, [], "S2", config)
    units = aggregate_groups(source.statistics, groups)
    with pytest.raises(ValueError, match="three atomic"):
        milp_assignment(units.iloc[:2], np.array([0.5, 0.25, 0.25]), 0, config.balance)


def test_no_feasible_solver_never_substitutes_random(audited, config, monkeypatch):
    import flir_pipeline.splitting.construction as construction

    monkeypatch.setattr(
        construction,
        "milp",
        lambda *a, **k: SimpleNamespace(x=None, status=2, message="infeasible"),
    )
    with pytest.raises(ValueError, match="no incumbent"):
        build_labeled_candidates(
            audited[0],
            audited[1] / "failed",
            config,
            populations=("expanded",),
            policies=("S2",),
        )
    assert not (audited[1] / "failed" / "summary.json").exists()


@pytest.mark.parametrize(
    "file", ["candidate_manifest.parquet", "annotation_audit.json", "relations.json"]
)
def test_audit_checksum_tampering_rejected(audited, file):
    path = audited[0] / file
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_labeled_audit(audited[0])


@pytest.mark.parametrize(
    "kind",
    [
        "annotation_counts",
        "annotation_bytes",
        "relation_lineage",
        "label_coverage",
        "adendo_split",
        "summary_count",
    ],
)
def test_semantic_corruption_even_with_updated_checksums(audited, kind):
    audit = audited[0]
    summary = read_json(audit / "summary.json")
    if kind.startswith("annotation") or kind == "label_coverage":
        name = "annotation_audit.json"
        payload = read_json(audit / name)
        row = next(r for r in payload["records"] if r["pairing_status"] == "matched")
        if kind == "annotation_counts":
            row["class_counts"] = {"0": 999}
        elif kind == "annotation_bytes":
            row["original_bytes_base64"] = "OTkgMC41IDAuNSAwLjIgMC4y"
        else:
            row["frame_ids"] = []
        write_json(audit / name, payload)
    elif kind == "relation_lineage":
        name = "relations.json"
        payload = read_json(audit / name)
        payload["records"][0]["left_content_id"] = "wrong"
        write_json(audit / name, payload)
    elif kind == "adendo_split":
        name = "candidate_manifest.json"
        payload = read_json(audit / name)
        next(r for r in payload["records"] if r["cohort"] == "adendo")[
            "original_split"
        ] = "train"
        write_json(audit / name, payload)
        pd.DataFrame(payload["records"]).to_parquet(
            audit / "candidate_manifest.parquet", index=False
        )
        summary["artifacts"]["candidate_manifest.parquet"] = file_sha256(
            audit / "candidate_manifest.parquet"
        )
    else:
        name = None
        summary["total_occurrences"] += 1
    if name:
        summary["artifacts"][name] = file_sha256(audit / name)
    write_json(audit / "summary.json", summary)
    with pytest.raises(ValueError):
        load_labeled_audit(audit)


def test_expected_ids_and_original_checksums(audited):
    audit, root, evidence = audited
    with pytest.raises(ValueError, match="Unexpected"):
        load_labeled_audit(audit, expected_dataset_id="wrong")
    with pytest.raises(ValueError, match="Unexpected"):
        load_labeled_audit(audit, expected_audit_id="wrong")
    source = load_labeled_audit(audit)
    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence checksum"):
        verify_original_sources(source, root, evidence)
    (root / "adendo.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="ZIP checksum"):
        verify_original_sources(source, root, evidence)


def test_metrics_tampering_rejected_after_receipt_repair(audited, config):
    audit, root, _ = audited
    output = root / "suite"
    receipt = build_labeled_candidates(
        audit, output, config, populations=("expanded",), policies=("S1",)
    )
    run = receipt["runs"][0]
    path = output / "runs" / run["split_id"] / "metrics.json"
    metrics = read_json(path)
    metrics["relations"]["visual_candidate"]["cross_split_occurrence_pairs"] = 999
    write_json(path, metrics)
    run["output_sha256"]["metrics.json"] = file_sha256(path)
    write_json(output / "summary.json", receipt)
    with pytest.raises(ValueError, match="Recomputed metrics"):
        verify_labeled_candidates(output)


def test_seed_batch_independence_and_no_historical_engine_dependency(
    audited, config, monkeypatch
):
    import flir_pipeline.splitting.storage as historical

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical experiments must not be loaded or rewritten")

    monkeypatch.setattr(historical, "load_inputs", forbidden)
    monkeypatch.setattr(historical, "build_run", forbidden)
    audit, root, _ = audited
    first = build_labeled_candidates(
        audit, root / "one", config, populations=("expanded",), policies=("S1",)
    )
    many = build_labeled_candidates(
        audit,
        root / "many",
        replace(config, balance=replace(config.balance, seeds=(0, 1))),
        populations=("expanded",),
        policies=("S1",),
    )
    assert first["runs"][0]["split_id"] == many["runs"][0]["split_id"]
    assert many["runs"][0]["split_id"] != many["runs"][1]["split_id"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"review_conflict_policy": "prefer_confirmed"},
        {"sensitivity_states": ["rejected"]},
        {"max_relative_record_deviation": -1},
    ],
)
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        LabeledConfig(**kwargs)


def test_explicit_ratios_and_sensitivity_state_selection(audited, config):
    source = load_labeled_audit(audited[0])
    changed = replace(
        config,
        balance=replace(config.balance, target_ratios=(0.6, 0.2, 0.2)),
        sensitivity_states=("confirmed_provenance",),
    )
    manifest, _, ratios = population(source, "expanded", changed)
    np.testing.assert_array_equal(ratios, [0.6, 0.2, 0.2])
    groups, _, selection = sensitivity_groups(manifest, source.relations, "S3", changed)
    assert groups.group_id.nunique() == manifest.content_id.nunique()
    assert selection["imposed_relation_ids"] == []


@pytest.mark.parametrize("kind", ["missing", "duplicate", "fractured", "objective"])
def test_membership_and_solver_corruption(audited, config, kind):
    audit, root, _ = audited
    output = root / "corrupt"
    receipt = build_labeled_candidates(
        audit, output, config, populations=("expanded",), policies=("S2",)
    )
    run = receipt["runs"][0]
    if kind == "objective":
        filename = "metadata.json"
        path = output / "runs" / run["split_id"] / filename
        meta = read_json(path)
        meta["solver"]["objective"] = 999
        write_json(path, meta)
    else:
        filename = "occurrence_memberships.parquet"
        path = output / "runs" / run["split_id"] / filename
        table = pd.read_parquet(path)
        if kind == "missing":
            table = table.iloc[1:]
        elif kind == "duplicate":
            table = pd.concat([table, table.iloc[:1]], ignore_index=True)
        else:
            index = table.index[table.content_id.duplicated()][0]
            table.loc[index, "new_split"] = (
                "test" if table.loc[index, "new_split"] != "test" else "val"
            )
        table.to_parquet(path, index=False)
    run["output_sha256"][filename] = file_sha256(path)
    write_json(output / "summary.json", receipt)
    with pytest.raises((ValueError, AssertionError)):
        verify_labeled_candidates(output)


def test_output_privacy_and_source_protection(audited, config):
    from pathlib import Path

    for output in (audited[0] / "inside", Path("artifacts/forbidden-labeled-test")):
        with pytest.raises(ValueError, match="outside the repository"):
            build_labeled_candidates(audited[0], output, config)
        assert not output.exists()


def test_cli_build_verify_and_scope(audited):
    audit, root, _ = audited
    config = root / "config.json"
    config.write_text(
        json.dumps({"balance": {"seeds": [0], "node_limit": 2}}), encoding="utf-8"
    )
    runner = CliRunner()
    for command in ("labeled-candidates", "verify-labeled"):
        assert runner.invoke(app, ["splitting", command, "--help"]).exit_code == 0
    result = runner.invoke(
        app,
        [
            "splitting",
            "labeled-candidates",
            "--audit",
            str(audit),
            "--config",
            str(config),
            "--population",
            "expanded",
            "--policy",
            "S1",
            "--seeds",
            "3",
            "--output",
            str(root / "cli"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        runner.invoke(
            app,
            [
                "splitting",
                "verify-labeled",
                str(root / "cli"),
                "--audit",
                str(audit),
                "--replay",
            ],
        ).exit_code
        == 0
    )
