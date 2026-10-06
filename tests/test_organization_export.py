"""Offline, synthetic evidence; never an export of the real Hypatia experiment."""

import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from test_explorer import experiment as explorer_experiment
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.clustering.base import ClusteringConfig, clustering_space_id
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.detection.final_report import write_payload
from flir_pipeline.explorer import organization
from flir_pipeline.explorer.discovery import read_json, sha256
from flir_pipeline.explorer.organization_contract import OrganizationExport, schema
from flir_pipeline.explorer.organization_sources import (
    CandidateEvidence,
    Sources,
    _occurrence_binding,
    experimental_candidates,
    load_candidates,
)
from flir_pipeline.sequences.experiments.artifacts import (
    binding,
    inspect,
    publish,
)
from flir_pipeline.sequences.experiments.artifacts import (
    tables as artifact_tables,
)
from flir_pipeline.similarity.storage import stable_id
from flir_pipeline.splitting.base import SplitConfig, identity_payload, split_space_id


def rebind(directory, name):
    meta = read_json(directory / "metadata.json")
    meta["output_sha256"][name] = sha256(directory / name)
    write_payload(directory / "metadata.json", meta)


def freeze(root, specs):
    config = {
        "training_seeds": [42, 43],
        "split_seeds": [0, 1],
        "train": {"batch": None},
    }
    identity = {"config": config, "splits": specs}
    plan = {"plan_id": stable_id(identity), "identity": identity}
    write_payload(root / "plan.json", plan)
    model = {"protocol_config": config, "device": "synthetic"}
    write_payload(
        root / "runtime_freeze.json",
        {
            "plan_id": plan["plan_id"],
            "model_config": model,
            "model_config_id": stable_id(model),
        },
    )


@pytest.fixture
def evidence(tmp_path):
    root, images, manifest, cluster, splits = explorer_experiment.__wrapped__(tmp_path)
    manifest["classes_present"] = "0|2"
    manifest["width"], manifest["height"] = 24, 16
    manifest.loc[manifest.frame_id.eq("duplicate"), "label_sha256"] = (
        "conflicting-label"
    )
    path = root / "manifest.parquet"
    manifest.to_parquet(path, index=False)
    dataset = dataset_id_from_manifest(manifest)
    cluster_ids = []
    for i in range(2):
        directory = cluster.directory if i == 0 else root / "clusters" / "second"
        if i:
            shutil.copytree(cluster.directory, directory)
        meta = read_json(directory / "metadata.json")
        config = ClusteringConfig(
            "hdbscan" if i else "optics", {"min_samples": 2, "min_cluster_size": 2}
        )
        effective = {"eps": 0.3 + i * 0.1}
        cid = clustering_space_id(
            dataset, "synthetic-features", "original_l2", None, config, effective, {}
        )
        meta.update(
            algorithm=config.algorithm,
            dataset_id=dataset,
            config=asdict(config),
            configuration_id=config.configuration_id,
            effective_parameters=effective,
            implementation_versions={},
            optional_files=[],
            feature_space_id="synthetic-features",
            reduction_space_id=None,
            clustering_space_id=cid,
        )
        if i:
            # This is a fresh synthetic copy, not a resumable scientific artifact.
            (directory / "algorithm_diagnostics.npz").unlink()
            meta["output_sha256"].pop("algorithm_diagnostics.npz")
            probabilities = directory / "membership_probabilities.npy"
            np.save(probabilities, np.array([0.9, 0.8, 1.0, 0.7, 0.0, 0.6]))
            meta["optional_files"] = [probabilities.name]
            meta["output_sha256"][probabilities.name] = sha256(probabilities)
        else:
            diagnostics = directory / "algorithm_diagnostics.npz"
            np.savez(
                diagnostics,
                reachability=np.array([np.inf, 0.2, 0.1, 0.3, 0.5, 0.7]),
                core_distances=np.array([0.1, 0.2, 0.3, 0.4, np.inf, 0.6]),
                ordering_rows=np.arange(6)[::-1],
            )
            meta["optional_files"] = [diagnostics.name]
            meta["output_sha256"][diagnostics.name] = sha256(diagnostics)
        write_payload(directory / "metadata.json", meta)
        cluster_ids.append(cid)
    specs = []
    for strategy, candidate, cluster_id in (
        ("historical", "historical", None),
        ("random_content", "random_content", None),
        ("cluster_aware", "C10", cluster_ids[0]),
        ("cluster_aware", "C12", cluster_ids[1]),
    ):
        original = next(s for s in splits if s.strategy == strategy)
        for seed in [0] if strategy == "historical" else [0, 1]:
            config = SplitConfig(strategy=strategy, seeds=(0, 1))
            identity = identity_payload(
                dataset, config, np.array([0.5, 0.25, 0.25]), seed, cluster_id
            )
            sid = split_space_id(identity)
            directory = root / "final-splits" / sid
            shutil.copytree(original.directory, directory)
            meta = read_json(directory / "metadata.json")
            meta.update(split_space_id=sid, identity_payload=identity)
            write_payload(directory / "metadata.json", meta)
            records = pd.read_parquet(directory / "record_split_assignments.parquet")
            if strategy != "historical" and seed:
                records["new_split"] = records.new_split.map(
                    {"train": "val", "val": "test", "test": "train"}
                )
                records.to_parquet(
                    directory / "record_split_assignments.parquet", index=False
                )
                rebind(directory, "record_split_assignments.parquet")
            contents = pd.read_parquet(directory / "source_groups.parquet")
            memberships = records.groupby("content_id").new_split.agg(
                lambda x: sorted(set(x))
            )
            contents["new_split"] = contents.content_id.map(
                memberships.map(lambda x: x[0] if len(x) == 1 else None)
            )
            contents["split_membership_set"] = contents.content_id.map(
                memberships.map(json.dumps)
            )
            contents.to_parquet(directory / "split_assignments.parquet", index=False)
            rebind(directory, "split_assignments.parquet")
            specs.append(
                dict(
                    strategy=candidate,
                    split_seed=seed,
                    split_space_id=sid,
                    dataset_id=dataset,
                    clustering_space_id=cluster_id,
                    split_metadata_sha256=sha256(directory / "metadata.json"),
                    record_counts={
                        s: int(records.new_split.eq(s).sum())
                        for s in ("train", "val", "test")
                    },
                    class_counts=[{"split": "train", "class_id": 0, "instances": 2}],
                )
            )
    plan = root / "plan"
    plan.mkdir()
    freeze(plan, specs)
    occurrences = manifest.assign(
        timeline_id=manifest.possible_sequence, position=manifest.possible_frame_index
    )
    signature = {"dataset_id": dataset, "checksums": {"manifest_sha256": sha256(path)}}
    intervals = pd.DataFrame(
        [
            dict(
                element_id="zone-A",
                timeline_id="A",
                start=10,
                end=15,
                kind="boundary_zone",
                decision="ambiguous",
                notes="inclusive uncertainty",
            )
        ]
    )
    cores = pd.DataFrame(
        [
            dict(
                element_id="core-A",
                timeline_id="A",
                start=1,
                end=40,
                kind="sequence_core_candidate",
                decision="candidate",
                notes="synthetic",
                status="candidate",
                origin="fixture",
            ),
            dict(
                element_id="core-B",
                timeline_id="B",
                start=2,
                end=4,
                kind="sequence_core_candidate",
                decision="candidate",
                notes="synthetic",
                status="candidate",
                origin="fixture",
            ),
        ]
    )
    membership = occurrences.loc[
        occurrences.timeline_id.isin(["A", "B"]),
        ["frame_id", "content_id", "timeline_id", "position"],
    ].copy()
    membership["target"], membership["evaluation_mask"] = "sequence_core", True
    membership["target_label"] = membership.timeline_id.map(
        {"A": "core-A", "B": "core-B"}
    )
    linkage = root / "evidence"
    structure = publish(
        linkage,
        "sequence_structure_review_v1",
        {},
        {"input": signature},
        {
            "occurrences": occurrences,
            "intervals": intervals,
            "cores": cores,
            "membership": membership,
        },
        {},
        {},
    )
    component = pd.DataFrame(
        {
            "core_id": ["core-A", "core-B"],
            "diagnostic_component_id": ["component-1"] * 2,
            "diagnostic_only": [True] * 2,
            "dependency_confirmed": [False] * 2,
        }
    )
    core_contents = (
        membership[["target_label", "content_id"]]
        .drop_duplicates()
        .rename(columns={"target_label": "core_id"})
    )
    recurrence = publish(
        linkage,
        "sequence_recurrence_v1",
        {},
        {"input": signature, "structure": binding(structure)},
        {
            "occurrences": occurrences,
            "diagnostic_components": component,
            "core_content_membership": core_contents,
            "structure_elements": cores,
        },
        {},
    )
    args = dict(
        manifest_path=path,
        plan_directory=plan,
        split_root=root / "final-splits",
        clustering_root=root / "clusters",
        linkage_root=linkage,
        output=root / "export",
        media_output=root / "media",
        data_root=images,
    )
    return args, manifest, specs, structure, recurrence


def payload(args):
    return {
        k: read_json(args["output"] / f"{k}.json")
        for k in OrganizationExport.model_fields
    }


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): (sha256(p), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_export_preserves_records_contents_splits_and_selected_clusters(evidence):
    args, manifest, _, _, _ = evidence
    before = {
        key: snapshot(args[key])
        for key in ("plan_directory", "split_root", "clustering_root", "linkage_root")
    }
    result = organization.export_organization(**args)
    data = payload(args)
    assert result["labeled_record_count"] == 8
    assert result["labeled_unique_content_count"] == 6
    assert result["split_count"] == 7
    assert result["strategy_count"] == 4
    duplicate = manifest.iloc[0].content_id
    assert {
        r["partition"]
        for r in data["split_memberships"]
        if r["strategy"] == "historical" and r["content_id"] == duplicate
    } == {"train", "test"}
    assert len(data["split_memberships"]) == 7 * 8
    assert len(data["cluster_memberships"]) == 2 * 6
    assert sum(r["probability"] is not None for r in data["cluster_memberships"]) == 6
    assert (
        sum(r["reachability_infinite"] is True for r in data["cluster_memberships"])
        == 1
    )
    assert all(
        r["reachability"] is None
        for r in data["cluster_memberships"]
        if r["reachability_infinite"]
    )
    assert sum(c["is_noise"] for c in data["clusters"]) == 2
    assert all(
        c["source_kind"] == "full_clustering_artifact"
        and c["full_clustering_artifact_available"]
        and c["algorithm"] is not None
        for c in data["clustering_configurations"]
    )
    assert any(
        c["annotation_consensus"] == "different_label_bytes" and c["class_ids"] is None
        for c in data["contents"]
    )
    assert len(data["timelines"]) == 3 and result["source_video_count"] == 0
    assert all(t["source_video_id"] is None for t in data["timelines"])
    assert all(not r["media_available"] for r in data["media"])
    assert before == {key: snapshot(args[key]) for key in before}
    assert all(
        sha256(args["output"] / name) == digest
        for name, digest in result["file_sha256"].items()
    )
    Draft202012Validator.check_schema(schema())
    Draft202012Validator(schema()).validate(data)
    OrganizationExport.model_validate(data)


def test_candidates_noncontiguous_and_boundary_zones_remain_diagnostic(evidence):
    args, _, _, structure, recurrence = evidence
    organization.export_organization(**args)
    data = payload(args)
    group = next(
        g for g in data["linkage_groups"] if g["kind"] == "diagnostic_component"
    )
    assert group["linkage_group_id"] == "component-1" and group["member_count"] == 5
    assert not group["ground_truth"] and not group["automatic_confirmation"]
    stored = artifact_tables(structure)
    cores = {
        g["linkage_group_id"]: g
        for g in data["linkage_groups"]
        if g["kind"] == "candidate_core"
    }
    assert set(cores) == set(stored["cores"].element_id)
    for row in stored["cores"].to_dict("records"):
        assert cores[row["element_id"]]["upstream_metadata"] == row
    assigned = stored["membership"].loc[stored["membership"].evaluation_mask]
    expected = {
        (core, content): sorted(rows.frame_id)
        for (core, content), rows in assigned.groupby(["target_label", "content_id"])
    }
    actual = {
        (m["linkage_group_id"], m["content_id"]): m["record_ids"]
        for m in data["linkage_memberships"]
        if m["evidence_artifact_id"] == inspect(structure)["artifact_id"]
    }
    assert actual == expected
    component_members = [
        m
        for m in data["linkage_memberships"]
        if m["evidence_artifact_id"] == inspect(recurrence)["artifact_id"]
    ]
    assert {m["content_id"]: m["record_ids"] for m in component_members} == {
        content: sorted(rows.frame_id)
        for content, rows in assigned.groupby("content_id")
    }
    assert all(m["role"] == "core_member" for m in data["linkage_memberships"])
    assert not data["candidate_pairs"]
    zone = data["boundary_zones"][0]
    assert (zone["start"], zone["end"], zone["inclusive"], zone["exact_cut"]) == (
        10,
        15,
        True,
        False,
    )
    timeline = next(t for t in data["timelines"] if t["inferred_family"] == "A")
    assert [p["frame_index"] for p in timeline["points"]] == [1, 3, 40]
    assert len(timeline["points"][0]["record_ids"]) == 2
    assert all(
        not data["manifest"]["semantics"][flag]
        for flag in (
            "ground_truth_clusters",
            "ground_truth_sequences",
            "automatic_confirmation",
            "sequence_instances_created",
            "clusters_are_sequences",
        )
    )
    assert not any("sequence_id" in r for r in data["clusters"])
    assert not any("split" in r for r in timeline["points"])


def test_preview_generated_once_per_content_and_missing_source_degrades(evidence):
    args, _, _, _, _ = evidence
    organization.export_organization(**args, include_previews=True)
    data = payload(args)
    assert all(m["media_available"] and m["width"] == 24 for m in data["media"])
    assert len(list(args["media_output"].glob("*.jpg"))) == 6
    assert all(
        sha256(args["media_output"] / m["relative_path"]) == m["checksum"]
        for m in data["media"]
    )
    before = {k: v for k, v in data.items() if k not in {"manifest", "media"}}
    organization.export_organization(
        **{**args, "data_root": args["data_root"] / "absent"}, include_previews=True
    )
    after = payload(args)
    assert before == {k: v for k, v in after.items() if k not in {"manifest", "media"}}
    assert all(
        not m["media_available"] and m["relative_path"] is None for m in after["media"]
    )


def test_deterministic_bytes_and_readonly_sources(evidence):
    args, _, _, _, _ = evidence
    organization.export_organization(**args, include_previews=True)
    before = {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }
    media_before = {p.name: p.read_bytes() for p in args["media_output"].glob("*.jpg")}
    organization.export_organization(**args, include_previews=True)
    assert before == {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }
    assert media_before == {
        p.name: p.read_bytes() for p in args["media_output"].glob("*.jpg")
    }


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_record",
        "duplicate_content",
        "bad_content",
        "bad_partition",
        "missing_cluster",
        "duplicate_run",
        "bad_plan",
        "bad_checksum",
    ],
)
def test_integrity_failures_preserve_previous_export(evidence, failure):
    args, manifest, specs, _, _ = evidence
    organization.export_organization(**args)
    before = snapshot(args["output"])
    directory = args["split_root"] / specs[0]["split_space_id"]
    if failure == "duplicate_record":
        pd.concat([manifest, manifest.iloc[:1]]).to_parquet(
            args["manifest_path"], index=False
        )
    elif failure == "missing_cluster":
        (args["clustering_root"] / "run" / "cluster_labels.npy").unlink()
    elif failure == "duplicate_run":
        shutil.copytree(directory, args["split_root"] / "duplicate")
    elif failure == "bad_plan":
        plan = read_json(args["plan_directory"] / "plan.json")
        plan["identity"]["splits"][0]["split_space_id"] = "missing"
        write_payload(args["plan_directory"] / "plan.json", plan)
    else:
        name = (
            "split_assignments.parquet"
            if failure == "duplicate_content"
            else "record_split_assignments.parquet"
        )
        path = directory / name
        rows = pd.read_parquet(path)
        if failure == "duplicate_content":
            rows = pd.concat([rows, rows.iloc[:1]])
        elif failure == "bad_content":
            rows.loc[0, "content_id"] = "unknown"
        else:
            rows.loc[0, "new_split"] = "invalid"
        rows.to_parquet(path, index=False)
        if failure != "bad_checksum":
            rebind(directory, name)
            specs[0]["split_metadata_sha256"] = sha256(directory / "metadata.json")
            freeze(args["plan_directory"], specs)
    with pytest.raises((ValueError, KeyError)):
        organization.export_organization(**args)
    assert snapshot(args["output"]) == before


@pytest.mark.parametrize(
    "table,field,value",
    [
        ("cluster_memberships", "content_id", "unknown"),
        ("linkage_memberships", "content_id", "unknown"),
        ("split_memberships", "partition", "validation"),
        ("clusters", "ground_truth", True),
        ("linkage_groups", "automatic_confirmation", True),
        ("linkage_groups", "kind", "candidate_pair"),
        ("linkage_memberships", "role", "query"),
        ("linkage_memberships", "role", "candidate"),
        ("linkage_memberships", "role", "query_and_candidate"),
    ],
)
def test_contract_rejects_unknown_references_or_scientific_promotion(
    evidence, table, field, value
):
    args, *_ = evidence
    organization.export_organization(**args)
    data = payload(args)
    data[table][0][field] = value
    with pytest.raises(ValueError):
        OrganizationExport.model_validate(data)


@pytest.mark.parametrize(
    "table", ["contents", "records", "clusters", "linkage_groups", "split_memberships"]
)
def test_contract_duplicate_primary_ids(evidence, table):
    args, *_ = evidence
    organization.export_organization(**args)
    data = payload(args)
    data[table].append(copy.deepcopy(data[table][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        OrganizationExport.model_validate(data)


def test_publication_io_failure_rolls_back_both_destinations(evidence, monkeypatch):
    args, *_ = evidence
    organization.export_organization(**args, include_previews=True)
    before = {key: snapshot(args[key]) for key in ("output", "media_output")}
    original = Path.rename

    def fail(path, target):
        if path.name.startswith(".export-stage-") and not path.name.endswith(
            "-previous"
        ):
            raise OSError("synthetic publish failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="synthetic publish"):
        organization.export_organization(**args, include_previews=True)
    assert before == {key: snapshot(args[key]) for key in before}


def test_concurrent_source_change_cancels_publication(evidence, monkeypatch):
    args, *_ = evidence
    organization.export_organization(**args)
    before = snapshot(args["output"])
    original = organization.preview_media

    def change(*a, **kw):
        result = original(*a, **kw)
        with (args["plan_directory"] / "plan.json").open("a") as stream:
            stream.write(" ")
        return result

    monkeypatch.setattr(organization, "preview_media", change)
    with pytest.raises(ValueError, match="Source changed"):
        organization.export_organization(**args)
    assert snapshot(args["output"]) == before


def test_destinations_protect_sources_unrelated_files_and_media_policy(evidence):
    args, *_ = evidence
    with pytest.raises(ValueError, match="disjoint"):
        organization.export_organization(**{**args, "output": args["split_root"]})
    args["output"].mkdir()
    (args["output"] / "keep.txt").write_text("user file")
    with pytest.raises(ValueError, match="unrelated"):
        organization.export_organization(**args)
    assert (args["output"] / "keep.txt").read_text() == "user file"
    with pytest.raises(ValueError, match="ignored artifacts"):
        organization.export_organization(
            **{**args, "media_output": Path("exports/frontend/private-images")}
        )


def test_cli_schema_and_missing_optional_evidence(evidence):
    args, *_ = evidence
    args["linkage_root"] = args["output"].parent / "absent"
    result = CliRunner().invoke(
        app,
        [
            "explorer",
            "export-organization",
            "--manifest",
            str(args["manifest_path"]),
            "--plan",
            str(args["plan_directory"]),
            "--split-root",
            str(args["split_root"]),
            "--clustering-root",
            str(args["clustering_root"]),
            "--linkage-root",
            str(args["linkage_root"]),
            "--output",
            str(args["output"]),
            "--media-output",
            str(args["media_output"]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert not payload(args)["linkage_groups"]
    assert "No candidate" in payload(args)["manifest"]["limitations"][-1]
    checked_in = Path(
        "exports/frontend/organization/schema/organization-evidence-v2.schema.json"
    )
    assert read_json(checked_in) == schema()
    assert (
        CliRunner().invoke(app, ["explorer", "export-organization", "--help"]).exit_code
        == 0
    )


@pytest.mark.parametrize("fallback", [False, True])
def test_no_fitting_or_scientific_metric_recomputation(evidence, monkeypatch, fallback):
    from flir_pipeline.clustering import storage as clustering
    from flir_pipeline.linkage import candidates
    from flir_pipeline.sequences.experiments import recurrence
    from flir_pipeline.splitting import construction, storage

    def reject(*a, **kw):
        raise AssertionError("Scientific recomputation forbidden")

    for module, names in (
        (clustering, ["fit_clustering", "evaluate_clustering"]),
        (candidates, ["generate_candidates"]),
        (recurrence, ["diagnostic_components", "recurrence_tables"]),
        (construction, ["random_assignment", "milp_assignment"]),
        (storage, ["evaluate_records", "build_run"]),
    ):
        for name in names:
            monkeypatch.setattr(module, name, reject)
    args = evidence[0]
    if fallback:
        args = {**args, "clustering_root": args["clustering_root"].parent / "absent"}
    organization.export_organization(**args)


def test_source_receipt_detects_replacement(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}")
    sources = Sources()
    sources.watch(path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="Source changed"):
        sources.watch(path)


@pytest.fixture
def frozen_evidence(evidence):
    args, *rest = evidence
    return (
        {**args, "clustering_root": args["clustering_root"].parent / "absent"},
        *rest,
    )


def edit_selected_groups(evidence, change):
    """Publish changed synthetic bytes and re-freeze, to test beyond checksums."""
    args, _, specs, *_ = evidence
    spec = next(s for s in specs if s["strategy"] == "C10" and s["split_seed"] == 1)
    directory = args["split_root"] / spec["split_space_id"]
    path = directory / "source_groups.parquet"
    change(pd.read_parquet(path)).to_parquet(path, index=False)
    rebind(directory, path.name)
    spec["split_metadata_sha256"] = sha256(directory / "metadata.json")
    freeze(args["plan_directory"], specs)


def test_frozen_export_preserves_exact_labels_nulls_and_all_provenance(frozen_evidence):
    args, manifest, specs, *_ = frozen_evidence
    # File ordering is operational, not membership: sort only before comparing.
    edit_selected_groups(frozen_evidence, lambda g: g.iloc[::-1])
    before = snapshot(args["split_root"])
    organization.export_organization(**args)
    data = payload(args)
    for config in data["clustering_configurations"]:
        cid = config["cluster_run_id"]
        selected = [s for s in specs if s["clustering_space_id"] == cid]
        assert config["source_kind"] == "frozen_split_membership"
        assert config["full_clustering_artifact_available"] is False
        assert config["membership_consistency_verified"] is True
        assert config["source_split_ids"] == sorted(
            s["split_space_id"] for s in selected
        )
        assert config["source_membership_checksums"] == {
            s["split_space_id"]: sha256(
                args["split_root"] / s["split_space_id"] / "source_groups.parquet"
            )
            for s in selected
        }
        assert all(
            config[k] is None
            for k in (
                "feature_space_id",
                "configuration_id",
                "model_id",
                "representation",
                "reduction_space_id",
                "reduction_seed",
                "extractor",
                "algorithm",
                "parameters",
                "effective_parameters",
            )
        )
        stored = (
            pd.read_parquet(
                args["split_root"]
                / selected[0]["split_space_id"]
                / "source_groups.parquet"
            )
            .set_index("content_id")
            .cluster_id.to_dict()
        )
        exported = [
            m for m in data["cluster_memberships"] if m["cluster_run_id"] == cid
        ]
        assert {m["content_id"]: m["cluster_id"] for m in exported} == stored
        assert set(stored) == set(manifest.content_id)
        assert all(m["is_noise"] == (m["cluster_id"] == -1) for m in exported)
        assert all(
            m[k] is None
            for m in exported
            for k in (
                "probability",
                "reachability",
                "reachability_infinite",
                "core_distance",
                "core_distance_infinite",
                "ordering_position",
            )
        )
    assert not any(
        s["artifact_kind"] == "clustering_run"
        for s in data["manifest"]["evidence_sources"]
    )
    assert any(
        "publication" in s and "unavailable" in s
        for s in data["manifest"]["limitations"]
    )
    assert before == snapshot(args["split_root"])
    Draft202012Validator(schema()).validate(data)
    OrganizationExport.model_validate(data)


@pytest.mark.parametrize(
    "column,value",
    [
        ("cluster_id", 19),
        ("group_id", "different-group"),
        ("group_type", "different-type"),
    ],
)
def test_frozen_cross_seed_disagreement_fails(frozen_evidence, column, value):
    def change(groups):
        groups.loc[groups.cluster_id.eq(0), column] = value
        return groups

    edit_selected_groups(frozen_evidence, change)
    with pytest.raises(ValueError, match="Frozen memberships differ"):
        organization.export_organization(**frozen_evidence[0])


@pytest.mark.parametrize(
    "failure",
    [
        "missing_content",
        "duplicate_content",
        "unknown_content",
        "negative_label",
        "fractional_label",
        "string_label",
        "boolean_label",
        "null_label",
        "null_group",
        "null_type",
        "noise_group",
        "fractured_cluster",
        "missing_content_id",
        "missing_cluster_id",
        "missing_group_id",
        "missing_group_type",
    ],
)
def test_invalid_frozen_membership_fails_closed(frozen_evidence, failure):
    def change(g):
        if failure == "missing_content":
            return g.iloc[1:]
        if failure == "duplicate_content":
            return pd.concat([g, g.iloc[:1]])
        if failure.startswith("missing_"):
            return g.drop(columns=failure.removeprefix("missing_"))
        if failure == "unknown_content":
            g.loc[0, "content_id"] = "unknown"
        elif failure == "negative_label":
            g.loc[0, "cluster_id"] = -2
        elif failure == "fractional_label":
            g["cluster_id"] = g.cluster_id.astype(float) + 0.5
        elif failure == "string_label":
            g["cluster_id"] = g.cluster_id.astype(str)
        elif failure == "boolean_label":
            g["cluster_id"] = True
        elif failure == "null_label":
            g["cluster_id"] = g.cluster_id.astype("Int64")
            g.loc[0, "cluster_id"] = pd.NA
        elif failure == "null_group":
            g.loc[0, "group_id"] = None
        elif failure == "null_type":
            g.loc[0, "group_type"] = None
        elif failure == "noise_group":
            g.loc[g.cluster_id.eq(-1), "group_type"] = "cluster"
        elif failure == "fractured_cluster":
            g.loc[g.cluster_id.eq(1), "cluster_id"] = 0
        return g

    edit_selected_groups(frozen_evidence, change)
    with pytest.raises(ValueError):
        organization.export_organization(**frozen_evidence[0])
    assert not frozen_evidence[0]["output"].exists()


def test_frozen_scientific_json_independent_of_discovery_order(
    frozen_evidence, monkeypatch
):
    from flir_pipeline.explorer import organization_sources

    args = frozen_evidence[0]
    organization.export_organization(**args)
    before = {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }
    discover = organization_sources.discover_runs

    def reversed_discovery(*a, **kw):
        runs, issues = discover(*a, **kw)
        return runs[::-1], issues

    monkeypatch.setattr(organization_sources, "discover_runs", reversed_discovery)
    organization.export_organization(**args)
    assert before == {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }


def test_full_and_frozen_sources_can_coexist(evidence):
    args = evidence[0]
    # Only relocate a newly created synthetic publication, outside discovery.
    (args["clustering_root"] / "second").rename(
        args["clustering_root"].parent / "outside"
    )
    organization.export_organization(**args)
    configs = payload(args)["clustering_configurations"]
    assert {c["source_kind"] for c in configs} == {
        "full_clustering_artifact",
        "frozen_split_membership",
    }


def test_frozen_split_checksum_tampering_fails(frozen_evidence):
    args, _, specs, *_ = frozen_evidence
    spec = next(s for s in specs if s["strategy"] == "C10")
    path = args["split_root"] / spec["split_space_id"] / "source_groups.parquet"
    groups = pd.read_parquet(path)
    groups.loc[groups.cluster_id.eq(0), "cluster_id"] = 99
    groups.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="checksum"):
        organization.export_organization(**args)


def test_unselected_split_cannot_supply_or_override_frozen_membership(frozen_evidence):
    args, _, specs, *_ = frozen_evidence
    spec = next(s for s in specs if s["strategy"] == "C10")
    original = args["split_root"] / spec["split_space_id"]
    unselected = args["split_root"] / "unselected"
    shutil.copytree(original, unselected)
    meta = read_json(unselected / "metadata.json")
    meta["identity_payload"]["seed"] = 99
    meta["split_space_id"] = split_space_id(meta["identity_payload"])
    write_payload(unselected / "metadata.json", meta)
    path = unselected / "source_groups.parquet"
    groups = pd.read_parquet(path)
    groups.loc[groups.cluster_id.eq(0), "cluster_id"] = 99
    groups.to_parquet(path, index=False)
    rebind(unselected, path.name)
    organization.export_organization(**args)
    assert all(c["cluster_id"] != 99 for c in payload(args)["cluster_memberships"])
    # A valid unselected publication cannot stand in for a missing selected split.
    original.rename(args["output"].parent / "selected-outside-root")
    with pytest.raises(ValueError, match="Cannot resolve selected split"):
        organization.export_organization(**args)


@pytest.mark.parametrize(
    "failure",
    ["checksum", "incomplete", "quality", "metadata", "no_metadata", "duplicate"],
)
def test_present_invalid_clustering_never_falls_back(evidence, failure):
    args = evidence[0]
    directory = args["clustering_root"] / "run"
    if failure == "checksum":
        np.save(directory / "cluster_labels.npy", np.zeros(6, dtype=int))
    elif failure == "incomplete":
        (directory / "cluster_labels.npy").unlink()
    elif failure == "quality":
        write_payload(directory / "quality.json", {"quality_valid": False})
        rebind(directory, "quality.json")
    elif failure == "metadata":
        (directory / "metadata.json").write_text("{broken")
    elif failure == "no_metadata":
        (directory / "metadata.json").unlink()
    else:
        shutil.copytree(directory, args["clustering_root"] / "copy")
    with pytest.raises(ValueError):
        organization.export_organization(**args)


@pytest.mark.parametrize(
    "failure", ["configuration", "diagnostic", "provenance", "digest", "availability"]
)
def test_frozen_contract_rejects_speculative_or_incomplete_evidence(
    frozen_evidence, failure
):
    args = frozen_evidence[0]
    organization.export_organization(**args)
    data = payload(args)
    config = data["clustering_configurations"][0]
    if failure == "configuration":
        config["algorithm"] = "dbscan"
    elif failure == "diagnostic":
        data["cluster_memberships"][0]["reachability"] = 0.5
    elif failure == "provenance":
        sid = config["source_split_ids"].pop()
        del config["source_membership_checksums"][sid]
    elif failure == "digest":
        config["source_membership_checksums"][config["source_split_ids"][0]] = "0" * 64
    else:
        config["full_clustering_artifact_available"] = True
    with pytest.raises(ValueError):
        OrganizationExport.model_validate(data)


@pytest.mark.parametrize("reserialized", [False, True])
def test_sequence_manifest_reserialization_export_provenance_and_determinism(
    evidence, reserialized
):
    args, manifest, _, structure, recurrence = evidence
    historical = sha256(args["manifest_path"])
    if reserialized:
        manifest.to_parquet(args["manifest_path"], index=False, compression="gzip")
    current = sha256(args["manifest_path"])
    assert (historical != current) is reserialized
    before = {
        key: snapshot(args[key])
        for key in ("plan_directory", "split_root", "clustering_root", "linkage_root")
    }
    manifest_before = (current, args["manifest_path"].stat().st_mtime_ns)
    organization.export_organization(**args)
    data = payload(args)
    ids = {inspect(p)["artifact_id"] for p in (structure, recurrence)}
    receipts = [
        s for s in data["manifest"]["evidence_sources"] if s["artifact_id"] in ids
    ]
    assert len(receipts) == 2
    for receipt in receipts:
        assert receipt["labeled_manifest_binding"] == {
            "historical_manifest_sha256": historical,
            "current_manifest_sha256": current,
            "exact_occurrence_binding_verified": True,
            "source_manifest_reserialized": reserialized,
        }
        assert "occurrences.parquet" in receipt["output_checksums"]
    assert (
        any(
            "sequence evidence sources recorded" in s
            for s in data["manifest"]["limitations"]
        )
        is reserialized
    )
    Draft202012Validator(schema()).validate(data)
    scientific = {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }
    organization.export_organization(**args)
    assert scientific == {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }
    assert (
        data["manifest"]["evidence_sources"]
        == payload(args)["manifest"]["evidence_sources"]
    )
    assert before == {key: snapshot(args[key]) for key in before}
    assert manifest_before == (
        sha256(args["manifest_path"]),
        args["manifest_path"].stat().st_mtime_ns,
    )


@pytest.mark.parametrize("reserialized", [False, True])
def test_sequence_manifest_opt_in_preserves_default_and_global_checks(
    evidence, reserialized
):
    args, manifest, _, structure, recurrence = evidence
    if reserialized:
        manifest.to_parquet(args["manifest_path"], index=False, compression="gzip")
    digest = sha256(args["manifest_path"])
    meta = inspect(structure)
    frame = artifact_tables(structure)["occurrences"]
    signature = meta["identity"]["sources"]["input"]
    calls = [
        lambda: _occurrence_binding(frame, manifest, signature, digest),
        lambda: experimental_candidates(
            structure,
            manifest,
            digest,
            Sources(),
            CandidateEvidence(),
            [structure, recurrence],
        ),
        lambda: load_candidates(
            [args["linkage_root"]], args["manifest_path"], manifest, Sources()
        ),
    ]
    for call in calls:
        if reserialized:
            with pytest.raises(ValueError, match="different manifest"):
                call()
        else:
            call()
    result = load_candidates(
        [args["linkage_root"]],
        args["manifest_path"],
        manifest,
        Sources(),
        allow_labeled_manifest_reserialization=True,
    )
    assert result.groups
    # The scientific inspector still requires exact source signatures.
    current_sources = copy.deepcopy(meta["identity"]["sources"])
    current_sources["input"]["checksums"]["manifest_sha256"] = digest
    if reserialized:
        with pytest.raises(ValueError, match="not bound"):
            inspect(structure, sources=current_sources)
    else:
        inspect(structure, sources=current_sources)


@pytest.mark.parametrize("kind", ["structure", "recurrence"])
@pytest.mark.parametrize(
    "failure,message",
    [
        ("dataset", "different manifest"),
        ("unknown_frame", "unknown/duplicate"),
        ("content", "content mapping changed"),
        ("timeline", "timeline differs"),
        ("position", "positions differ"),
        ("duplicate", "unknown/duplicate"),
    ],
)
def test_sequence_reserialization_binding_mismatches_fail_closed(
    evidence, kind, failure, message
):
    args, manifest, _, structure, recurrence = evidence
    manifest.to_parquet(args["manifest_path"], index=False, compression="gzip")
    original = structure if kind == "structure" else recurrence
    meta = inspect(original)
    tables = artifact_tables(original)
    sources = copy.deepcopy(meta["identity"]["sources"])
    frame = tables["occurrences"].copy()
    if failure == "dataset":
        sources["input"]["dataset_id"] = "different-dataset"
    elif failure == "unknown_frame":
        frame.loc[0, "frame_id"] = "unknown"
    elif failure == "content":
        frame.loc[0, "content_id"] = "different-content"
    elif failure == "timeline":
        frame.loc[0, "timeline_id"] = "different-timeline"
    elif failure == "position":
        frame.loc[0, "position"] += 1
    else:
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    tables["occurrences"] = frame
    # Fresh immutable synthetic publications pass inspection; failure must be
    # the semantic binding, not an intentionally stale artifact checksum.
    changed = publish(
        args["output"].parent / "changed",
        meta["artifact_kind"],
        meta["identity"]["config"],
        sources,
        tables,
        {},
    )
    inspect(changed)
    roots = [changed] if kind == "structure" else [changed, structure]
    with pytest.raises(ValueError, match=message):
        load_candidates(
            roots,
            args["manifest_path"],
            manifest,
            Sources(),
            allow_labeled_manifest_reserialization=True,
        )


@pytest.mark.parametrize("kind", ["structure", "recurrence"])
@pytest.mark.parametrize("file", ["occurrences.parquet", "receipt.json"])
def test_sequence_reserialization_never_bypasses_artifact_integrity(
    evidence, kind, file
):
    args, manifest, _, structure, recurrence = evidence
    manifest.to_parquet(args["manifest_path"], index=False, compression="gzip")
    directory = structure if kind == "structure" else recurrence
    if file == "receipt.json":
        write_payload(directory / file, {"metadata_sha256": "0" * 64})
    else:
        with (directory / file).open("ab") as stream:
            stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum|receipt|Modified artifact file"):
        organization.export_organization(**args)


def test_sequence_normalized_occurrences_do_not_claim_full_table_identity(evidence):
    args, manifest, _, structure, _ = evidence
    meta = inspect(structure)
    tables = artifact_tables(structure)
    tables["occurrences"] = tables["occurrences"][
        ["frame_id", "content_id", "timeline_id", "position"]
    ]
    normalized = publish(
        args["output"].parent / "normalized",
        meta["artifact_kind"],
        meta["identity"]["config"],
        meta["identity"]["sources"],
        tables,
        {},
    )
    manifest.to_parquet(args["manifest_path"], index=False, compression="gzip")
    organization.export_organization(**{**args, "linkage_root": normalized})
    data = payload(args)
    receipt = next(
        s
        for s in data["manifest"]["evidence_sources"]
        if s["artifact_id"] == normalized.name
    )
    receipt["labeled_manifest_binding"].pop("exact_occurrence_binding_verified")
    receipt["labeled_manifest_binding"]["exact_tabular_identity_verified"] = True
    with pytest.raises(ValueError, match="verification scope"):
        OrganizationExport.model_validate(data)
