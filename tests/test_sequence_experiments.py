"""Offline semantic validation: synthetic visual vectors, no pretrained models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_sequences import synthetic_sources
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.sequences.experiments.artifacts import inspect, tables
from flir_pipeline.sequences.experiments.boundary import boundary_tables
from flir_pipeline.sequences.experiments.config import (
    SEMANTICS,
    BoundaryConfig,
    ClusterGrid,
    RecurrenceConfig,
    ReductionGrid,
    SuiteConfig,
    TransitionConfig,
)
from flir_pipeline.sequences.experiments.evaluation import (
    evaluation_tables,
    stability_tables,
)
from flir_pipeline.sequences.experiments.fitting import clustering_tables, fit_visual
from flir_pipeline.sequences.experiments.recurrence import (
    cluster_recurrence,
    recurrence_tables,
)
from flir_pipeline.sequences.experiments.runner import run_boundary, suite, verify
from flir_pipeline.sequences.experiments.sources import (
    ExperimentSources,
    load_sources,
    scientific_input_id,
)
from flir_pipeline.sequences.experiments.structure import (
    EvidenceEnvelope,
    Interval,
    content_targets,
    expected_binding,
    import_evidence,
    structure_membership,
    validate_intervals,
)
from flir_pipeline.sequences.experiments.transitions import (
    compare_zones,
    transition_tables,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def synthetic_memory(
    pattern=(0,) * 24 + (1,) * 24 + (0,) * 24, *, weak_encoder=False, duplicate=False
):
    n = len(pattern)
    random = np.random.default_rng(37)
    vectors = np.eye(8)[list(pattern)] + random.normal(0, 0.005, (n, 8))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(
        np.float32
    )
    ids = [f"c{i:04d}" for i in range(n)]
    records = pd.DataFrame(
        {
            "frame_id": [f"f{i}" for i in range(n)],
            "content_id": ids,
            "timeline_id": "synthetic",
            "family": "synthetic",
            "position": range(n),
            "temporal_source": "synthetic_validation",
            "content_row": range(n),
            "original_split": "train",
        }
    )
    if duplicate:
        # Last A image repeats the FIRST exact content, at a distinct position.
        records.loc[n - 1, ["content_id", "content_row"]] = [ids[0], 0]
        vectors = vectors[:-1]
        ids = ids[:-1]
        historical = records.iloc[[0]].copy()
        historical["frame_id"] = "historical-copy"
        historical["original_split"] = "test"
        records = pd.concat([records, historical], ignore_index=True)
    dino = (
        np.tile(np.eye(8, dtype=np.float32)[0], (len(ids), 1))
        if weak_encoder
        else vectors.copy()
    )
    return ExperimentSources(
        records,
        ids,
        {"clip": vectors, "dinov2": dino},
        {
            "dataset_id": "synthetic-dataset",
            "checksums": {"manifest_sha256": "a" * 64},
            "feature_spaces": {
                "clip": {"synthetic": True},
                "dinov2": {"synthetic": True},
            },
        },
        "synthetic",
    )


def zones(source):
    return pd.DataFrame(
        [
            dict(
                element_id=f"z{i}",
                timeline_id="synthetic",
                start=a,
                end=b,
                kind="boundary_zone",
                decision="supported",
                notes="synthetic review",
            )
            for i, (a, b) in enumerate(((23, 24), (47, 48)))
        ],
        columns=list(Interval.model_fields),
    )


def assignments(source, labels):
    return pd.DataFrame(
        {
            "run_id": "run",
            "content_id": source.contents,
            "cluster_id": np.asarray(labels, dtype=int),
        }
    )


def small_config(reductions=()):
    return SuiteConfig(
        reductions=reductions,
        clustering=(
            ClusterGrid(
                algorithm="dbscan",
                parameters={"min_samples": 3},
                grid={"eps_quantile": [0.7, 0.9]},
            ),
            ClusterGrid(
                algorithm="optics", parameters={"min_samples": 3, "min_cluster_size": 5}
            ),
            ClusterGrid(
                algorithm="hdbscan",
                parameters={"min_samples": 3, "min_cluster_size": 5},
            ),
            ClusterGrid(algorithm="agglomerative", parameters={"n_clusters": 3}),
        ),
        boundary=BoundaryConfig(
            windows=(1, 3, 5), baseline_radius=10, baseline_min_count=3
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(unknown=1),
        dict(boundary={"windows": [3, 1]}),
        dict(encoders=["clip", "clip"]),
        dict(recurrence={"thresholds": {"clip": [0.5]}}),
        dict(boundary_grid={"rank_threshold": []}),
        dict(max_runs=1),
        dict(reductions=[{"method": "tsne", "seeds": []}]),
    ],
)
def test_strict_config_and_grid_limits(change):
    with pytest.raises(ValueError):
        SuiteConfig(**change)


def test_continuous_sequence_has_no_boundary_candidate():
    source = synthetic_memory((0,) * 72)
    config = small_config().boundary
    result = boundary_tables(source, config, "stable")
    assert result["candidate_zones"].empty
    assert result["temporal_scores"].shape[0] == 71
    assert result["occurrences"].shape[0] == 72


def test_two_contiguous_scenes_and_short_transition_produce_zones():
    for pattern in ((0,) * 36 + (1,) * 36, (0,) * 35 + (2, 3) + (1,) * 35):
        source = synthetic_memory(pattern)
        result = boundary_tables(source, small_config().boundary, "synthetic")
        candidates = result["candidate_zones"]
        assert len(candidates)
        assert ((candidates.start <= 36) & (candidates.end >= 35)).any()
        assert candidates.status.eq("candidate").all()
        assert (candidates.end > candidates.start).all()
        assert not {"sequence_id", "split_id", "exact_boundary"} & set(candidates)
        scores = result["temporal_scores"]
        assert "clip_w3_local_mad" in scores and "clip_adjacent_cosine" in scores


def test_recurrence_keeps_A_B_A_cores_and_exact_dependency_separate():
    source = synthetic_memory(duplicate=True)
    membership, cores = structure_membership(source, zones(source))
    assert len(cores) == 3
    assert cores.status.eq("candidate").all() and cores.decision.eq("").all()
    assert len(membership) == 3 * len(source.records)
    result = recurrence_tables(source, cores, RecurrenceConfig(rank_fraction=1.0))
    recurrence = result["pairs"].loc[result["pairs"].exact_shared_contents.gt(0)]
    assert len(recurrence) == 1 and recurrence.refined_candidate.all()
    assert recurrence.exact_copy_dependency_observed.all()
    assert not result["pairs"].dependency_confirmed.any()
    assert result["diagnostic_components"].diagnostic_only.all()
    assert not result["diagnostic_components"].dependency_confirmed.any()
    assert result["core_content_membership"].core_id.nunique() == 3
    assert result["nearest_matches"].exact_content.any()
    assert len(result["exact_copy_edges"]) == 1
    assert not result["exact_copy_edges"].sequence_instances_merged.any()
    mask = content_targets(membership, source.contents, "sequence_core")
    assert not mask.loc[mask.content_id.eq("c0000"), "evaluation_mask"].iloc[0]


def test_encoder_disagreement_not_averaged_and_broad_screen_flagged():
    source = synthetic_memory(weak_encoder=True)
    _, cores = structure_membership(source, zones(source))
    config = RecurrenceConfig(rank_fraction=1.0)
    result = recurrence_tables(source, cores, config)
    assert result["pairs"].encoder_disagreement.any()
    assert result["candidate_diagnostics"].non_discriminative_warning.any()
    assert set(result["encoder_scores"].encoder) == {"clip", "dinov2"}
    assert not any("mean_cosine" in c for c in result["pairs"])


def test_broad_77_of_78_style_regression_is_data_driven():
    source = synthetic_memory((0,) * 78)
    cores = pd.DataFrame(
        [
            dict(
                element_id=f"core{i:02d}",
                timeline_id="synthetic",
                start=i * 6,
                end=i * 6 + 5,
                kind="sequence_core",
                decision="supported",
                notes="fixture",
            )
            for i in range(13)
        ]
    )
    result = recurrence_tables(source, cores, RecurrenceConfig())
    diagnostic = (
        result["candidate_diagnostics"].set_index("policy").loc["broad_candidate"]
    )
    assert diagnostic.possible_pairs == 78
    assert diagnostic.candidate_rate > 0.5 and diagnostic.non_discriminative_warning
    assert not result["pairs"].dependency_confirmed.any()


@pytest.mark.parametrize("noise", [False, True])
def test_all_noise_single_cluster_metrics_report_masks_coverage(noise):
    source = synthetic_memory()
    membership, _ = structure_membership(source, zones(source))
    labels = np.full(len(source.contents), -1 if noise else 0)
    result = evaluation_tables(source, assignments(source, labels), membership)
    core = result["metrics"].loc[result["metrics"].target.eq("sequence_core")]
    all_items = core.loc[core.noise_policy.eq("all_evaluated")].iloc[0]
    assert all_items.evaluated_n == 68
    assert all_items.evaluated_content_coverage == 68 / 72
    assert all_items.trivial_partition
    mask = result["evaluation_mask"].query("target == 'sequence_core'")
    assert mask.evaluation_mask.sum() == 68
    if noise:
        nonnoise = core.loc[core.noise_policy.eq("clustered_evaluated")].iloc[0]
        assert nonnoise.evaluated_n == 0 and pd.isna(nonnoise.ari)
        assert result["cluster_summary"].empty
    else:
        assert result["cluster_summary"].merging_sequence_count.max() == 3


def test_reviewed_instance_and_known_source_targets_are_posthoc():
    source = synthetic_memory()
    intervals = zones(source)
    for kind in ("sequence_instance", "known_source_video"):
        intervals = pd.concat(
            [
                intervals,
                pd.DataFrame(
                    [
                        dict(
                            element_id=f"{kind}-{i}",
                            timeline_id="synthetic",
                            start=a,
                            end=b,
                            kind=kind,
                            decision="supported",
                            notes="explicit synthetic reference",
                        )
                        for i, (a, b) in enumerate(((0, 22), (25, 46), (49, 71)))
                    ]
                ),
            ],
            ignore_index=True,
        )
    validate_intervals(intervals, source)
    membership, _ = structure_membership(source, intervals)
    result = evaluation_tables(
        source, assignments(source, np.repeat([0, 1, 2], 24)), membership
    )
    assert set(result["metrics"].target) == {
        "sequence_core",
        "sequence_instance",
        "known_source_video",
    }
    assert (result["metrics"].evaluated_n == 68).all()


def test_transition_persistence_return_events_and_partial_review_precision():
    source = synthetic_memory()
    labels = np.repeat([0, 1, 0], 24)
    labels[10] = 2
    result = transition_tables(
        source, assignments(source, labels), TransitionConfig(), zones(source)
    )
    assert result["transitions"].isolated_one_frame.any()
    assert not result["transitions"].scene_boundary_confirmed.any()
    assert result["cluster_returns"].immediate_A_B_A.any()
    metric = result["transition_metrics"].iloc[0]
    assert metric.reviewed_zone_recall == 1
    assert metric.candidate_precision is None and metric.false_transitions is None
    assert metric.outside_reviewed_zones == 1
    exhaustive = transition_tables(
        source, assignments(source, labels), TransitionConfig(), zones(source), True
    )
    assert exhaustive["transition_metrics"].iloc[0].false_transitions == 1


def test_zone_hit_uses_interval_and_tolerance_not_fabricated_exact_boundary():
    source = synthetic_memory()
    candidates = pd.DataFrame([dict(timeline_id="synthetic", start=22, end=22)])
    assert compare_zones(candidates, zones(source), tolerance=0)["hit_zone_count"] == 0
    assert compare_zones(candidates, zones(source), tolerance=1)["hit_zone_count"] == 1


def test_noise_interrupts_transitions_and_noisy_coclusters_not_evidence():
    source = synthetic_memory()
    membership, cores = structure_membership(source, zones(source))
    labels = np.full(72, -1)
    assignment = assignments(source, labels)
    result = transition_tables(source, assignment, TransitionConfig())
    assert result["transitions"].empty
    assert result["transition_metrics"].iloc[0].nonnoise_cut_coverage == 0
    direct = recurrence_tables(source, cores, RecurrenceConfig())["pairs"]
    evidence = cluster_recurrence(assignment, cores, source, direct, RecurrenceConfig())
    assert evidence["cluster_recurrence"].shared_clusters.eq(0).all()
    assert not evidence["cluster_recurrence"].distributed_candidate.any()


def test_row_order_and_historical_splits_cannot_enter_fitting_or_boundary():
    source = synthetic_memory(duplicate=True)
    config = small_config()
    before, _ = clustering_tables(source, config, scientific_input_id(source))
    boundary = boundary_tables(source, config.boundary, scientific_input_id(source))
    source.records = source.records.sample(frac=1, random_state=6).reset_index(
        drop=True
    )
    source.records["original_split"] = "val"
    after, _ = clustering_tables(source, config, scientific_input_id(source))
    pd.testing.assert_frame_equal(before["assignments"], after["assignments"])
    pd.testing.assert_frame_equal(
        boundary["candidate_zones"],
        boundary_tables(source, config.boundary, scientific_input_id(source))[
            "candidate_zones"
        ],
    )
    assert after["assignments"].groupby("run_id").size().eq(len(source.contents)).all()
    assert len(after["occurrences"]) > len(source.contents)
    x = source.embeddings["clip"]
    a, _ = fit_visual(
        x, source.contents, "agglomerative", {"n_clusters": 3, "linkage": "average"}
    )
    b, _ = fit_visual(
        x[::-1],
        source.contents[::-1],
        "agglomerative",
        {"n_clusters": 3, "linkage": "average"},
    )
    np.testing.assert_array_equal(a, b[::-1])


def test_stability_reports_common_coverage_and_parameter_axis():
    source = synthetic_memory()
    fitted, _ = clustering_tables(source, small_config(), "synthetic")
    result = stability_tables(fitted["runs"], fitted["assignments"])
    assert "parameter_perturbation" in set(result.comparison)
    assert {
        "common_clustered_coverage",
        "all_points_ari",
        "noise_membership_agreement",
    } <= set(result)


def test_artifact_immutable_deterministic_and_every_file_verified(tmp_path):
    source = synthetic_memory()
    config = small_config().boundary
    first = run_boundary(source, config, tmp_path / "out")
    second = run_boundary(source, config, tmp_path / "out")
    assert first == second
    assert inspect(first)["semantics"] == SEMANTICS
    for name in [*inspect(first)["output_checksums"], "metadata.json", "receipt.json"]:
        path = first / name
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        assert not verify(first, source)["quality_valid"], name
        path.write_bytes(original)
    assert verify(first, source)["quality_valid"]
    data = pd.read_parquet(first / "candidate_zones.parquet")
    data.loc[0, "score"] = -100
    data.to_parquet(first / "candidate_zones.parquet", index=False)
    meta = read_json(first / "metadata.json")
    meta["output_checksums"]["candidate_zones.parquet"] = file_sha256(
        first / "candidate_zones.parquet"
    )
    write_json(first / "metadata.json", meta)
    write_json(
        first / "receipt.json",
        {"metadata_sha256": file_sha256(first / "metadata.json")},
    )
    assert not verify(first, source)["quality_valid"]


def test_schema_import_source_binding_and_conflicts(tmp_path):
    source = synthetic_memory()
    producer = tmp_path / "external.csv"
    producer.write_text("external manual evidence", encoding="utf-8")
    envelope = {
        "schema_version": "sequence_evidence_import_v1",
        "artifact_kind": "video11_manual_transition_review_v2",
        **expected_binding(source),
        "producer_files": {producer.name: file_sha256(producer)},
        "reviewer": "synthetic reviewer",
        "reviewed_at": "2026-09-01T12:00:00Z",
        "ground_truth": False,
        "split_created": False,
        "automatic_confirmation": False,
        "intervals": zones(source).to_dict("records"),
    }
    path = tmp_path / "normalized.json"
    write_json(path, envelope)
    output = import_evidence(path, source, tmp_path / "out")
    assert verify(output, source)["quality_valid"]
    assert len(tables(output)["cores"]) == 3
    for field, bad in (
        ("dataset_id", "wrong"),
        ("ground_truth", True),
        ("manifest_sha256", "b" * 64),
        ("unknown", "value"),
    ):
        write_json(path, {**envelope, field: bad})
        with pytest.raises(ValueError):
            import_evidence(path, source, tmp_path / "out")
    duplicate = {
        **envelope,
        "intervals": [*envelope["intervals"], envelope["intervals"][0]],
    }
    with pytest.raises(ValueError):
        EvidenceEnvelope.model_validate(duplicate)
    producer.write_text("changed evidence", encoding="utf-8")
    write_json(path, envelope)
    with pytest.raises(ValueError, match="producer file changed"):
        import_evidence(path, source, tmp_path / "out")


@pytest.fixture
def disk_sources(tmp_path):
    paths = synthetic_sources(tmp_path / "source", sizes=(72,))
    for directory in paths[1:]:
        index = pd.read_parquet(directory / "content_index.parquet")
        sample_indices = (
            index.representative_frame_id.str.rsplit("-", n=1)
            .str[-1]
            .astype(int)
            .to_numpy()
        )
        vector_source = synthetic_memory()
        x = vector_source.embeddings["clip"][sample_indices]
        for name in ("embeddings_raw.npy", "embeddings_l2.npy"):
            np.save(directory / name, x)
    return paths


def test_end_to_end_source_bound_suite_and_cli(disk_sources, tmp_path):
    source = load_sources(*disk_sources)
    config = small_config()
    directory = suite(source, config, tmp_path / "out")
    assert verify(directory, source)["quality_valid"]
    assert tables(directory)["comparison"].run_id.nunique() == 10
    flags = [
        "--manifest",
        str(disk_sources[0]),
        "--clip-features",
        str(disk_sources[1]),
        "--dinov2-features",
        str(disk_sources[2]),
    ]
    runner = CliRunner()
    for name in (
        "boundary",
        "clustering",
        "recurrence",
        "cluster-transitions",
        "evaluate",
        "suite",
        "verify",
        "summary",
        "import-evidence",
        "review-package",
        "review-import",
    ):
        result = runner.invoke(app, ["sequences", "experiment", name, "--help"])
        assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["sequences", "experiment", "verify", str(directory), *flags]
    )
    assert result.exit_code == 0, result.output
    manifest = pd.read_parquet(disk_sources[0]).iloc[::-1]
    manifest.to_parquet(disk_sources[0], index=False)
    changed = load_sources(*disk_sources)
    assert scientific_input_id(source) == scientific_input_id(changed)
    assert not verify(directory, changed)["quality_valid"]


def test_real_reducer_adapters_seeds_are_evaluated_offline():
    source = synthetic_memory()
    config = small_config(
        (
            ReductionGrid(
                method="tsne",
                seeds=(0, 1),
                parameters={"perplexity": 5.0, "max_iter": 300},
            ),
            ReductionGrid(
                method="pacmap",
                seeds=(0, 1),
                parameters={
                    "n_neighbors": 3,
                    "MN_ratio": 1.0,
                    "FP_ratio": 1.0,
                    "num_iters": [2, 2, 2],
                },
            ),
        )
    )
    result, details = clustering_tables(source, config, "synthetic")
    assert result["failures"].empty, result["failures"].to_dict("records")
    assert set(result["runs"].representation) == {"original_l2", "tsne", "pacmap"}
    stability = stability_tables(result["runs"], result["assignments"])
    assert "reduction_seed" in set(stability.comparison)
    assert details["spaces"]
    assert len(result["coordinates"])


def test_unstable_seed_fixture_does_not_hide_behind_common_clustered_ari():
    contents = [f"c{i:02d}" for i in range(12)]
    labels = [
        [0, 0, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7],
    ]
    runs = pd.DataFrame(
        [
            dict(
                run_id=f"s{i}",
                encoder="clip",
                algorithm="dbscan",
                representation="tsne",
                space_id=f"space-{i}",
                seed=i,
                parameters_json='{"min_samples": 2}',
                reduction_json=json.dumps({"method": "tsne", "seed": i}),
            )
            for i in range(2)
        ]
    )
    assignments_table = pd.DataFrame(
        [
            dict(run_id=f"s{i}", content_id=c, cluster_id=lab)
            for i in range(2)
            for c, lab in zip(contents, labels[i], strict=True)
        ]
    )
    result = stability_tables(runs, assignments_table).iloc[0]
    assert result.comparison == "reduction_seed"
    assert (
        result.common_clustered_ari == 1 and result.common_clustered_coverage == 1 / 3
    )
    assert result.all_points_ari < 1 and result.noise_membership_agreement == 1 / 3


def test_ambiguous_exact_membership_and_conflicting_timeline_rejected():
    source = synthetic_memory()
    interval = zones(source)
    conflict = dict(
        element_id="bad-instance",
        timeline_id="synthetic",
        start=20,
        end=27,
        kind="sequence_instance",
        decision="supported",
        notes="invalid overlap",
    )
    with pytest.raises(ValueError, match="unresolved boundary zone"):
        validate_intervals(
            pd.concat([interval, pd.DataFrame([conflict])], ignore_index=True), source
        )
    source.records.loc[1, "position"] = 0
    with pytest.raises(ValueError, match="Conflicting contents"):
        source.timelines()


def test_gaps_undefined_mad_and_source_read_only_output(disk_sources, tmp_path):
    source = synthetic_memory((0,) * 72)
    source.embeddings = {
        e: np.tile(np.eye(8, dtype=np.float32)[0], (72, 1)) for e in ("clip", "dinov2")
    }
    source.records.loc[36:, "position"] += 10
    result = boundary_tables(source, small_config().boundary, "synthetic")
    scores = result["temporal_scores"]
    assert scores.segment_id.nunique() == 2 and not scores.position.eq(46).any()
    assert scores.clip_w1_robust_z.isna().all() and result["candidate_zones"].empty
    disk = load_sources(*disk_sources)
    with pytest.raises(ValueError, match="separate from immutable"):
        run_boundary(disk, small_config().boundary, disk_sources[1])


def test_labeled_source_split_metadata_and_row_shuffle_preserve_fits(
    disk_sources, tmp_path
):
    from test_linkage import labeled_inputs

    paths = labeled_inputs(tmp_path / "labeled", disk_sources[1:])
    manifest = pd.read_parquet(paths[0])
    manifest["possible_sequence"] = "synthetic-family"
    manifest["possible_frame_index"] = [1, 2, 3, 1]
    manifest.to_parquet(paths[0], index=False)
    before = load_sources(*paths, family="synthetic-family")
    a, _ = fit_visual(
        before.embeddings["clip"],
        before.contents,
        "agglomerative",
        {"n_clusters": 2, "linkage": "average"},
    )
    manifest["original_split"] = ["test", "train", "val", "train"]
    manifest.iloc[::-1].to_parquet(paths[0], index=False)
    after = load_sources(*paths, family="synthetic-family")
    b, _ = fit_visual(
        after.embeddings["clip"],
        after.contents,
        "agglomerative",
        {"n_clusters": 2, "linkage": "average"},
    )
    np.testing.assert_array_equal(a, b)
    assert scientific_input_id(before) == scientific_input_id(after)
    assert after.records.temporal_source.eq("filename_heuristic").all()
    assert (
        len(after.records) == 4
        and len(after.contents) == 3
        and len(after.timelines()) == 3
    )


def test_cli_suite_persists_config_and_failed_cells_exit_nonzero(
    disk_sources, tmp_path
):
    import yaml

    runner = CliRunner()
    flags = [
        "--manifest",
        str(disk_sources[0]),
        "--clip-features",
        str(disk_sources[1]),
        "--dinov2-features",
        str(disk_sources[2]),
        "--output",
        str(tmp_path / "out"),
    ]
    config = SuiteConfig(
        encoders=("clip",),
        reductions=(),
        clustering=(
            ClusterGrid(algorithm="agglomerative", parameters={"n_clusters": 3}),
        ),
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    result = runner.invoke(
        app, ["sequences", "experiment", "suite", *flags, "--profile", str(profile)]
    )
    assert result.exit_code == 0, result.output
    path = Path(result.stdout.strip())
    assert inspect(path)["identity"]["config"] == config.model_dump(mode="json")
    broken = config.model_dump(mode="json")
    broken["clustering"][0]["parameters"]["n_clusters"] = 999
    profile.write_text(yaml.safe_dump(broken), encoding="utf-8")
    result = runner.invoke(
        app,
        ["sequences", "experiment", "clustering", *flags, "--profile", str(profile)],
    )
    assert result.exit_code == 1
    path = Path(result.stdout.strip())
    assert len(tables(path)["failures"]) == 1
    assert not read_json(path / "summary.json")["all_requested_cells_succeeded"]
