"""Descriptive comparisons of frozen variants; pairing never asserts byte identity."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.clustering.metrics import assignment_agreement
from flir_pipeline.data.local_images import relative_posix_path
from flir_pipeline.data.variants import validate_variant, variant_fields
from flir_pipeline.sequences.experiments.artifacts import (
    binding,
    inspect,
    publish,
    tables,
)
from flir_pipeline.sequences.experiments.config import SuiteConfig
from flir_pipeline.sequences.experiments.evaluation import validate_assignments
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id


@dataclass
class FrozenSuite:
    path: Path
    source: dict
    config: dict
    data: dict
    children: dict
    metadata: dict

    @property
    def occurrences(self):
        return self.children["clustering"]["occurrences"]


def read_suite(path):
    """Verify all stored bytes and child bindings, without refitting or raw images."""
    path = Path(path).resolve()
    meta, data = inspect(path), tables(path)
    if meta["artifact_kind"] != "sequence_experiment_suite_v1":
        raise ValueError("Supply a completed sequence_experiment_suite_v1 publication")
    summary = read_json(path / "summary.json")
    if (
        summary.get("all_requested_cells_succeeded") is not True
        or not data["failures"].empty
    ):
        raise ValueError("Cannot compare an incomplete experiment suite")
    source = meta["identity"]["sources"]["input"]
    variant = validate_variant(
        source["dataset_variant"],
        source["dataset_id"],
        source["checksums"]["manifest_sha256"],
    )
    if any(source.get(k) != v for k, v in variant_fields(variant).items()):
        raise ValueError("Suite variant fields conflict")
    config = SuiteConfig.model_validate(meta["identity"]["config"])
    expected = {f"boundary_{i}" for i in range(len(config.boundaries()))} | {
        "clustering",
        "transitions",
    }
    if "structure" in meta["identity"]["sources"]:
        expected |= {"evaluation", "recurrence"}
    children, frozen = {}, meta["identity"]["sources"]["children"]
    if (
        not data["artifacts"].role.is_unique
        or set(data["artifacts"].role) != expected
        or set(frozen) != expected
    ):
        raise ValueError("Incomplete or duplicate suite child coverage")
    for row in data["artifacts"].itertuples():
        relative_posix_path(row.relative_directory)
        child = (path.parent.parent / row.relative_directory).resolve()
        if not child.is_relative_to(path.parent.parent):
            raise ValueError("Suite child locator escapes its root")
        bound = binding(child)
        if bound != frozen[row.role] or bound != {
            "artifact_id": row.artifact_id,
            "metadata_sha256": row.metadata_sha256,
        }:
            raise ValueError("Suite child binding changed")
        child_meta = inspect(child)
        if child_meta["identity"]["sources"].get("input") != source:
            raise ValueError("Suite mixes dataset variants or feature sources")
        if (
            "clustering" in child_meta["identity"]["sources"]
            and child_meta["identity"]["sources"]["clustering"] != frozen["clustering"]
        ):
            raise ValueError("Suite child references different clustering")
        children[row.role] = tables(child)
    fitted = children["clustering"]
    occurrences = fitted["occurrences"]
    if not occurrences.frame_id.is_unique or occurrences.empty:
        raise ValueError("Invalid suite occurrence identity")
    validate_assignments(fitted["assignments"], sorted(occurrences.content_id.unique()))
    expected_runs = (
        len(config.encoders)
        * (int(config.original_l2) + sum(len(g.expand()) for g in config.reductions))
        * sum(len(g.expand()) for g in config.clustering)
    )
    if (
        len(fitted["runs"]) != expected_runs
        or not fitted["runs"].run_id.is_unique
        or set(fitted["assignments"].run_id) != set(fitted["runs"].run_id)
    ):
        raise ValueError("Suite does not cover every configured clustering cell")
    return FrozenSuite(path, source, meta["identity"]["config"], data, children, meta)


PAIR_COLUMNS = [
    "left_frame_id",
    "right_frame_id",
    "mapping_method",
    "confidence",
    "evidence",
    "ground_truth",
]


def normalize_pairs(frame, left, right):
    if list(frame) != PAIR_COLUMNS or frame.empty:
        raise ValueError(
            "Expected nonempty correspondence CSV with columns: "
            + ",".join(PAIR_COLUMNS)
        )
    frame = frame.copy()
    for side, suite in (("left", left), ("right", right)):
        key = f"{side}_frame_id"
        if not frame[key].is_unique or not set(frame[key]) <= set(
            suite.occurrences.frame_id
        ):
            raise ValueError(
                "Correspondence requires known, one-to-one occurrence IDs on both sides"
            )
        index = suite.occurrences.set_index("frame_id")
        for name in ("content_id", "timeline_id", "position", "temporal_source"):
            frame[f"{side}_{name}"] = frame[key].map(index[name])
    for name in ("mapping_method", "evidence"):
        if not frame[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("Correspondence requires mapping method and evidence")
    confidence = pd.to_numeric(frame.confidence, errors="raise")
    if not np.isfinite(confidence).all() or not confidence.between(0, 1).all():
        raise ValueError("Correspondence confidence must be finite in [0,1]")
    frame["confidence"] = confidence.astype(float)

    def boolean(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError("Correspondence ground_truth must be an explicit boolean")

    frame["ground_truth"] = frame.ground_truth.map(boolean).astype(bool)
    frame["byte_identity_asserted"] = False
    return frame.sort_values(["left_frame_id", "right_frame_id"]).reset_index(drop=True)


def coverage(pairs, left, right):
    return {
        f"{side}_{key}": value
        for side, suite in (("left", left), ("right", right))
        for key, value in {
            "total_occurrences": len(suite.occurrences),
            "paired_occurrences": len(pairs),
            "unpaired_occurrences": len(suite.occurrences) - len(pairs),
            "occurrence_coverage": len(pairs) / len(suite.occurrences),
            "total_contents": suite.occurrences.content_id.nunique(),
            "paired_contents": pairs[f"{side}_content_id"].nunique()
            if len(pairs)
            else 0,
        }.items()
    }


def validate_sides(left, right):
    if left.source["dataset_variant_id"] == right.source["dataset_variant_id"]:
        raise ValueError(
            "Variant comparison requires two distinct dataset_variant_id values"
        )


def safe_output(output, *sources):
    if any(Path(output).resolve().is_relative_to(Path(p).resolve()) for p in sources):
        raise ValueError("Never publish inside an immutable input")


def import_correspondence(left_path, right_path, csv, output):
    left, right = read_suite(left_path), read_suite(right_path)
    validate_sides(left, right)
    safe_output(output, left.path, right.path)
    pairs = normalize_pairs(
        pd.read_csv(csv, dtype=str, keep_default_na=False), left, right
    )
    return publish(
        Path(output),
        "sequence_variant_correspondence_v1",
        {"pairing_policy": "explicit_one_to_one_occurrences_v1"},
        {
            "variants": {"left": left.source, "right": right.source},
            "suites": {"left": binding(left.path), "right": binding(right.path)},
            "import_sha256": file_sha256(csv),
        },
        {
            "pairs": pairs,
            "left_occurrences": left.occurrences,
            "right_occurrences": right.occurrences,
        },
        {
            **coverage(pairs, left, right),
            "byte_identity_asserted": False,
            "external_ground_truth_rows": int(pairs.ground_truth.sum()),
            "confidence_is_external_evidence": True,
        },
        media={"media/source_correspondence.csv": Path(csv).read_bytes()},
    )


def load_correspondence(path, left, right):
    meta, data = inspect(path), tables(path)
    if meta["artifact_kind"] != "sequence_variant_correspondence_v1" or meta[
        "identity"
    ]["sources"]["variants"] != {"left": left.source, "right": right.source}:
        raise ValueError(
            "Correspondence belongs to different variants/sources or orientation"
        )
    if meta["identity"]["sources"]["suites"] != {
        "left": binding(left.path),
        "right": binding(right.path),
    }:
        raise ValueError("Correspondence belongs to different suite publications")
    frozen_csv = Path(path) / "media/source_correspondence.csv"
    if file_sha256(frozen_csv) != meta["identity"]["sources"]["import_sha256"]:
        raise ValueError("Frozen correspondence input changed")
    expected = normalize_pairs(
        pd.read_csv(frozen_csv, dtype=str, keep_default_na=False), left, right
    )
    pd.testing.assert_frame_equal(expected, data["pairs"])
    for side, suite in (("left", left), ("right", right)):
        pd.testing.assert_frame_equal(data[f"{side}_occurrences"], suite.occurrences)
    return expected


def content_pairing(pairs):
    """Exact copies cannot multiply ARI weight or force a many-to-one label match."""
    columns = ["left_content_id", "right_content_id"]
    if pairs.empty:
        return pd.DataFrame(columns=[*columns, "eligible", "exclusion_reason"])
    edges = pairs[columns].drop_duplicates().sort_values(columns).reset_index(drop=True)
    left_degree = edges.groupby(columns[0])[columns[1]].transform("nunique")
    right_degree = edges.groupby(columns[1])[columns[0]].transform("nunique")
    edges["eligible"] = left_degree.eq(1) & right_degree.eq(1)
    edges["exclusion_reason"] = np.where(
        edges.eligible, "", "non_bijective_content_relation"
    )
    return edges


def run_keys(suite):
    # Label agreement is meaningful on paired observations even when a variant
    # changes preprocessing/feature space. Match the requested fitting protocol;
    # report feature-space equality independently, never assume it.
    result = suite.children["clustering"]["runs"].copy()
    result["configuration_key"] = [
        stable_id(
            {
                "encoder": row.encoder,
                "representation": row.representation,
                "seed": row.seed,
                "algorithm": row.algorithm,
                "parameters": json.loads(row.parameters_json),
                "reduction": json.loads(row.reduction_json),
            }
        )
        for row in result.itertuples()
    ]
    if not result.configuration_key.is_unique:
        raise ValueError("Duplicate comparable run configurations")
    result["feature_space_signature"] = [
        stable_id(suite.source["feature_spaces"][encoder]) for encoder in result.encoder
    ]
    return result


def tagged(frame, side, suite):
    result = frame.copy()
    result.insert(0, "variant_side", side)
    result.insert(1, "dataset_variant_id", suite.source["dataset_variant_id"])
    result.insert(2, "variant_name", suite.source["variant_name"])
    return result


def descriptive_tables(left, right):
    distributions, boundaries, shortlists, cluster_rows, encoder_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    detailed, adjacent = {}, []
    for side, suite in (("left", left), ("right", right)):

        def add(name, frame, side=side, suite=suite):
            detailed.setdefault(name, []).append(tagged(frame, side, suite))

        add("assignments", suite.children["clustering"]["assignments"])
        add("runs", run_keys(suite))
        add("stability", suite.data["stability"])
        add("boundary_stability", suite.data["boundary_stability"])
        for run, group in suite.children["clustering"]["assignments"].groupby("run_id"):
            cluster_rows.append(
                dict(
                    variant_side=side,
                    run_id=run,
                    contents=len(group),
                    cluster_count=group.loc[
                        group.cluster_id.ge(0), "cluster_id"
                    ].nunique(),
                    noise_count=int(group.cluster_id.eq(-1).sum()),
                    noise_rate=float(group.cluster_id.eq(-1).mean()),
                )
            )
        if "evaluation" in suite.children:
            for name in ("metrics", "diagnostics", "evaluation_mask"):
                add("evaluation_" + name, suite.children["evaluation"][name])
        for i, config in enumerate(
            SuiteConfig.model_validate(suite.config).boundaries()
        ):
            data = suite.children[f"boundary_{i}"]
            key = stable_id(config.model_dump(mode="json"))
            zones, scores = data["candidate_zones"], data["temporal_scores"]
            add("boundary_zones", zones.assign(boundary_configuration=key))
            for signal in ("adjacent", "multiscale", "current_v1"):
                policies = (
                    ("consensus",)
                    if signal == "current_v1"
                    else ("clip", "dinov2", "consensus")
                )
                for policy in policies:
                    selected = zones.loc[
                        zones.signal.eq(signal) & zones.encoder_policy.eq(policy)
                    ]
                    flags = zone_flags(scores, selected)
                    boundaries.append(
                        dict(
                            variant_side=side,
                            boundary_configuration=key,
                            signal=signal,
                            encoder_policy=policy,
                            zone_count=len(selected),
                            scored_positions=len(scores),
                            candidate_positions=int(flags.sum()),
                            candidate_position_rate=float(flags.mean())
                            if len(flags)
                            else None,
                        )
                    )
            for signal in ("adjacent", "multiscale"):
                column = f"{signal}_encoder_disagreement"
                encoder_rows.append(
                    dict(
                        variant_side=side,
                        component="boundary",
                        configuration=key,
                        signal=signal,
                        evaluated=len(scores),
                        disagreement_count=int(scores[column].sum())
                        if len(scores)
                        else 0,
                        disagreement_rate=float(scores[column].mean())
                        if len(scores)
                        else None,
                    )
                )
            # Adjacent similarities are independent of boundary thresholds/windows;
            # record each source adjacency once, not once per grid configuration.
            if i == 0:
                for encoder in ("clip", "dinov2"):
                    values = (
                        scores[f"{encoder}_adjacent_cosine"].to_numpy()
                        if len(scores)
                        else np.array([])
                    )
                    finite = values[np.isfinite(values)]
                    distributions.append(
                        dict(
                            variant_side=side,
                            encoder=encoder,
                            count=len(finite),
                            undefined_count=len(values) - len(finite),
                            mean=float(finite.mean()) if len(finite) else None,
                            **{
                                name: float(np.quantile(finite, q))
                                if len(finite)
                                else None
                                for name, q in (
                                    ("min", 0),
                                    ("q05", 0.05),
                                    ("median", 0.5),
                                    ("q95", 0.95),
                                    ("max", 1),
                                )
                            },
                        )
                    )
                    if len(scores):
                        adjacent.append(
                            tagged(
                                scores[
                                    ["timeline_id", "previous_position", "position"]
                                ].assign(encoder=encoder, cosine=values),
                                side,
                                suite,
                            )
                        )
        if "recurrence" in suite.children:
            recurrence = suite.children["recurrence"]
            add("recurrence_cores", recurrence["structure_elements"])
            add("recurrence_pairs", recurrence["pairs"])
            add("recurrence_scores", recurrence["encoder_scores"])
            for row in recurrence["candidate_diagnostics"].to_dict("records"):
                shortlists.append({"variant_side": side, **row})
            pairs = recurrence["pairs"]
            encoder_rows.append(
                dict(
                    variant_side=side,
                    component="recurrence",
                    configuration="",
                    signal="encoder_candidate",
                    evaluated=len(pairs),
                    disagreement_count=int(pairs.encoder_disagreement.sum()),
                    disagreement_rate=float(pairs.encoder_disagreement.mean())
                    if len(pairs)
                    else None,
                )
            )
    result = {
        key: pd.concat([v for v in values if len(v)] or values, ignore_index=True)
        for key, values in detailed.items()
    }
    result.update(
        adjacent_distributions=pd.DataFrame(distributions),
        boundary_candidates=pd.DataFrame(boundaries),
        clustering_coverage=pd.DataFrame(cluster_rows),
        candidate_shortlists=pd.DataFrame(
            shortlists,
            columns=[
                "variant_side",
                "policy",
                "possible_pairs",
                "selected_pairs",
                "candidate_rate",
                "non_discriminative_warning",
            ],
        ),
        encoder_agreement=pd.DataFrame(encoder_rows),
        adjacent_pairs=pd.concat(adjacent, ignore_index=True)
        if adjacent
        else pd.DataFrame(
            columns=[
                "variant_side",
                "dataset_variant_id",
                "variant_name",
                "timeline_id",
                "previous_position",
                "position",
                "encoder",
                "cosine",
            ]
        ),
    )
    return result


def zone_flags(points, zones):
    flags = np.zeros(len(points), dtype=bool)
    for zone in zones.itertuples():
        flags |= (
            (
                points.timeline_id.eq(zone.timeline_id)
                & points.position.between(zone.start, zone.end)
            )
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    return flags


def compare_assignments(left, right, pairs):
    keys_a, keys_b = run_keys(left), run_keys(right)
    columns = ["configuration_key", "run_id", "feature_space_signature"]
    matched = keys_a[columns].merge(
        keys_b[columns],
        on="configuration_key",
        suffixes=("_left", "_right"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    edges = content_pairing(pairs)
    eligible = edges.loc[edges.eligible.astype(bool)]
    rows = []
    for row in matched.loc[matched["_merge"].eq("both")].itertuples():
        a = (
            left.children["clustering"]["assignments"]
            .query("run_id == @row.run_id_left")
            .set_index("content_id")
        )
        b = (
            right.children["clustering"]["assignments"]
            .query("run_id == @row.run_id_right")
            .set_index("content_id")
        )
        n = len(eligible)
        scores = (
            assignment_agreement(
                a.loc[eligible.left_content_id, "cluster_id"].to_numpy(dtype=int),
                b.loc[eligible.right_content_id, "cluster_id"].to_numpy(dtype=int),
            )
            if n
            else {
                f"{policy}_{metric}": 0 if metric in {"n", "coverage"} else None
                for policy in ("all_points", "common_clustered")
                for metric in ("ari", "ami", "n", "coverage", "trivial")
            }
        )
        rows.append(
            dict(
                configuration_key=row.configuration_key,
                left_run_id=row.run_id_left,
                right_run_id=row.run_id_right,
                feature_space_equal=row.feature_space_signature_left
                == row.feature_space_signature_right,
                paired_unique_contents=n,
                left_content_coverage=n / len(a),
                right_content_coverage=n / len(b),
                excluded_non_bijective_edges=int((~edges.eligible.astype(bool)).sum()),
                pairing_available=not pairs.empty,
                **scores,
            )
        )
    matched["_merge"] = matched["_merge"].astype(str)
    return {
        "run_matching": matched.rename(columns={"_merge": "configuration_match"}),
        "content_correspondence": edges,
        "assignment_agreement": pd.DataFrame(rows),
    }


def compare_zones(left, right, pairs):
    rows = []
    a_configs = {
        stable_id(c.model_dump(mode="json")): i
        for i, c in enumerate(SuiteConfig.model_validate(left.config).boundaries())
    }
    b_configs = {
        stable_id(c.model_dump(mode="json")): i
        for i, c in enumerate(SuiteConfig.model_validate(right.config).boundaries())
    }
    known = (
        pairs.dropna(subset=["left_position", "right_position"])
        if not pairs.empty
        else pairs
    )
    for key in sorted(set(a_configs) & set(b_configs)):
        a = left.children[f"boundary_{a_configs[key]}"]["candidate_zones"]
        b = right.children[f"boundary_{b_configs[key]}"]["candidate_zones"]
        policies = [
            (s, p)
            for s in ("adjacent", "multiscale")
            for p in ("clip", "dinov2", "consensus")
        ]
        for signal, policy in [*policies, ("current_v1", "consensus")]:
            selected_a = a.loc[a.signal.eq(signal) & a.encoder_policy.eq(policy)]
            selected_b = b.loc[b.signal.eq(signal) & b.encoder_policy.eq(policy)]
            flags = []
            for side, selected in (("left", selected_a), ("right", selected_b)):
                points = (
                    known[[f"{side}_timeline_id", f"{side}_position"]].rename(
                        columns={
                            f"{side}_timeline_id": "timeline_id",
                            f"{side}_position": "position",
                        }
                    )
                    if len(known)
                    else pd.DataFrame(columns=["timeline_id", "position"])
                )
                flags.append(zone_flags(points, selected))
            union = int((flags[0] | flags[1]).sum())
            rows.append(
                dict(
                    boundary_configuration=key,
                    signal=signal,
                    encoder_policy=policy,
                    paired_known_occurrences=len(known),
                    left_occurrence_coverage=len(known) / len(left.occurrences),
                    right_occurrence_coverage=len(known) / len(right.occurrences),
                    mapped_zone_membership_jaccard=int((flags[0] & flags[1]).sum())
                    / union
                    if union
                    else None,
                    left_zone_count=len(selected_a),
                    right_zone_count=len(selected_b),
                    exact_boundaries_created=False,
                )
            )
    return pd.DataFrame(rows)


def compare_recurrence(left, right, pairs):
    core_pairs, comparison = [], []
    if (
        not pairs.empty
        and "recurrence" in left.children
        and "recurrence" in right.children
    ):
        mapping = pairs.set_index("left_frame_id").right_frame_id.to_dict()

        def core_frames(suite):
            return {
                c.element_id: set(
                    suite.occurrences.loc[
                        suite.occurrences.timeline_id.eq(c.timeline_id)
                        & suite.occurrences.position.between(c.start, c.end),
                        "frame_id",
                    ]
                )
                for c in suite.children["recurrence"]["structure_elements"].itertuples()
            }

        first, second = core_frames(left), core_frames(right)
        for key, frames in first.items():
            candidates = [
                b
                for b, values in second.items()
                if frames
                and frames <= set(mapping)
                and {mapping[f] for f in frames} == values
            ]
            if len(candidates) == 1:
                core_pairs.append(
                    dict(
                        left_core=key,
                        right_core=candidates[0],
                        paired_occurrences=len(frames),
                        rule="complete_equal_mapped_occurrence_sets",
                    )
                )
        mapped = {r["left_core"]: r["right_core"] for r in core_pairs}
        if len(set(mapped.values())) != len(mapped):
            raise ValueError("Ambiguous core correspondence; never select a best match")
        right_pairs = {
            tuple(sorted((r.left, r.right))): r
            for r in right.children["recurrence"]["pairs"].itertuples()
        }
        for a in left.children["recurrence"]["pairs"].itertuples():
            if a.left not in mapped or a.right not in mapped:
                continue
            b = right_pairs.get(tuple(sorted((mapped[a.left], mapped[a.right]))))
            if b is not None:
                comparison.append(
                    dict(
                        left_pair_id=a.pair_id,
                        right_pair_id=b.pair_id,
                        left_refined_candidate=bool(a.refined_candidate),
                        right_refined_candidate=bool(b.refined_candidate),
                        left_encoder_disagreement=bool(a.encoder_disagreement),
                        right_encoder_disagreement=bool(b.encoder_disagreement),
                    )
                )
    return {
        "core_correspondence": pd.DataFrame(
            core_pairs,
            columns=["left_core", "right_core", "paired_occurrences", "rule"],
        ),
        "recurrence_pair_comparison": pd.DataFrame(
            comparison,
            columns=[
                "left_pair_id",
                "right_pair_id",
                "left_refined_candidate",
                "right_refined_candidate",
                "left_encoder_disagreement",
                "right_encoder_disagreement",
            ],
        ),
    }


def compare_variants(left_path, right_path, output, correspondence=None):
    left, right = read_suite(left_path), read_suite(right_path)
    validate_sides(left, right)
    safe_output(
        output, left.path, right.path, *([correspondence] if correspondence else [])
    )
    pairs = (
        load_correspondence(correspondence, left, right)
        if correspondence
        else pd.DataFrame(
            columns=[
                "left_frame_id",
                "right_frame_id",
                "left_content_id",
                "right_content_id",
            ]
        )
    )
    result = descriptive_tables(left, right)
    result.update(compare_assignments(left, right, pairs))
    result["boundary_zone_pairing"] = compare_zones(left, right, pairs)
    result.update(compare_recurrence(left, right, pairs))
    result["paired_occurrences"] = pairs
    result["left_occurrences"] = left.occurrences
    result["right_occurrences"] = right.occurrences
    sources = {
        "variants": {"left": left.source, "right": right.source},
        "suites": {"left": binding(left.path), "right": binding(right.path)},
    }
    if correspondence:
        sources["correspondence"] = binding(correspondence)
    summary = {
        **coverage(pairs, left, right),
        "correspondence_available": correspondence is not None,
        "protocol_equal": left.config == right.config,
        "feature_spaces_equal": left.source["feature_spaces"]
        == right.source["feature_spaces"],
        "software_equal": left.metadata["identity"]["software"]
        == right.metadata["identity"]["software"],
        "matched_run_configurations": int(
            result["run_matching"].configuration_match.eq("both").sum()
        ),
        "paired_cores": len(result["core_correspondence"]),
        "compared_recurrence_pairs": len(result["recurrence_pair_comparison"]),
        "left_recurrence_pairs": len(
            left.children.get("recurrence", {}).get("pairs", [])
        ),
        "right_recurrence_pairs": len(
            right.children.get("recurrence", {}).get("pairs", [])
        ),
        "evaluation_population_policy": "Per-variant masks and denominators; no automatic causal or superiority claim",
        "boundary_comparison_policy": "Mapped occurrence membership in uncertainty zones; never exact boundaries",
        "assignment_policy": "Unique bijective content pairs only; duplicates do not multiply ARI/AMI weight",
        "automatic_winner_selected": False,
        "clustering_shortlist_status": "not produced by source suites",
        "recurrence_protocol_equal": left.config["recurrence"]
        == right.config["recurrence"],
        "component_availability": {
            side: {
                "temporal_recall_visual_coherence": "available"
                if "evaluation" in suite.children
                else "unavailable: no reviewed structure in source suite",
                "recurrence_and_shortlist": "available"
                if "recurrence" in suite.children
                else "unavailable: no reviewed structure in source suite",
                "stability_comparisons": len(suite.data["stability"]),
            }
            for side, suite in (("left", left), ("right", right))
        },
        "recurrence_pair_coverage": {
            side: len(result["recurrence_pair_comparison"])
            / len(suite.children["recurrence"]["pairs"])
            if "recurrence" in suite.children
            and len(suite.children["recurrence"]["pairs"])
            else None
            for side, suite in (("left", left), ("right", right))
        },
        "core_coverage": {
            side: {
                "total": len(
                    suite.children.get("recurrence", {}).get("structure_elements", [])
                ),
                "paired": len(result["core_correspondence"]),
                "unpaired": len(
                    suite.children.get("recurrence", {}).get("structure_elements", [])
                )
                - len(result["core_correspondence"]),
            }
            for side, suite in (("left", left), ("right", right))
        },
        "raw_sources_reverified": False,
        "verification_scope": "Frozen suite/child/pairing bytes and source identities",
    }
    return publish(
        Path(output),
        "sequence_variant_comparison_v1",
        {"comparison_protocol": "paired_variant_evidence_v1"},
        sources,
        result,
        summary,
    )
