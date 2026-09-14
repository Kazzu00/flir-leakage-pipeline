"""Screening followed by bounded seed/parameter comparisons, with immutable receipts."""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flir_pipeline.clustering.algorithms import effective_parameters, k_distances
from flir_pipeline.clustering.base import ClusteringConfig
from flir_pipeline.clustering.metrics import assignment_agreement, summarize_agreements
from flir_pipeline.clustering.selection import (
    SELECTION_POLICY,
    final_candidates,
    parameter_neighbors,
    shortlist_screening,
)
from flir_pipeline.clustering.storage import (
    ClusteringFamily,
    clustering_provenance,
    load_family,
    run_to_store,
    verify_run,
)
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)


def read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "shortlist_reason" in frame:
        frame["shortlist_reason"] = frame.shortlist_reason.fillna("")
    return frame


def load_families(specification: Path) -> dict[str, ClusteringFamily]:
    spec = yaml.safe_load(specification.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != {"manifest", "encoders"} or not isinstance(spec["encoders"], dict) or not spec["encoders"]:
        raise ValueError("Inputs require a manifest and explicitly named encoder sources")
    families = {}
    for encoder, paths in spec["encoders"].items():
        if set(paths) != {"feature_directory", "similarity_directory", "reduction_benchmark"}:
            raise ValueError("Each encoder requires exact feature, similarity and reduction benchmark directories")
        family = load_family(Path(paths["feature_directory"]), Path(paths["similarity_directory"]), Path(spec["manifest"]), Path(paths["reduction_benchmark"]))
        if family.source.feature["extractor"] != encoder:
            raise ValueError("Encoder name does not match source metadata")
        families[encoder] = family
    populations = {tuple(sorted(f.context.content_ids)) for f in families.values()}
    if len(populations) != 1:
        raise ValueError("Compared encoders must cover the same unique contents")
    return families


def run_row(directory: Path) -> dict:
    m = read_json(directory/"metadata.json")
    return {"encoder": m["extractor"], "representation": m["representation"], "algorithm": m["algorithm"],
            "reduction_seed": m["reduction_seed"], "reduction_space_id": m["reduction_space_id"],
            "clustering_space_id": m["clustering_space_id"], "configuration_id": m["configuration_id"],
            "hyperparameters": json.dumps(m["config"]["hyperparameters"], sort_keys=True),
            "effective_eps": m["effective_parameters"].get("eps"), "fit_seconds": m["fit_seconds"],
            "evaluation_seconds": m["evaluation_seconds"], **read_json(directory/"metrics.json")}


def _run_reference(directory: Path, root: Path) -> dict:
    meta = read_json(directory/"metadata.json")
    return {"path": directory.relative_to(root).as_posix(), "metadata_sha256": file_sha256(directory/"metadata.json"),
            "clustering_space_id": meta["clustering_space_id"], "encoder": meta["extractor"]}


def screening_to_store(families: dict[str, ClusteringFamily], configs: list[ClusteringConfig],
                       output_root: Path = Path("artifacts/clustering")) -> Path:
    if not configs or len({c.configuration_id for c in configs}) != len(configs):
        raise ValueError("Screening requires distinct explicit configurations")
    started = time.perf_counter()
    directories, aliases, curves, quantiles = [], [], [], []
    total = len(families)*3*len(configs)
    number = 0
    for encoder, family in sorted(families.items()):
        for representation in ("original_l2", "tsne", "pacmap"):
            space = family.spaces[(representation, None if representation == "original_l2" else 0)]
            seen = {}
            for ms in sorted({c.hyperparameters["min_samples"] for c in configs if c.algorithm == "dbscan"}):
                values = k_distances(space.distances, ms)
                curves.append(pd.DataFrame({"encoder": encoder, "representation": representation, "min_samples": ms,
                                            "sorted_row": np.arange(len(values)), "k_distance": np.sort(values)}))
                for q in (0., .25, .5, .8, .85, .9, .95, .97, .99, 1.):
                    quantiles.append({"encoder": encoder, "representation": representation, "min_samples": ms,
                                      "quantile": q, "k_distance": float(np.quantile(values, q))})
            for config in configs:
                number += 1
                try:
                    effective = effective_parameters(config, space.distances)
                except ValueError as error:
                    if config.algorithm != "dbscan" or "Nonpositive k-distance" not in str(error):
                        raise
                    aliases.append({"encoder": encoder, "representation": representation, "configuration_id": config.configuration_id,
                                    "status": "non_executable_nonpositive_epsilon", "equivalent_to": None})
                    continue
                key = stable_id({"algorithm": config.algorithm, "parameters": effective})
                if key in seen:
                    aliases.append({"encoder": encoder, "representation": representation, "configuration_id": config.configuration_id,
                                    "status": "identical_effective_parameters", "equivalent_to": seen[key]})
                    continue
                seen[key] = config.configuration_id
                directory = run_to_store(family, space, config, output_root)
                directories.append(directory)
                row = run_row(directory)
                print(f"SCREEN {number}/{total} {encoder} {representation} {config.algorithm} clusters={row['n_clusters_excluding_noise']} noise={row['noise_fraction']:.3f} fit={row['fit_seconds']:.2f}s", flush=True)
    run_refs = [_run_reference(p, output_root) for p in directories]
    sweep_id = stable_id({"runs": sorted(r["clustering_space_id"] for r in run_refs), "policy": SELECTION_POLICY})
    output = output_root/"screening"/sweep_id
    if output.exists():
        if not verify_collection(output)["quality_valid"]:
            raise ValueError("Invalid/incomplete screening preserved")
        return output
    table, shortlist, omitted = shortlist_screening(pd.DataFrame([run_row(p) for p in directories]))
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output/"screening.csv", index=False)
    shortlist.to_csv(output/"shortlist.csv", index=False)
    pd.DataFrame(aliases, columns=["encoder", "representation", "configuration_id", "status", "equivalent_to"]).to_csv(output/"redundant_or_nonexecuted.csv", index=False)
    pd.DataFrame(quantiles).to_csv(output/"k_distance_quantiles.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_parquet(output/"k_distance_curves.parquet", index=False)
    write_json(output/"summary.json", {"runs": len(table), "runs_by_algorithm": table.algorithm.value_counts().to_dict(),
                                      "degenerate_runs": int((table.n_clusters_excluding_noise <= 1).sum()), "shortlist_count": len(shortlist),
                                      "pareto_count": int(table.pareto_stage_a.sum()), "omitted_criteria": omitted,
                                      "elapsed_seconds": time.perf_counter()-started, "fit_seconds": float(table.fit_seconds.sum())})
    metadata = {**clustering_provenance(), "artifact_kind": "clustering_screening", "collection_id": sweep_id,
                "selection_policy": SELECTION_POLICY, "runs": run_refs,
                "source_signatures": {e: f.source.signatures for e, f in families.items()},
                "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/"metadata.json", metadata)
    if not verify_collection(output)["quality_valid"]:
        raise ValueError("Screening publication failed verification and was retained")
    return output


def comparison_to_store(screening: Path, families: dict[str, ClusteringFamily],
                        output_root: Path = Path("artifacts/clustering")) -> Path:
    if not verify_collection(screening)["quality_valid"]:
        raise ValueError("Screening failed verification")
    screen_meta = read_json(screening/"metadata.json")
    if screen_meta["source_signatures"] != {e: f.source.signatures for e, f in families.items()}:
        raise ValueError("Comparison source signatures differ from screening")
    screening_root = screening.parents[1]
    if screening_root.resolve() != output_root.resolve():
        raise ValueError("Use the same artifact root for screening and comparison")
    locations = {r["clustering_space_id"]: screening_root/r["path"] for r in screen_meta["runs"]}
    # Original features alone cannot bind seed stability: selected reduction
    # families must also match before any new seed is fitted or cache returned.
    for reference in screen_meta["runs"]:
        run_meta = read_json(locations[reference["clustering_space_id"]]/"metadata.json")
        space = families[reference["encoder"]].spaces[(run_meta["representation"], run_meta["reduction_seed"])]
        if run_meta["input_signatures"] != space.signatures or run_meta["reduction_space_id"] != space.reduction_space_id:
            raise ValueError("Comparison reduction sources differ from screening")
    table = read_table(screening/"screening.csv")
    shortlist = table.loc[table.shortlist_stage_a].copy()
    comparison_id = stable_id({"screening_metadata_sha256": file_sha256(screening/"metadata.json"), "policy": SELECTION_POLICY})
    output = output_root/"comparison"/comparison_id
    if output.exists():
        if not verify_collection(output)["quality_valid"]:
            raise ValueError("Invalid/incomplete comparison preserved")
        return output
    comparisons, evaluated, seed_directories = [], [], []
    for number, row in enumerate(shortlist.to_dict("records"), 1):
        family = families[row["encoder"]]
        directory = locations[row["clustering_space_id"]]
        config = ClusteringConfig(**read_json(directory/"metadata.json")["config"])
        local = table.loc[(table.encoder == row["encoder"]) & (table.representation == row["representation"]) & (table.algorithm == row["algorithm"])]
        pool = {r.clustering_space_id: ClusteringConfig(**read_json(locations[r.clustering_space_id]/"metadata.json")["config"]) for r in local.itertuples()}
        left = np.load(directory/"cluster_labels.npy", allow_pickle=False)
        parameter_pairs = []
        for neighbor_id, axis in parameter_neighbors(config, pool):
            agreement = assignment_agreement(left, np.load(locations[neighbor_id]/"cluster_labels.npy", allow_pickle=False))
            parameter_pairs.append(agreement)
            comparisons.append({"reference_id": row["clustering_space_id"], "left_id": row["clustering_space_id"], "right_id": neighbor_id,
                                "kind": "parameters", "axis": axis, **agreement})
        fields = {f"parameters_{k}": v for k, v in summarize_agreements(parameter_pairs).items()}
        if row["representation"] != "original_l2":
            three = [directory]
            for seed in (1, 2):
                added = run_to_store(family, family.spaces[(row["representation"], seed)], config, output_root)
                three.append(added)
                seed_directories.append(added)
            seed_pairs = []
            for a, b in combinations(three, 2):
                agreement = assignment_agreement(np.load(a/"cluster_labels.npy", allow_pickle=False), np.load(b/"cluster_labels.npy", allow_pickle=False))
                seed_pairs.append(agreement)
                comparisons.append({"reference_id": row["clustering_space_id"], "left_id": a.name, "right_id": b.name,
                                    "kind": "seeds", "axis": "reduction_seed", **agreement})
            fields.update({f"seeds_{k}": v for k, v in summarize_agreements(seed_pairs).items()})
            fields["seeds_all_nondegenerate"] = all(read_json(p/"metrics.json")["n_clusters_excluding_noise"] > 1 for p in three)
        else:
            fields.update(seeds_comparison_count=0, seeds_all_nondegenerate=None,
                          seeds_common_clustered_all_nontrivial=None)
        evaluated.append({**row, **fields})
        print(f"STABILITY {number}/{len(shortlist)} {row['encoder']} {row['representation']} {row['algorithm']}", flush=True)
    if not evaluated:
        raise ValueError("Screening has no nondegenerate shortlist; preserve it and review the scientific protocol")
    final, candidates, omissions = final_candidates(pd.DataFrame(evaluated))
    all_locations = {**locations, **{p.name: p for p in seed_directories}}
    output.mkdir(parents=True, exist_ok=False)
    final.to_csv(output/"evaluated_shortlist.csv", index=False)
    candidates.to_csv(output/"candidates.csv", index=False)
    final.loc[final.reference_for_review].to_csv(output/"references.csv", index=False)
    pd.DataFrame(comparisons).to_parquet(output/"assignment_comparisons.parquet", index=False)
    all_rows = pd.DataFrame([run_row(p) for p in all_locations.values()])
    all_rows.to_csv(output/"all_runs.csv", index=False)
    write_json(output/"summary.json", {"screening_runs": len(table), "additional_seed_runs": len(set(seed_directories)),
                                      "total_runs": len(all_rows), "runs_by_algorithm": all_rows.algorithm.value_counts().to_dict(),
                                      "degenerate_runs": int((all_rows.n_clusters_excluding_noise <= 1).sum()),
                                      "clusters_min": int(all_rows.n_clusters_excluding_noise.min()), "clusters_max": int(all_rows.n_clusters_excluding_noise.max()),
                                      "noise_fraction_min": float(all_rows.noise_fraction.min()), "noise_fraction_max": float(all_rows.noise_fraction.max()),
                                      "shortlist_count": len(final), "candidate_count": len(candidates), "review_reference_count": int(final.reference_for_review.sum()),
                                      "comparison_count": len(comparisons), "omitted_criteria": omissions})
    metadata = {**clustering_provenance(), "artifact_kind": "clustering_comparison", "collection_id": comparison_id,
                "selection_policy": SELECTION_POLICY, "screening_path": screening.relative_to(output_root).as_posix(),
                "screening_metadata_sha256": file_sha256(screening/"metadata.json"),
                "source_signatures": screen_meta["source_signatures"],
                "runs": [_run_reference(p, output_root) for p in all_locations.values()],
                "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/"metadata.json", metadata)
    if not verify_collection(output)["quality_valid"]:
        raise ValueError("Comparison publication failed verification and was retained")
    return output


def verify_collection(directory: Path, families: dict[str, ClusteringFamily] | None = None) -> dict:
    checks = {"metadata_exists": (directory/"metadata.json").is_file()}
    try:
        meta = read_json(directory/"metadata.json")
        root = directory.parents[1]
        checks["policy_valid"] = meta["selection_policy"] == SELECTION_POLICY
        checks["output_checksums_valid"] = bool(meta["output_sha256"]) and all(file_sha256(directory/name) == digest for name, digest in meta["output_sha256"].items())
        checks["run_ids_unique"] = bool(meta["runs"]) and len({r["clustering_space_id"] for r in meta["runs"]}) == len(meta["runs"])
        checks["all_runs_valid"] = all(file_sha256(root/r["path"]/"metadata.json") == r["metadata_sha256"]
                                        and verify_run(root/r["path"], families[r["encoder"]] if families is not None else None)["quality_valid"] for r in meta["runs"])
        if meta["artifact_kind"] == "clustering_screening":
            expected_id = stable_id({"runs": sorted(r["clustering_space_id"] for r in meta["runs"]), "policy": SELECTION_POLICY})
            recorded = read_table(directory/"screening.csv")
            rebuilt, shortlist, _ = shortlist_screening(pd.DataFrame([run_row(root/r["path"]) for r in meta["runs"]]))
            pd.testing.assert_frame_equal(rebuilt, recorded, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            pd.testing.assert_frame_equal(shortlist.reset_index(drop=True), read_table(directory/"shortlist.csv"), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
        elif meta["artifact_kind"] == "clustering_comparison":
            screening = root/meta["screening_path"]
            checks["screening_binding_valid"] = file_sha256(screening/"metadata.json") == meta["screening_metadata_sha256"]
            expected_id = stable_id({"screening_metadata_sha256": meta["screening_metadata_sha256"], "policy": SELECTION_POLICY})
            recorded = read_table(directory/"evaluated_shortlist.csv")
            rebuilt, candidates, _ = final_candidates(recorded)
            pd.testing.assert_frame_equal(rebuilt, recorded, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            pd.testing.assert_frame_equal(candidates.reset_index(drop=True), read_table(directory/"candidates.csv"), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            locations = {r["clustering_space_id"]: root/r["path"] for r in meta["runs"]}
            all_rows = pd.DataFrame([run_row(p) for p in locations.values()])
            pd.testing.assert_frame_equal(all_rows, read_table(directory/"all_runs.csv"), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            screen_table = read_table(screening/"screening.csv")
            screen_shortlist = screen_table.loc[screen_table.shortlist_stage_a].reset_index(drop=True)
            pd.testing.assert_frame_equal(recorded[screen_table.columns], screen_shortlist, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            pairs = pd.read_parquet(directory/"assignment_comparisons.parquet")
            for pair in pairs.to_dict("records"):
                expected = assignment_agreement(np.load(locations[pair["left_id"]]/"cluster_labels.npy", allow_pickle=False), np.load(locations[pair["right_id"]]/"cluster_labels.npy", allow_pickle=False))
                if any(not (pd.isna(pair[k]) if v is None else pair[k] == v) for k, v in expected.items()):
                    raise ValueError("Saved ARI/AMI does not match assignments")
            expected_pairs = set()
            for row in recorded.to_dict("records"):
                identity = row["clustering_space_id"]
                local = screen_table.loc[(screen_table.encoder == row["encoder"]) & (screen_table.representation == row["representation"]) & (screen_table.algorithm == row["algorithm"])]
                pool = {r.clustering_space_id: ClusteringConfig(**read_json(locations[r.clustering_space_id]/"metadata.json")["config"]) for r in local.itertuples()}
                for neighbor, axis in parameter_neighbors(pool[identity], pool):
                    expected_pairs.add((identity, identity, neighbor, "parameters", axis))
                reduced = row["representation"] != "original_l2"
                if reduced:
                    seeds = all_rows.loc[(all_rows.encoder == row["encoder"]) & (all_rows.representation == row["representation"]) & (all_rows.configuration_id == row["configuration_id"])].sort_values("reduction_seed")
                    if seeds.reduction_seed.tolist() != [0, 1, 2] or seeds.iloc[0].clustering_space_id != identity:
                        raise ValueError("Incomplete or mismatched seed family")
                    for a, b in combinations(seeds.clustering_space_id, 2):
                        expected_pairs.add((identity, a, b, "seeds", "reduction_seed"))
                    if row["seeds_all_nondegenerate"] != bool((seeds.n_clusters_excluding_noise > 1).all()):
                        raise ValueError("Saved seed degeneracy differs from runs")
                elif row["seeds_comparison_count"] != 0:
                    raise ValueError("Original controls have no reduction seed comparisons")
                for kind in (("parameters", "seeds") if reduced else ("parameters",)):
                    group = pairs.loc[(pairs.reference_id == identity) & (pairs.kind == kind)]
                    for key, value in summarize_agreements(group.to_dict("records")).items():
                        saved = row[f"{kind}_{key}"]
                        equal = pd.isna(saved) if value is None else np.isclose(saved, value, atol=1e-12, rtol=1e-12)
                        if not equal:
                            raise ValueError("Saved stability aggregate differs from pair agreements")
            pair_columns = ["reference_id", "left_id", "right_id", "kind", "axis"]
            recorded_pairs = list(pairs[pair_columns].itertuples(index=False, name=None))
            if len(recorded_pairs) != len(expected_pairs) or set(recorded_pairs) != expected_pairs:
                raise ValueError("Comparison pairs do not follow the prescribed perturbations")
            pd.testing.assert_frame_equal(rebuilt.loc[rebuilt.reference_for_review].reset_index(drop=True), read_table(directory/"references.csv"), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            checks["agreements_recomputed"] = True
            checks["stability_aggregates_and_pair_protocol_recomputed"] = True
        else:
            raise ValueError("Unknown clustering collection kind")
        checks["identity_valid"] = meta["collection_id"] == expected_id
        checks["selection_recomputed"] = True
        if families is not None:
            checks["source_matches"] = meta["source_signatures"] == {e: f.source.signatures for e, f in families.items()}
        checks["quality_valid"] = all(checks.values())
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, AssertionError) as error:
        checks.update(quality_valid=False, error_type=type(error).__name__)
    return checks
