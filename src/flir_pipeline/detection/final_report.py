"""Final descriptive reporting over immutable, fully verified detector evidence.

This extends the existing evidence gate and hierarchical aggregation. It never
trains, bootstraps anew, changes split membership, or promotes a partial matrix.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.detection.association import (
    COMPLETE,
    DEFAULT_CONFIG,
    RESIDUALS,
    STRATEGY_LABELS,
    AssociationEvidence,
    load_evidence,
    load_registry,
    prespecified_table,
)
from flir_pipeline.detection.export_contract import (
    SCHEMA_VERSION,
    DetectionExport,
    Interval,
    export_schema,
)
from flir_pipeline.detection.metrics import aggregate_runs
from flir_pipeline.detection.protocol import METRICS, verify_plan
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
)

SPLIT_KEYS = ["strategy", "split_seed", "split_space_id"]
SEMANTICS = {
    "causal": False,
    "unit_of_analysis": "split_mean_across_detector_seeds",
    "comparison": "Comparación descriptiva controlada: dependencia residual, composición, dificultad visual y membresía de train/validation/test pueden variar conjuntamente. No estima inflación causal por leakage.",
    "temporal_at5": "Proxy de distancia de índices de frame/nombre Δ≤5; no son segundos ni ground truth.",
    "clustering": "Los clústeres no identifican secuencias ground truth ni acreditan eliminación de leakage.",
    "missing_values": "null significa no disponible o no definido; nunca se sustituye por cero.",
}
BASE_LIMITATIONS = [
    "Las estrategias pueden tener distinta composición de test, distribución de clases y dificultad visual.",
    "Las detector seeds de un split comparten contexto; no son observaciones independientes de split.",
    "Puede permanecer dependencia residual entre frames.",
    "El bootstrap por imagen supone independencia y puede subestimar incertidumbre con dependencia residual entre frames.",
    SEMANTICS["temporal_at5"],
    SEMANTICS["clustering"],
    SEMANTICS["comparison"],
]


def clean(value):
    """Convert only missing values to null; infinities are integrity failures."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if not np.isfinite(value):
            raise ValueError("Non-finite scientific value")
    return value


def write_payload(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            clean(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def records(table: pd.DataFrame) -> list[dict]:
    return clean(table.to_dict("records"))


def require_complete(evidence: AssociationEvidence) -> None:
    gate = evidence.fairness
    if (
        not evidence.complete
        or evidence.issues
        or gate["errors"]
        or gate["expected_runs"] == 0
        or gate["completed_runs"] != gate["expected_runs"]
        or len(evidence.matrix) != gate["expected_runs"]
        or not evidence.matrix.status.eq("COMPLETE").all()
    ):
        raise ValueError(
            f"Final report requires complete controlled evidence: {gate['completed_runs']}/"
            f"{gate['expected_runs']}; state={evidence.state}; errors={gate['errors']}. "
            "Use the clone containing the completed frozen experiment; inspect invalid/missing cells. "
            "Existing exports were not changed."
        )


def correlation(points: pd.DataFrame, method: str = "pearson") -> float | None:
    paired = points[["x", "y"]].dropna()
    if len(paired) < 3 or paired.x.nunique() < 2 or paired.y.nunique() < 2:
        return None
    if not np.isfinite(paired.to_numpy()).all():
        raise ValueError("Invalid association point")
    # pandas ranks implement average ranks for ties; no inferential p-values.
    if method == "spearman":
        paired = paired.rank(method="average")
    return clean(float(paired.x.corr(paired.y)))


def centered_correlation(points: pd.DataFrame) -> float | None:
    paired = points.dropna(subset=["x", "y"]).copy()
    # Rounding of a constant group's mean must not create artificial variation.
    varying = paired.groupby("strategy")[["x", "y"]].nunique().gt(1).any()
    if not varying.all():
        return None
    paired[["x", "y"]] -= paired.groupby("strategy")[["x", "y"]].transform("mean")
    return correlation(paired)


def association_results(
    evidence: AssociationEvidence, registry: pd.DataFrame
) -> list[dict]:
    require_complete(evidence)
    result = []
    for spec in (
        prespecified_table(evidence, registry)
        .sort_values("association_id")
        .itertuples()
    ):
        selected = evidence.associations.loc[
            evidence.associations.class_id.eq(spec.class_id)
        ].sort_values([*SPLIT_KEYS, "detector_seed"])
        grouped = selected.groupby(SPLIT_KEYS, sort=True)
        if (grouped[spec.residual_metric].nunique(dropna=False) != 1).any():
            raise ValueError(
                "Residual context varies across detector seeds of the same split"
            )
        points = grouped.agg(
            x=(spec.residual_metric, "first"), y=(spec.detector_metric, "mean")
        ).reset_index()
        r, rho, centered = (
            correlation(points),
            correlation(points, "spearman"),
            centered_correlation(points),
        )
        reduction = None if r is None or centered is None else abs(r) - abs(centered)
        # No post-hoc significance threshold: expose the actual change in |r|.
        message = "Asociación global y asociación centrada por estrategia son descriptivas, no causales."
        if reduction is not None:
            message += f" El centrado cambia |r| de {abs(r):.6f} a {abs(centered):.6f}."
            if reduction > 0:
                message += " La reducción es compatible con una contribución de diferencias entre estrategias a la asociación global."
        result.append(
            {
                **{
                    k: getattr(spec, k)
                    for k in (
                        "association_id",
                        "category",
                        "population",
                        "class_id",
                        "detector_metric",
                        "residual_metric",
                    )
                },
                "prespecified": True,
                "n_splits": len(points),
                "n_valid_splits": len(points.dropna(subset=["x", "y"])),
                "global_pearson": r,
                "global_spearman": rho,
                "within_strategy_centered_pearson": centered,
                "per_strategy": [
                    {
                        "strategy": strategy,
                        "n_valid_splits": len(part.dropna(subset=["x", "y"])),
                        "pearson": correlation(part),
                        "spearman": correlation(part, "spearman"),
                    }
                    for strategy, part in points.groupby("strategy", sort=True)
                ],
                "points": records(points),
                "interpretation": {
                    "causal": False,
                    "unit_of_analysis": SEMANTICS["unit_of_analysis"],
                    "global_vs_within": message,
                    "centering_reduces_absolute_r": None
                    if reduction is None
                    else reduction > 0,
                    "absolute_r_reduction": reduction,
                    "undefined_correlation": "null si hay menos de tres pares válidos o si una variable es constante; sin p-values ni inferencia causal.",
                },
            }
        )
    return result


def analysis_tables(evidence: AssociationEvidence) -> dict[str, pd.DataFrame]:
    require_complete(evidence)
    rows = evidence.associations.sort_values(
        [*SPLIT_KEYS, "detector_seed", "class_id"]
    ).reset_index(drop=True)
    aggregates = aggregate_runs(rows)
    training = aggregates["training_seed_summary"]
    split_summary = aggregates["split_seed_summary"]
    population_keys = ["strategy", "split_seed", "class_id"]
    means = (
        training.pivot(index=population_keys, columns="metric", values="mean")
        .add_prefix("mean_")
        .reset_index()
    )
    stds = training.loc[
        training.metric.eq("map50_95"), [*population_keys, "std"]
    ].rename(columns={"std": "std_map50_95"})
    means = means.merge(stds, on=population_keys, validate="one_to_one")
    means = means.merge(
        evidence.context[SPLIT_KEYS],
        on=["strategy", "split_seed"],
        validate="many_to_one",
    )
    overall = rows.loc[rows.class_id.eq(-1)]
    splits = means.loc[means.class_id.eq(-1)].drop(columns="class_id")
    splits = splits.merge(
        evidence.context[[*SPLIT_KEYS, *RESIDUALS]],
        on=SPLIT_KEYS,
        validate="one_to_one",
    )
    counts = overall.groupby(SPLIT_KEYS).size().rename("n_detector_runs").reset_index()
    splits = splits.merge(counts, on=SPLIT_KEYS, validate="one_to_one")
    strategies, classes, variance = [], [], []
    for strategy, part in splits.groupby("strategy", sort=True):
        summary = split_summary.loc[
            (split_summary.strategy == strategy) & (split_summary.class_id == -1)
        ].set_index("metric")
        map_stats = summary.loc["map50_95"]
        between, within = map_stats["std"], part.std_map50_95.mean()
        strategies.append(
            {
                "strategy": strategy,
                "display_name": STRATEGY_LABELS[strategy],
                "n_splits": len(part),
                "n_detector_runs": int(part.n_detector_runs.sum()),
                **{f"mean_{m}": summary.loc[m, "mean"] for m in METRICS},
                "std_between_splits": between,
                **{f"{k}_map50_95": map_stats[k] for k in ("median", "min", "max")},
                **{
                    f"mean_{k}": part[k].mean()
                    for k in ("temporal_at5", "dinov2_nn_mean", "clip_nn_mean")
                },
            }
        )
        variance.append(
            {
                "strategy": strategy,
                "n_splits": len(part),
                "mean_within_split_detector_seed_std": within,
                "between_split_std": between,
                "between_vs_within_ratio": between / within
                if pd.notna(within) and within > 0
                else None,
            }
        )
    for (strategy, cid), part in split_summary.loc[
        split_summary.class_id.ge(0)
    ].groupby(["strategy", "class_id"], sort=True):
        stats = part.set_index("metric")
        names = rows.loc[rows.class_id.eq(cid), "class_name"].unique()
        if len(names) != 1:
            raise ValueError("Inconsistent canonical class name")
        classes.append(
            {
                "strategy": strategy,
                "class_id": cid,
                "class_name": names[0],
                "n_splits": int(evidence.context.strategy.eq(strategy).sum()),
                **{f"mean_{m}": stats.loc[m, "mean"] for m in METRICS},
                "std_map50_95": stats.loc["map50_95", "std"],
            }
        )
    class_rows = rows.loc[rows.class_id.ge(0)]
    if (
        class_rows.groupby([*SPLIT_KEYS, "class_id"]).support.nunique(dropna=False) != 1
    ).any():
        raise ValueError("Test support differs across detector seeds")
    support = class_rows[
        [*SPLIT_KEYS, "class_id", "class_name", "support"]
    ].drop_duplicates()
    return {
        "run_level_metrics": rows,
        "split_level_metrics": splits,
        "split_class_metrics": means.loc[means.class_id.ge(0)].reset_index(drop=True),
        "strategy_level_metrics": pd.DataFrame(strategies),
        "class_level_metrics": pd.DataFrame(classes),
        "test_support": support.reset_index(drop=True),
        "variance_summary": pd.DataFrame(variance),
    }


def _overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def check_destinations(
    plan: Path, artifacts: Path, split_root: Path, analysis: Path, frontend: Path
) -> None:
    sources = [
        plan,
        artifacts / "runs",
        artifacts / "protocol",
        artifacts / "views",
        split_root,
        Path("artifacts/detection/runs"),
        Path("artifacts/detection/protocol"),
        Path("artifacts/splitting/runs"),
    ]
    outputs = [analysis.resolve(), frontend.resolve()]
    if _overlap(*outputs) or any(
        _overlap(o, s.resolve()) for o in outputs for s in sources
    ):
        raise ValueError(
            "Report destinations must be disjoint from source artifacts and each other"
        )
    if frontend.resolve().is_relative_to(artifacts.resolve()):
        raise ValueError("Frontend export must live outside heavy detection artifacts")
    for output in (analysis, frontend):
        if output.is_symlink() or any(p.is_symlink() for p in output.rglob("*")):
            raise ValueError("Report destination cannot contain symlinks")
        if output.exists() and not output.is_dir():
            raise ValueError("Report destination must be a directory")
    # Replacing a derived directory must never discard unrelated user files.
    table_names = {
        "run_level_metrics",
        "split_level_metrics",
        "split_class_metrics",
        "strategy_level_metrics",
        "class_level_metrics",
        "test_support",
        "variance_summary",
        "prespecified_association_results",
        "bootstrap_intervals",
    }
    from flir_pipeline.detection.final_report_plot import FIGURES

    analysis_files = {f"{n}.{ext}" for n in table_names for ext in ("csv", "parquet")}
    analysis_files |= {f"figures/{n}.{ext}" for n in FIGURES for ext in ("png", "svg")}
    analysis_files |= {"REPORT.md", "receipt.json"}
    frontend_files = {f"{n}.json" for n in DetectionExport.model_fields}
    frontend_files.add(f"schema/{SCHEMA_VERSION}.schema.json")
    for output, allowed in ((analysis, analysis_files), (frontend, frontend_files)):
        existing = {
            p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
        }
        if existing - allowed:
            raise ValueError(
                "Report destination contains unrelated files; choose a dedicated output directory"
            )


def source_snapshot(
    plan_directory: Path, artifacts: Path, split_root: Path, plan: dict, registry: Path
) -> dict:
    """Fingerprint consumed sources before verification and again before publication.

    Run trees include weights/stats only for read-only integrity checking. Their
    paths and hashes stay in the local receipt, never in the frontend payload.
    """
    paths = {"registry": registry}
    for prefix, directory in (
        ("protocol", plan_directory),
        ("runs", artifacts / "runs"),
    ):
        paths.update(
            {
                f"{prefix}/{p.relative_to(directory).as_posix()}": p
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            }
        )
    for split in plan["identity"]["splits"]:
        sid = split["split_space_id"]
        for prefix, root, names in (
            ("views", artifacts / "views", ("materialization.json", "records.parquet")),
            (
                "splits",
                split_root,
                (
                    "metadata.json",
                    "quantile_pairs_dinov2.parquet",
                    "quantile_pairs_clip.parquet",
                ),
            ),
        ):
            for name in names:
                path = root / sid / name
                if path.exists():
                    paths[f"{prefix}/{sid}/{name}"] = path
    return {key: file_sha256(path) for key, path in sorted(paths.items())}


def _publish(staged: list[tuple[Path, Path]]) -> None:
    """Replace derived directories with rollback on ordinary I/O failure.

    Stages and backups are siblings on the destination filesystem. Source paths
    have already been excluded. Concurrent writers must not target these outputs.
    """
    backups, published = [], []
    try:
        for stage, destination in staged:
            backup = stage.parent / f"{stage.name}-previous"
            if destination.exists():
                destination.rename(backup)
                backups.append((backup, destination))
            stage.rename(destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            shutil.rmtree(destination)
        for backup, destination in reversed(backups):
            backup.rename(destination)
        raise
    for backup, _ in backups:
        shutil.rmtree(backup)


def generate_final_report(
    plan_directory: Path,
    artifacts: Path,
    split_root: Path,
    analysis_output: Path,
    frontend_output: Path,
    association_config: Path = DEFAULT_CONFIG,
) -> dict:
    check_destinations(
        plan_directory, artifacts, split_root, analysis_output, frontend_output
    )
    if any(
        _overlap(output.resolve(), association_config.resolve())
        for output in (analysis_output, frontend_output)
    ):
        raise ValueError(
            "Report destinations must not overlap the association registry"
        )
    plan = verify_plan(plan_directory)
    before = source_snapshot(
        plan_directory, artifacts, split_root, plan, association_config
    )
    evidence = load_evidence(plan_directory, artifacts, split_root=split_root)
    require_complete(evidence)
    if verify_plan(plan_directory) != plan:
        raise ValueError("Plan changed while loading evidence; publication refused")
    freeze = read_json(plan_directory / "runtime_freeze.json")
    config = freeze["model_config"]["protocol_config"]
    registry = load_registry(association_config)
    tables = analysis_tables(evidence)
    associations = association_results(evidence, registry)
    tables["prespecified_association_results"] = pd.DataFrame(
        [
            {
                k: v
                for k, v in a.items()
                if k not in {"points", "per_strategy", "interpretation"}
            }
            for a in associations
        ]
    )
    runs, intervals = [], []
    for row in tables["run_level_metrics"].query("class_id == -1").itertuples():
        directory = artifacts / "runs" / row.detector_run_id
        training = read_json(directory / "training_summary.json")
        runs.append(
            {
                **{
                    k: getattr(row, k)
                    for k in (*SPLIT_KEYS, "detector_run_id", "detector_seed", *METRICS)
                },
                "state": "COMPLETE",
                **{
                    k: training.get(k)
                    for k in ("best_epoch", "training_seconds", "peak_memory_bytes")
                },
            }
        )
        bootstrap = read_json(directory / "bootstrap.json")
        entries = bootstrap["intervals"]
        expected = {(cid, m) for cid in range(-1, 5) for m in METRICS}
        if (
            len(entries) != len(expected)
            or {(e["class_id"], e["metric"]) for e in entries} != expected
        ):
            raise ValueError("Missing or duplicate stored bootstrap intervals")
        for entry in sorted(entries, key=lambda e: (e["class_id"], e["metric"])):
            interval = Interval.model_validate(
                {"detector_run_id": row.detector_run_id, **entry}
            )
            if interval.valid_resamples > bootstrap["resamples"]:
                raise ValueError("Bootstrap valid_resamples exceeds stored resamples")
            intervals.append(interval.model_dump())
    tables["bootstrap_intervals"] = pd.DataFrame(intervals)
    provenance = execution_provenance()
    provenance["report_package_versions"] = {
        package: version(package)
        for package in (
            "numpy",
            "pandas",
            "pyarrow",
            "matplotlib",
            "pydantic",
            "PyYAML",
        )
    }
    limitations = [
        *BASE_LIMITATIONS,
        *[
            f"{r.strategy}: {r.n_splits} split(s); "
            + (
                "variación entre splits no definida."
                if r.n_splits == 1
                else "generalización limitada a estas particiones evaluadas."
            )
            for r in tables["strategy_level_metrics"].itertuples()
        ],
    ]
    protocol = {
        "plan_id": plan["plan_id"],
        "model_config_id": freeze["model_config_id"],
        "model": Path(config["model"].replace("\\", "/")).name,
        "epochs": config["train"]["epochs"],
        "batch": config["train"]["batch"],
        "detector_seeds": sorted(config["training_seeds"]),
        "bootstrap_method": "image_percentile_bootstrap",
        **{
            k: config["evaluation"][k]
            for k in ("bootstrap_resamples", "bootstrap_seed", "confidence_level")
        },
    }
    strategies, variance = (
        records(tables["strategy_level_metrics"]),
        records(tables["variance_summary"]),
    )
    bundle = {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "state": COMPLETE,
            "scientific_result": True,
            "generated_from_verified_artifacts": True,
            "plan_id": plan["plan_id"],
            "model_config_id": freeze["model_config_id"],
            "expected_runs": evidence.fairness["expected_runs"],
            "completed_runs": evidence.fairness["completed_runs"],
            "strategy_count": len(strategies),
            "split_count": len(evidence.context),
            "detector_seed_count": len(config["training_seeds"]),
            "excluded_small_pilots": evidence.excluded_pilots,
            "generated_at": provenance["created_at"],
            "source_commit": provenance["git_commit"],
            "source_worktree_dirty": provenance["git_worktree_dirty"],
            "limitations": limitations,
            "source_receipt_sha256": {
                k: before[k]
                for k in (
                    "protocol/plan.json",
                    "protocol/runtime_freeze.json",
                    "registry",
                )
            },
            "file_sha256": {},
        },
        "summary": {
            "state": COMPLETE,
            "scientific_result": True,
            "protocol": protocol,
            "headline_metrics": {r["strategy"]: r["mean_map50_95"] for r in strategies},
            "strategies": strategies,
            "variance": variance,
            "association_summary": {
                a["association_id"]: {
                    k: a[k]
                    for k in (
                        "n_splits",
                        "n_valid_splits",
                        "global_pearson",
                        "global_spearman",
                        "within_strategy_centered_pearson",
                    )
                }
                for a in associations
            },
            "interpretation": SEMANTICS,
            "limitations": limitations,
        },
        "strategies": strategies,
        "variance": variance,
        "splits": records(tables["split_level_metrics"]),
        "runs": runs,
        "classes": records(tables["class_level_metrics"]),
        "support": records(tables["test_support"]),
        "associations": associations,
        "bootstrap": intervals,
    }
    bundle = DetectionExport.model_validate(clean(bundle)).model_dump()
    from flir_pipeline.detection.final_report_plot import (
        generate_figures,
        markdown_report,
    )

    stages = []
    try:
        for output in (analysis_output, frontend_output):
            output.parent.mkdir(parents=True, exist_ok=True)
            stages.append(
                Path(
                    tempfile.mkdtemp(prefix=f".{output.name}-stage-", dir=output.parent)
                )
            )
        analysis, frontend = stages
        for name, table in tables.items():
            table.to_csv(analysis / f"{name}.csv", index=False)
            table.to_parquet(analysis / f"{name}.parquet", index=False)
        generate_figures(tables, bundle, analysis / "figures")
        (analysis / "REPORT.md").write_text(markdown_report(bundle), encoding="utf-8")
        (frontend / "schema").mkdir()
        write_payload(
            frontend / "schema" / f"{SCHEMA_VERSION}.schema.json", export_schema()
        )
        for name, payload in bundle.items():
            if name != "manifest":
                write_payload(frontend / f"{name}.json", payload)
        bundle["manifest"]["file_sha256"] = {
            p.relative_to(frontend).as_posix(): file_sha256(p)
            for p in sorted(frontend.rglob("*.json"))
        }
        DetectionExport.model_validate(bundle)
        write_payload(frontend / "manifest.json", bundle["manifest"])
        after = source_snapshot(
            plan_directory, artifacts, split_root, plan, association_config
        )
        if before != after:
            raise ValueError(
                "Source artifacts changed during verification/reporting; publication refused"
            )
        receipt = {
            "state": COMPLETE,
            "scientific_result": True,
            "gate": evidence.fairness,
            "plan_id": plan["plan_id"],
            "model_config_id": freeze["model_config_id"],
            "sources_unchanged": True,
            "source_sha256": before,
            "provenance": provenance,
            "report_code_sha256": {
                p.name: file_sha256(p)
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "analysis_sha256": {
                p.relative_to(analysis).as_posix(): file_sha256(p)
                for p in sorted(analysis.rglob("*"))
                if p.is_file()
            },
            "frontend_sha256": {
                p.relative_to(frontend).as_posix(): file_sha256(p)
                for p in sorted(frontend.rglob("*"))
                if p.is_file()
            },
        }
        write_payload(analysis / "receipt.json", receipt)
        _publish([(analysis, analysis_output), (frontend, frontend_output)])
        return bundle["manifest"]
    finally:
        for stage in stages:
            if stage.exists():
                shutil.rmtree(stage)
