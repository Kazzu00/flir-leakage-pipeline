"""Evidence-gated detector review: absent scientific results remain visibly pending."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from flir_pipeline.detection.metrics import aggregate_runs
from flir_pipeline.detection.protocol import METRICS, experiment_matrix, verify_plan
from flir_pipeline.detection.runtime import fair_comparison, verify_run
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json

STRATEGIES = ["historical", "random_content", "C10", "C12"]
COLORS = ["#7d8597", "#e0a526", "#137c8b", "#6943a5"]
FIGURES = ["01_split_correlation_context", "02_map50_95_by_strategy", "03_map50_by_strategy",
           "04_precision_recall_by_strategy", "05_heavy_machinery_metrics", "06_per_class_map",
           "07_training_seed_variability", "08_split_seed_variability", "09_metrics_vs_residual_similarity"]
TITLES = ["Correlación residual antes de entrenar", "mAP@50–95 por estrategia", "mAP@50 por estrategia",
          "Precision y Recall a umbral fijo", "Heavy Machinery", "mAP por clase",
          "Variabilidad entre training seeds", "Variabilidad entre split seeds", "Métricas y similitud residual"]


def collect_results(plan_directory: Path, artifacts: Path) -> tuple[pd.DataFrame, dict]:
    plan = verify_plan(plan_directory)
    matrix = experiment_matrix(plan["identity"]["splits"], plan["identity"]["config"]["training_seeds"])
    lookup = {s["split_space_id"]: s for s in plan["identity"]["splits"]}
    freeze_path = plan_directory/"runtime_freeze.json"
    if not freeze_path.exists():
        return pd.DataFrame(), {"controlled": False, "completed_runs": 0, "expected_runs": len(matrix), "errors": ["runtime_not_frozen"]}
    freeze = read_json(freeze_path)
    metas, rows = [], []
    for meta_path in sorted((artifacts/"runs").glob("*/metadata.json")):
        meta = read_json(meta_path)
        if meta["plan_id"] != plan["plan_id"]:
            continue
        split = lookup[meta["identity"]["split_space_id"]]
        meta = verify_run(meta_path.parent, split, freeze)
        metas.append(meta)
        common = {"detector_run_id": meta["detector_run_id"], "strategy": meta["strategy"], "split_seed": meta["split_seed"],
                  "detector_seed": meta["detector_seed"], **meta["context"]}
        intervals = read_json(meta_path.parent/"bootstrap.json")["intervals"]
        ci_columns = {}
        for interval in intervals:
            target = ci_columns.setdefault(interval["class_id"], {})
            target.update({f"{interval['metric']}_ci_{key}": interval[key] for key in ("lower", "upper", "valid_resamples")})
        overall = read_json(meta_path.parent/"metrics.json")["overall"]
        rows.append({**common, "class_id": -1, "class_name": "Overall", "support": overall["instance_count"], **{m: overall[m] for m in METRICS}, **ci_columns[-1]})
        classes = pd.read_parquet(meta_path.parent/"metrics_per_class.parquet")
        rows.extend({**common, **r, **ci_columns[r["class_id"]]} for r in classes.to_dict("records"))
    return pd.DataFrame(rows), fair_comparison(metas, matrix)


def generate_report(plan_directory: Path, artifacts: Path, output: Path) -> dict:
    plan = verify_plan(plan_directory)
    output.mkdir(parents=True, exist_ok=True)
    figures = output/"figures"
    tables = output/"tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    context = pd.DataFrame([{"strategy": s["strategy"], "split_seed": s["split_seed"], "split_space_id": s["split_space_id"], **s["context"]}
                            for s in plan["identity"]["splits"]])
    context.to_csv(tables/"split_context.csv", index=False)
    audit = pd.read_csv(plan_directory/"candidate_audit.csv")
    audit.to_csv(tables/"candidate_audit.csv", index=False)
    results, fairness = collect_results(plan_directory, artifacts)
    pilot_rows = []
    for p in sorted((artifacts/"small_pilot").glob("*/pilot.json")):
        receipt = read_json(p)
        pilot_rows.append({k: receipt[k] for k in ("strategy", "state", "training_seconds", "evaluation_bootstrap_seconds", "batch", "image_size", "epochs_completed", "estimated_full_training_hours", "scientific_result")})
    pd.DataFrame(pilot_rows).to_csv(tables/"pilot_validation.csv", index=False)
    aggregate = {}
    if len(results):
        results.to_csv(tables/"run_metrics.csv", index=False)
        if fairness["controlled"]:
            aggregate = aggregate_runs(results)
            for name, table in aggregate.items():
                table.to_csv(tables/f"{name}.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    cols = ["exact_duplicate_cross_split_count", "dinov2_top001_pairs", "clip_top001_pairs", "dinov2_nn_mean", "temporal_at5", "class_deviation_pp"]
    labels = ["Contenidos exactos cross-split", "Pares DINOv2 top 0.1%", "Pares CLIP top 0.1%", "NN DINOv2 medio", "Temporal Δ≤5 (fracción)", "Desviación de clases (pp)"]
    for ax, column, label in zip(axs.flat, cols, labels, strict=True):
        group = context.groupby("strategy")[column].mean().reindex(STRATEGIES)
        ax.bar(STRATEGIES, group, color=COLORS)
        ax.set_title(label)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Splits congelados · medias entre split seeds; histórico tiene una sola partición", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures/f"{FIGURES[0]}.png", dpi=150)
    plt.close(fig)
    for i in range(1, 9):
        fig, ax = plt.subplots(figsize=(11, 5))
        if not fairness["controlled"]:
            ax.axis("off")
            ax.text(.5, .62, TITLES[i], ha="center", fontsize=20, fontweight="bold", transform=ax.transAxes)
            ax.text(.5, .42, "PENDIENTE · experimento final no completado", ha="center", fontsize=16, color="#137c8b", transform=ax.transAxes)
            ax.text(.5, .24, f"Runs finales verificables: {fairness['completed_runs']}/{fairness['expected_runs']}\nEl piloto pequeño verifica infraestructura; no aporta una comparación científica.",
                    ha="center", fontsize=12, transform=ax.transAxes)
        else:
            overall = results.loc[results.class_id == -1]
            if i in (1, 2):
                metric = "map50_95" if i == 1 else "map50"
                ax.boxplot([overall.loc[overall.strategy == s, metric] for s in STRATEGIES])
                ax.set_xticks(range(1, 5), STRATEGIES)
                ax.set_ylabel(metric)
            elif i in (3, 4):
                cohort = overall if i == 3 else results.loc[results.class_id == 4]
                columns = ["precision", "recall"] if i == 3 else list(METRICS)
                cohort.groupby("strategy")[columns].mean().reindex(STRATEGIES).plot.bar(ax=ax, rot=0)
                ax.set_ylim(0, 1)
            elif i == 5:
                values = results.loc[results.class_id >= 0].pivot_table(index="class_name", columns="strategy", values="map50_95").reindex(columns=STRATEGIES)
                heat = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
                ax.set_xticks(range(4), STRATEGIES)
                ax.set_yticks(range(len(values)), values.index)
                fig.colorbar(heat, ax=ax, label="mAP@50–95")
            elif i == 6:
                table = aggregate["training_seed_summary"].query("class_id == -1 and metric == 'map50_95'")
                for s, color in zip(STRATEGIES, COLORS, strict=True):
                    part = table.loc[table.strategy == s]
                    ax.scatter(part.split_seed, part["std"], color=color, label=s)
                ax.set(xlabel="Split seed", ylabel="SD entre detector seeds")
                ax.legend()
            elif i == 7:
                table = aggregate["training_seed_summary"].query("class_id == -1 and metric == 'map50_95'")
                for s, color in zip(STRATEGIES, COLORS, strict=True):
                    part = table.loc[table.strategy == s]
                    ax.plot(part.split_seed, part["mean"], "o-", color=color, label=s)
                ax.set(xlabel="Split seed", ylabel="Media mAP@50–95 entre detector seeds")
                ax.legend()
            else:
                for s, color in zip(STRATEGIES, COLORS, strict=True):
                    part = overall.loc[overall.strategy == s]
                    ax.scatter(part.dinov2_nn_mean, part.map50_95, color=color, label=s)
                ax.set(xlabel="NN DINOv2 residual medio", ylabel="mAP@50–95")
                ax.legend()
            ax.set_title(TITLES[i])
        fig.tight_layout()
        fig.savefig(figures/f"{FIGURES[i]}.png", dpi=150)
        plt.close(fig)
    metadata = {"plan_id": plan["plan_id"], "fair_comparison": fairness, "small_pilot_count": len(pilot_rows),
                "scientific_state": "COMPLETE_CONTROLLED_COMPARISON" if fairness["controlled"] else "PENDING_FINAL_EXPERIMENT",
                "figure_policy": "Pending panels contain no invented metric values; small pilot metrics excluded from scientific comparison.",
                "plan_sha256": file_sha256(plan_directory/"plan.json"),
                "output_sha256": {p.relative_to(output).as_posix(): file_sha256(p) for p in [*figures.glob("*.png"), *tables.glob("*.csv")]}}
    budget = artifacts/"compute_budget.json"
    if budget.exists():
        metadata["compute_budget"] = read_json(budget)
    write_json(output/"report_metadata.json", metadata)
    return metadata
