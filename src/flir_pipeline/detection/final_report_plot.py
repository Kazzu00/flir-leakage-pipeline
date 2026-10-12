"""Thesis figures and narrative for the verified final report, without inference."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from flir_pipeline.data.classes import class_name
from flir_pipeline.detection.association import (
    COLORS,
    MARKERS,
    METRIC_LABELS,
    RESIDUAL_LABELS,
    STRATEGIES,
)

STYLE = dict(zip(STRATEGIES, zip(COLORS, MARKERS, strict=True), strict=True))
CAPTION = "descriptive association — not causal estimate"
FIGURES = (
    "A_overall_performance",
    "B_per_class_performance",
    "C_variability",
    "D_test_support",
    "E_prespecified_associations",
    "F_temporal_at5",
)


def _save(fig, directory: Path, name: str) -> None:
    try:
        fig.savefig(
            directory / f"{name}.png",
            dpi=220,
            metadata={"Software": "flir-leakage-pipeline"},
        )
        fig.savefig(
            directory / f"{name}.svg",
            metadata={"Date": None, "Creator": "flir-leakage-pipeline"},
        )
    finally:
        plt.close(fig)


def _number(value) -> str:
    return "no definido" if value is None or pd.isna(value) else f"{value:.3f}"


def _metric_limits(ax) -> None:
    lower, upper = ax.get_ylim()
    ax.set_ylim(max(0, lower), min(1, upper))


def _association_panel(ax, association: dict) -> None:
    points = pd.DataFrame(association["points"])
    for strategy, group in points.groupby("strategy", sort=True):
        color, marker = STYLE[strategy]
        ax.scatter(group.x, group.y, color=color, marker=marker, label=strategy, s=38)
    population = (
        "Overall"
        if association["class_id"] == -1
        else class_name(association["class_id"])
    )
    ax.set_title(
        f"{association['association_id']} · {association['category']} · {population}\n"
        f"r={_number(association['global_pearson'])}; ρ={_number(association['global_spearman'])}; "
        f"r centrado={_number(association['within_strategy_centered_pearson'])}",
        fontsize=10,
    )
    label = RESIDUAL_LABELS[association["residual_metric"]]
    if association["residual_metric"] == "temporal_at5":
        label += " · distancia de índices, no segundos"
    ax.set(xlabel=label, ylabel=METRIC_LABELS[association["detector_metric"]])
    _metric_limits(ax)
    ax.grid(alpha=0.2)


def generate_figures(
    tables: dict[str, pd.DataFrame], bundle: dict, directory: Path
) -> None:
    directory.mkdir()
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "flir-detection-export-v1",
        }
    ):
        splits = tables["split_level_metrics"]
        runs = tables["run_level_metrics"].query("class_id == -1")
        positions = {r.split_space_id: i for i, r in enumerate(splits.itertuples())}
        short_labels = {"random_content": "random", "historical": "histórico"}
        labels = [
            f"{short_labels.get(r.strategy, r.strategy)}\nsplit {r.split_seed}"
            for r in splits.itertuples()
        ]
        fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
        for row in splits.itertuples():
            x = positions[row.split_space_id]
            color, marker = STYLE[row.strategy]
            part = runs.loc[runs.split_space_id.eq(row.split_space_id)].sort_values(
                "detector_seed"
            )
            # Fixed offsets identify detector seeds without random visual jitter.
            offset = np.linspace(-0.19, 0.19, len(part)) if len(part) > 1 else [0]
            ax.scatter(
                x + np.asarray(offset),
                part.map50_95,
                color=color,
                marker=marker,
                alpha=0.65,
                s=24,
            )
            ax.plot(
                [x - 0.25, x + 0.25],
                [row.mean_map50_95] * 2,
                color="black",
                linewidth=2,
            )
        ax.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            ylabel="mAP@50–95",
            title="A · Detector seeds y media por split (segmento negro) · comparación descriptiva",
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("random = random_content; histórico = historical")
        _metric_limits(ax)
        _save(fig, directory, FIGURES[0])

        fig, axes = plt.subplots(
            1, 5, figsize=(16, 5), layout="constrained", sharey=True
        )
        classes = tables["split_class_metrics"]
        strategies = sorted(splits.strategy.unique())
        for cid, ax in enumerate(axes):
            for i, strategy in enumerate(strategies):
                part = classes.loc[
                    (classes.class_id == cid) & (classes.strategy == strategy)
                ]
                color, marker = STYLE[strategy]
                offsets = np.linspace(-0.2, 0.2, len(part)) if len(part) > 1 else [0]
                ax.scatter(
                    i + np.asarray(offsets),
                    part.mean_map50_95,
                    color=color,
                    marker=marker,
                )
            ax.set(
                title=class_name(cid),
                xticks=range(len(strategies)),
                xticklabels=strategies,
            )
            ax.tick_params(axis="x", rotation=50)
        axes[0].set_ylabel("Media mAP@50–95 entre detector seeds")
        _metric_limits(axes[0])
        fig.suptitle("B · Un punto por split; histórico tiene una única partición")
        _save(fig, directory, FIGURES[1])

        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for i, row in enumerate(bundle["variance"]):
            within, between = (
                row["mean_within_split_detector_seed_std"],
                row["between_split_std"],
            )
            if within is not None:
                ax.scatter(
                    i - 0.1,
                    within,
                    color="#137c8b",
                    marker="o",
                    label="SD media entre detector seeds" if i == 0 else None,
                )
            if between is not None:
                ax.scatter(
                    i + 0.1,
                    between,
                    color="#6943a5",
                    marker="D",
                    label="SD entre splits" if i == 0 else None,
                )
            ratio = row["between_vs_within_ratio"]
            text = f"razón={ratio:.2f}" if ratio is not None else "razón no definida"
            ax.annotate(
                text,
                (i, max(within or 0, between or 0)),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
            )
        ax.set(
            xticks=range(len(bundle["variance"])),
            xticklabels=[r["strategy"] for r in bundle["variance"]],
            ylabel="SD de mAP@50–95",
            title="C · Variación por construcción del split y por detector seed (SD muestral)",
        )
        ax.margins(y=0.3)
        ax.legend()
        _save(fig, directory, FIGURES[2])

        support = tables["test_support"].pivot(
            index="class_id", columns="split_space_id", values="support"
        )
        support = support.reindex(index=range(5), columns=list(positions))
        fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
        maximum = max(1, float(support.to_numpy().max()))
        heat = ax.imshow(support, cmap="Blues", aspect="auto", vmin=0, vmax=maximum)
        for cid in range(5):
            for x, value in enumerate(support.iloc[cid]):
                ax.text(
                    x,
                    cid,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > maximum * 0.55 else "black",
                )
        ax.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            yticks=range(5),
            yticklabels=[class_name(i) for i in range(5)],
            title="D · Soporte de test: instancias por clase y split",
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("random = random_content; histórico = historical")
        fig.colorbar(
            heat, ax=ax, label="Instancias anotadas", ticks=MaxNLocator(integer=True)
        )
        _save(fig, directory, FIGURES[3])

        associations = bundle["associations"]
        fig, axes = plt.subplots(
            (len(associations) + 1) // 2,
            2,
            figsize=(14, 4 * ((len(associations) + 1) // 2)),
            squeeze=False,
            layout="constrained",
        )
        for ax, association in zip(axes.flat, associations, strict=False):
            _association_panel(ax, association)
        for ax in list(axes.flat)[len(associations) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(
            f"E · Asociaciones preespecificadas · un punto por split\n{CAPTION}"
        )
        _save(fig, directory, FIGURES[4])
        temporal = [a for a in associations if a["residual_metric"] == "temporal_at5"]
        # The registry owns what is primary; a missing temporal entry is never invented.
        if temporal:
            fig, axes = plt.subplots(
                1,
                len(temporal),
                figsize=(9 * len(temporal), 5),
                squeeze=False,
                layout="constrained",
            )
            for ax, association in zip(axes.flat, temporal, strict=True):
                _association_panel(ax, association)
                ax.legend(fontsize=8)
            fig.suptitle(f"F · Temporalidad inferida por índice\n{CAPTION}")
            _save(fig, directory, FIGURES[5])


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = [
            "null"
            if row[c] is None
            else f"{row[c]:.6f}"
            if isinstance(row[c], float)
            else str(row[c])
            for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_report(bundle: dict) -> str:
    m, summary = bundle["manifest"], bundle["summary"]
    p = summary["protocol"]
    lines = [
        "# Reporte final descriptivo del detector",
        "",
        "## Experimental setup",
        "",
        f"Modelo `{p['model']}`; {p['epochs']} epochs; batch {p['batch']}; {m['split_count']} splits; "
        f"{m['completed_runs']} runs; detector seeds {p['detector_seeds']}.",
        "",
        f"Plan `{p['plan_id']}`; runtime congelado `{p['model_config_id']}`. "
        f"Código del reporte: `{m['source_commit']}` (working tree dirty: {m['source_worktree_dirty']}).",
        "",
        f"Bootstrap existente `{p['bootstrap_method']}`: {p['bootstrap_resamples']} remuestreos, "
        f"seed {p['bootstrap_seed']}, nivel {p['confidence_level']}. No se promedian intervalos ni se recalcula bootstrap.",
        "",
        "## Integrity gate",
        "",
        f"Gate controlado superado: {m['completed_runs']}/{m['expected_runs']}; sin errores. "
        f"Pilotos pequeños excluidos: {m['excluded_small_pilots']}. Las celdas Stage A se cuentan una sola vez.",
        "",
        "## Overall results",
        "",
        _markdown_table(
            bundle["strategies"],
            [
                "strategy",
                "n_splits",
                "mean_map50_95",
                "std_between_splits",
                "mean_map50",
                "mean_precision",
                "mean_recall",
            ],
        ),
        "",
        "Cada estrategia promedia primero detector seeds por split y después splits con el mismo peso. "
        "Las diferencias están condicionadas por composición y dificultad; no definen un ganador causal.",
        "",
        "![Observaciones y medias](figures/A_overall_performance.png)",
        "",
        "## Split vs detector-seed variability",
        "",
        _markdown_table(
            bundle["variance"],
            [
                "strategy",
                "mean_within_split_detector_seed_std",
                "between_split_std",
                "between_vs_within_ratio",
            ],
        ),
        "",
        "SD muestral (ddof=1). La razón compara SD entre medias de splits con la media de SD entre detector seeds; "
        "es un resumen descriptivo, no un modelo de componentes de varianza. Una sola partición no define SD entre splits.",
        "",
    ]
    for row in bundle["variance"]:
        ratio = row["between_vs_within_ratio"]
        if ratio is not None:
            lines.append(
                f"- {row['strategy']}: razón {ratio:.3f}; la variación entre splits fue "
                f"{'mayor' if ratio > 1 else 'igual o menor'} que la variación media entre detector seeds bajo este protocolo."
            )
    lines += [
        "",
        "![Variabilidad](figures/C_variability.png)",
        "",
        "## Per-class results",
        "",
        _markdown_table(
            bundle["classes"],
            [
                "strategy",
                "class_name",
                "n_splits",
                "mean_map50_95",
                "std_map50_95",
                "mean_recall",
            ],
        ),
        "",
        "El soporte describe instancias, no imágenes independientes. Especialmente para Heavy Machinery, "
        "el soporte y la composición deben examinarse antes de interpretar diferencias de Recall. "
        "No se atribuyen esas diferencias únicamente a dependencia residual.",
        "",
        "![Clases](figures/B_per_class_performance.png)",
        "",
        "![Soporte](figures/D_test_support.png)",
        "",
        "## Pre-specified residual associations",
        "",
        _markdown_table(
            bundle["associations"],
            [
                "association_id",
                "n_splits",
                "n_valid_splits",
                "global_pearson",
                "global_spearman",
                "within_strategy_centered_pearson",
            ],
        ),
        "",
        "Un punto por split después de promediar detector seeds. El centrado resta las medias de cada estrategia "
        "en x e y usando pares válidos; no controla causalmente composición o dificultad. Correlaciones con "
        "menos de tres pares o variables constantes permanecen null.",
        "",
    ]
    lines += [
        f"- {a['association_id']}: {a['interpretation']['global_vs_within']}"
        for a in bundle["associations"]
    ]
    lines += [
        "",
        "![Asociaciones](figures/E_prespecified_associations.png)",
        "",
        "## Interpretation",
        "",
        summary["interpretation"]["comparison"],
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {text}" for text in summary["limitations"]]
    lines += [
        "",
        "## Frontend export",
        "",
        "Contrato `detection-export-v1`. `summary.json` es la entrada de presentación; `manifest.json` "
        "identifica fuentes verificadas y hashes de los archivos. `bootstrap.json` conserva intervalos existentes "
        "por run/clase/métrica. El frontend solo visualiza valores publicados; no agrega seeds ni calcula correlaciones.",
        "",
    ]
    return "\n".join(lines)
