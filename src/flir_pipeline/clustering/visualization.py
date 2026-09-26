"""Verified posterior figures and runtime ZIP exemplars for a bounded review set."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flir_pipeline.clustering.experiments import (  # noqa: E402
    read_table,
    verify_collection,
)
from flir_pipeline.clustering.storage import (  # noqa: E402
    ClusteringFamily,
    clustering_provenance,
)
from flir_pipeline.similarity.reporting import _read_verified_image  # noqa: E402
from flir_pipeline.similarity.storage import (  # noqa: E402
    file_sha256,
    read_json,
    write_json,
)

ALGORITHM_COLORS = {"dbscan": "#267997", "optics": "#ad6340", "hdbscan": "#588843"}
REPRESENTATIONS = ("original_l2", "tsne", "pacmap")
DISPLAY = {"original_l2": "L2 original", "tsne": "t-SNE 2D", "pacmap": "PaCMAP 2D"}


def _save(fig, path: Path) -> None:
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def exemplar_clusters(summary: pd.DataFrame) -> list[tuple[str, int]]:
    """Choose by measured size/sequence properties, independently of image appearance."""
    if summary.empty:
        return []
    sizes = summary.sort_values(["n_members", "cluster_id"])
    median = sizes.assign(gap=(sizes.n_members-sizes.n_members.median()).abs()).sort_values(["gap", "cluster_id"])
    known = summary.loc[summary.known_sequence_members > 0]
    rows = [("Pequeño", int(sizes.iloc[0].cluster_id)), ("Mediano", int(median.iloc[0].cluster_id)),
            ("Grande", int(sizes.sort_values(["n_members", "cluster_id"], ascending=[False, True]).iloc[0].cluster_id))]
    if len(known):
        coherent = known.sort_values(["dominant_sequence_fraction", "known_sequence_members", "cluster_id"], ascending=[False, False, True])
        diverse = known.sort_values(["sequence_entropy_bits", "sequence_count", "cluster_id"], ascending=[False, False, True])
        rows += [("Mayor fracción de secuencia", int(coherent.iloc[0].cluster_id)),
                 ("Mayor entropía de secuencia", int(diverse.iloc[0].cluster_id))]
    return rows


def exemplar_members(labels: np.ndarray, cluster_id: int, medoid: int,
                     context, count: int = 4) -> list[int]:
    """Show medoid plus nearby/middle/far members in ORIGINAL Euclidean distance."""
    members = np.flatnonzero(labels == cluster_id)
    if medoid not in members:
        raise ValueError("Medoid is not a member of the displayed cluster")
    others = sorted((i for i in members if i != medoid), key=lambda i: (context.original_distances[medoid, i], context.content_ids[i]))
    ordered = [medoid, *others]
    targets = [0, 1, (len(ordered)-1)//2, len(ordered)-1]
    selected = list(dict.fromkeys(ordered[i] for i in targets if i < len(ordered)))
    selected.extend(i for i in ordered if i not in selected)
    return selected[:count]


def _gallery(family: ClusteringFamily, directory: Path, row: pd.Series, archive: zipfile.ZipFile,
             output: Path, reference_name: str) -> list[dict]:
    labels = np.load(directory/"cluster_labels.npy", allow_pickle=False)
    clusters = pd.read_parquet(directory/"cluster_summary.parquet")
    contexts = family.context
    index = family.source.content_index
    id_to_row = {identity: i for i, identity in enumerate(contexts.content_ids)}
    selected_roles = []
    for role, cluster in exemplar_clusters(clusters):
        summary = clusters.loc[clusters.cluster_id == cluster].iloc[0]
        medoid = id_to_row[summary.medoid_content_id]
        selected_roles.append((role, cluster, exemplar_members(labels, cluster, medoid, contexts), summary))
    noise = sorted(np.flatnonzero(labels == -1), key=lambda i: contexts.content_ids[i])
    selection = np.random.default_rng(0).choice(noise, min(4, len(noise)), replace=False).tolist() if noise else []
    selected_roles.append(("Noise · semilla de muestra 0", -1, selection, None))
    fig, axes = plt.subplots(len(selected_roles), 4, figsize=(14, 2.6*len(selected_roles)), squeeze=False)
    selections = []
    for i, (role, cluster, members, summary) in enumerate(selected_roles):
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j >= len(members):
                ax.text(.5, .5, "Sin miembros disponibles", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                continue
            member = members[j]
            picture = _read_verified_image(archive, index.iloc[member])
            picture.thumbnail((400, 320))
            ax.imshow(picture, cmap="gray")
            provenance = contexts.provenance.iloc[member]
            frame = "?" if pd.isna(provenance.frame_index) else str(int(provenance.frame_index))
            sequence = provenance.sequence_id or "desconocida"
            member_kind = "Medoide" if cluster != -1 and j == 0 else "Miembro" if cluster != -1 else "Noise"
            cluster_text = f"cluster {cluster}, n={int(summary.n_members)}" if cluster != -1 else f"n={len(noise)}"
            ax.set_title(f"{role} · {cluster_text}\n{member_kind} · {sequence} · índice {frame}", fontsize=8)
            selections.append({"reference": reference_name, "role": role, "cluster_id": cluster, "display_position": j,
                               "content_id": contexts.content_ids[member], "is_medoid": member_kind == "Medoide"})
    fig.suptitle(f"{reference_name} · {row.encoder.upper()} · {DISPLAY[row.representation]} · {row.algorithm.upper()}\n"
                 "Medoide y distancias originales; roles pueden repetir un cluster. Índices temporales inferidos.", fontsize=12)
    _save(fig, output)
    return selections


def generate_clustering_report(comparison: Path, families: dict[str, ClusteringFamily],
                               images_archive: Path, output: Path = Path("reports/clustering")) -> dict:
    if any(f.source.similarity.get("provenance_mode") == "sampled_video_grid" for f in families.values()):
        raise ValueError("This historical ZIP/sequence review is unavailable for sampled video; numerical clustering remains supported")
    if not verify_collection(comparison)["quality_valid"]:
        raise ValueError("Comparison must pass verification before reporting")
    meta = read_json(comparison/"metadata.json")
    if meta["source_signatures"] != {e: f.source.signatures for e, f in families.items()}:
        raise ValueError("Report sources do not match clustering")
    root = comparison.parents[1]
    screening = root/meta["screening_path"]
    screen = read_table(screening/"screening.csv")
    all_runs = read_table(comparison/"all_runs.csv")
    evaluated = read_table(comparison/"evaluated_shortlist.csv")
    references = read_table(comparison/"references.csv").sort_values(["encoder", "representation"]).reset_index(drop=True)
    references.insert(0, "reference", [f"R{i+1}" for i in range(len(references))])
    curves = pd.read_parquet(screening/"k_distance_curves.parquet")
    agreement = pd.read_parquet(comparison/"assignment_comparisons.parquet")
    agreement = agreement.merge(evaluated[["clustering_space_id", "encoder", "representation", "algorithm"]], left_on="reference_id", right_on="clustering_space_id", validate="many_to_one")
    locations = {r["clustering_space_id"]: root/r["path"] for r in meta["runs"]}
    figures, tables = output/"figures", output/"tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figure_names, selections, historical = [], [], []
    encoders = sorted(families)
    fig, axes = plt.subplots(3, len(encoders), figsize=(7*len(encoders), 12), squeeze=False)
    for i, representation in enumerate(REPRESENTATIONS):
        for j, encoder in enumerate(encoders):
            ax = axes[i, j]
            selected = curves.loc[(curves.encoder == encoder) & (curves.representation == representation)]
            for ms, group in selected.groupby("min_samples", sort=True):
                ax.plot(np.linspace(0, 1, len(group)), group.k_distance, label=f"min_samples={ms}", lw=1.4)
            ax.set(title=f"{encoder.upper()} · {DISPLAY[representation]}", xlabel="Fracción acumulada de contenidos", ylabel="k-distance euclidiana (escala propia)")
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
    fig.suptitle("¿Qué escala utiliza DBSCAN en cada espacio? · min_samples incluye self", fontsize=13)
    name = "01_k_distance_scales.png"
    _save(fig, figures/name)
    figure_names.append(name)
    fig, axes = plt.subplots(3, len(encoders), figsize=(7*len(encoders), 12), squeeze=False)
    for i, representation in enumerate(REPRESENTATIONS):
        for j, encoder in enumerate(encoders):
            ax = axes[i, j]
            selected = screen.loc[(screen.encoder == encoder) & (screen.representation == representation)]
            for algorithm, group in selected.groupby("algorithm", sort=True):
                ax.scatter(group.noise_fraction, group.n_clusters_excluding_noise, s=30, alpha=.65, label=algorithm.upper(), color=ALGORITHM_COLORS[algorithm])
            degenerate = selected.loc[selected.n_clusters_excluding_noise <= 1]
            ax.scatter(degenerate.noise_fraction, degenerate.n_clusters_excluding_noise, marker="x", color="#333333", s=55, label="0–1 grupos")
            ax.set(title=f"{encoder.upper()} · {DISPLAY[representation]}", xlabel="Fracción de noise", ylabel="Grupos excluyendo noise", xlim=(-.03, 1.03))
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
    fig.suptitle("Screening completo · más grupos o menos noise no implican mejor agrupamiento", fontsize=13)
    name = "02_screening_structure.png"
    _save(fig, figures/name)
    figure_names.append(name)
    fig, axes = plt.subplots(1, len(encoders), figsize=(7*len(encoders), 6), squeeze=False)
    for ax, encoder in zip(axes.ravel(), encoders, strict=True):
        for representation, marker in zip(REPRESENTATIONS, ("o", "^", "s"), strict=True):
            selected = screen.loc[(screen.encoder == encoder) & (screen.representation == representation) & screen.silhouette_original_space.notna()]
            for algorithm, group in selected.groupby("algorithm", sort=True):
                ax.scatter(group.silhouette_original_space, group.weighted_mean_intra_cluster_similarity, marker=marker, color=ALGORITHM_COLORS[algorithm], s=32, alpha=.65, label=f"{DISPLAY[representation]} · {algorithm}")
            short = selected.loc[selected.shortlist_stage_a]
            ax.scatter(short.silhouette_original_space, short.weighted_mean_intra_cluster_similarity, s=85, marker=marker, facecolors="none", edgecolors="black", linewidths=1)
        ax.set(title=encoder.upper(), xlabel="Silhouette en L2 original, sin noise", ylabel="Coseno intracluster, ponderado por pares")
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    fig.suptitle("Cohesión en el espacio original · contorno negro: shortlist de Fase A", fontsize=13)
    name = "03_original_space_cohesion.png"
    _save(fig, figures/name)
    figure_names.append(name)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    categories = [(e, r) for e in encoders for r in REPRESENTATIONS]
    for i, kind in enumerate(("parameters", "seeds")):
        for j, metric in enumerate(("ari", "ami")):
            ax = axes[i, j]
            for policy, shift, color in (("all_points", -.12, "#267997"), ("common_clustered", .12, "#ad6340")):
                for position, (encoder, representation) in enumerate(categories):
                    values = agreement.loc[(agreement.kind == kind) & (agreement.encoder == encoder) & (agreement.representation == representation), f"{policy}_{metric}"].dropna()
                    if len(values):
                        ax.vlines(position+shift, values.min(), values.max(), color=color, alpha=.45, lw=2)
                        ax.scatter(position+shift, values.mean(), color=color, s=32, label=policy if position == (1 if kind == "seeds" else 0) else None)
            ax.set_xticks(range(len(categories)), [f"{e.upper()}\n{DISPLAY[r]}" for e, r in categories], rotation=20, ha="right", fontsize=8)
            lower = min(-.05, float(agreement[[f"all_points_{metric}", f"common_clustered_{metric}"]].min().min())-.03)
            ax.set(title=f"{kind} · {metric.upper()}", ylabel="Acuerdo ajustado: media y rango observado", ylim=(lower, 1.04))
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("Estabilidad de la shortlist · noise como categoría y common-clustered separados", fontsize=13)
    name = "04_assignment_stability.png"
    _save(fig, figures/name)
    figure_names.append(name)
    with zipfile.ZipFile(images_archive) as archive:
        for row in references.itertuples(index=False):
            row = pd.Series(row._asdict())
            family = families[row.encoder]
            directory = locations[row.clustering_space_id]
            run_meta = read_json(directory/"metadata.json")
            space = family.spaces[(row.representation, None if row.representation == "original_l2" else 0)]
            if run_meta["input_signatures"] != space.signatures:
                raise ValueError("Displayed reference does not match the selected source coordinates")
            labels = np.load(directory/"cluster_labels.npy", allow_pickle=False)
            view = family.spaces[("pacmap", 0)] if row.representation == "original_l2" else space
            fig, ax = plt.subplots(figsize=(10, 7))
            noise = labels == -1
            ax.scatter(*view.values[noise].T, s=9, color="#b4b9bc", marker="x", linewidths=.6, label=f"Noise: {int(noise.sum())}")
            if (~noise).any():
                count = len(set(labels)-{-1})
                points = ax.scatter(*view.values[~noise].T, c=labels[~noise], cmap=plt.get_cmap("turbo", max(2, count)), s=9, alpha=.85)
                fig.colorbar(points, ax=ax, label="cluster_id local (sin orden semántico)", ticks=np.unique(np.linspace(0, count-1, min(8, count), dtype=int)))
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xlabel="Coordenada de vista 1", ylabel="Coordenada de vista 2",
                   title=f"{row.reference} · {row.encoder.upper()} · {row.algorithm.upper()}\nClustering: {DISPLAY[row.representation]} · vista: {DISPLAY[view.representation]}, semilla 0")
            ax.legend(fontsize=9)
            name = f"{row.reference}_{row.encoder}_{row.representation}_clusters.png"
            _save(fig, figures/name)
            figure_names.append(name)
            name = f"{row.reference}_{row.encoder}_{row.representation}_exemplars.png"
            selections.extend(_gallery(family, directory, row, archive, figures/name, row.reference))
            figure_names.append(name)
            summary = pd.read_parquet(directory/"cluster_summary.parquet")
            for memberships, group in summary.groupby("historical_split_memberships"):
                members = json.loads(memberships)
                category = "+".join(members) if members else "unknown"
                historical.append({"reference": row.reference, "encoder": row.encoder, "representation": row.representation,
                                   "algorithm": row.algorithm, "memberships": category, "clusters": len(group)})
    historical_table = pd.DataFrame(historical, columns=["reference", "encoder", "representation", "algorithm", "memberships", "clusters"])
    fig, ax = plt.subplots(figsize=(12, 5))
    if len(historical_table):
        pivot = historical_table.pivot_table(index="reference", columns="memberships", values="clusters", aggfunc="sum", fill_value=0)
        pivot.plot.bar(stacked=True, ax=ax, colormap="tab20")
    ax.set(title="Pertenencias históricas de clusters · múltiples splits no son una penalización", ylabel="Número de clusters", xlabel="Referencia de revisión")
    ax.legend(title="Unión de pertenencias", fontsize=8)
    name = "05_historical_split_interpretation.png"
    _save(fig, figures/name)
    figure_names.insert(4, name)
    for name, table in (("screening", screen), ("all_runs", all_runs), ("evaluated_shortlist", evaluated),
                         ("candidates", read_table(comparison/"candidates.csv")), ("references", references), ("historical_memberships", historical_table)):
        table.to_csv(tables/f"{name}.csv", index=False)
    agreement.to_parquet(tables/"assignment_comparisons.parquet", index=False)
    pd.DataFrame(selections).to_parquet(tables/"exemplar_selections.parquet", index=False)
    overview = all_runs.groupby(["encoder", "representation", "algorithm"]).agg(
        runs=("clustering_space_id", "size"), clusters_min=("n_clusters_excluding_noise", "min"), clusters_max=("n_clusters_excluding_noise", "max"),
        noise_min=("noise_fraction", "min"), noise_max=("noise_fraction", "max"), degenerate=("single_cluster", "sum"),
        fit_seconds=("fit_seconds", "sum")).reset_index()
    overview.to_csv(tables/"overview.csv", index=False)
    write_json(output/"summary.json", read_json(comparison/"summary.json"))
    metadata = {**clustering_provenance(), "artifact_kind": "clustering_review", "comparison_id": meta["collection_id"],
                "comparison_metadata_sha256": file_sha256(comparison/"metadata.json"), "figure_names": figure_names,
                "figure_count": len(figure_names), "reference_count": len(references), "exemplar_image_count": len(selections),
                "source_images_read_only": True, "sampling_seed": 0,
                "output_sha256": {p.relative_to(output).as_posix(): file_sha256(p) for p in [*(figures/n for n in figure_names), *tables.glob("*"), output/"summary.json"]}}
    write_json(output/"report_metadata.json", metadata)
    return metadata
