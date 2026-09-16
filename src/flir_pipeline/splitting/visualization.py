"""Nine bounded figures and safe aggregate tables for the Spanish progress review."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flir_pipeline.similarity.storage import file_sha256, read_json, write_json
from flir_pipeline.splitting.experiments import verify_comparison

FIGURES = (
    "01_split_sizes_comparison.png", "02_class_balance_comparison.png", "03_exact_duplicate_overlap.png",
    "04_cross_split_nn_similarity_dinov2.png", "05_cross_split_nn_similarity_clip.png",
    "06_high_similarity_cross_split_pairs.png", "07_temporal_cross_split_rates.png",
    "08_cluster_fracture_comparison.png", "09_split_candidate_pareto.png",
)


def generate_report(comparison: Path, output: Path) -> None:
    if not verify_comparison(comparison)["quality_valid"]:
        raise ValueError("Cannot report an unverified split comparison")
    meta = read_json(comparison/"metadata.json")
    root = comparison.parents[1]
    directories = {Path(r["path"]).name: root/r["path"] for r in meta["runs"]}
    runs = pd.read_csv(comparison/"runs.csv")
    candidates = pd.read_csv(comparison/"clustering_candidates.csv")
    order = ["historical", "random_content", *candidates.candidate_label.tolist()]
    labels = ["Historical", "Random", *candidates.candidate_label.tolist()]
    figures, tables = output/"figures", output/"tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for path in comparison.glob("*.csv"):
        (tables/path.name).write_bytes(path.read_bytes())
    summary = read_json(comparison/"summary.json")
    write_json(output/"summary.json", summary)
    class_rows, size_rows, quantile_rows, temporal_rows, sequence_rows = [], [], [], [], []
    for row in runs.itertuples():
        directory = directories[row.split_space_id]
        prefix = {"candidate_label": row.candidate_label, "strategy": row.strategy, "seed": row.seed, "split_space_id": row.split_space_id}
        class_rows.append(pd.read_parquet(directory/"class_balance.parquet").assign(**prefix))
        current = read_json(directory/"split_summary.json")
        sequence_rows.append({**prefix, "sequences": current["temporal"]["sequence_count"],
                              **{f"fraction_{k}_splits": v for k,v in current["temporal"]["sequence_split_fractions"].items()}})
        size_rows.append(pd.DataFrame(current["balance"]["sizes"]).assign(**prefix))
        for encoder in ("dinov2", "clip"):
            quantile_rows.append(pd.read_parquet(directory/f"quantile_pairs_{encoder}.parquet").assign(encoder=encoder, **prefix))
        temporal_rows.append(pd.read_parquet(directory/"temporal_cross_split.parquet").assign(**prefix))
    classes, sizes, quantiles, temporal = (pd.concat(x, ignore_index=True) for x in (class_rows, size_rows, quantile_rows, temporal_rows))
    pd.DataFrame(sequence_rows).to_csv(tables/"sequence_fragmentation.csv", index=False)
    for name, frame in (("class_balance",classes),("sizes",sizes),("quantile_pairs",quantiles),("temporal",temporal)):
        frame.to_csv(tables/f"{name}.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "figure.dpi": 110})

    def save(fig, index):
        fig.savefig(figures/FIGURES[index], dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True)
    y = np.arange(len(order))
    for ax, split, color in zip(axes, ("train","val","test"), ("#167D9A","#D48A29","#5F65AD"), strict=True):
        grouped = sizes.loc[sizes.split == split].groupby("candidate_label").records.agg(["mean","min","max"]).reindex(order)
        ax.barh(y, grouped["mean"], color=color, alpha=.8)
        ax.errorbar(grouped["mean"], y, xerr=[grouped["mean"]-grouped["min"],grouped["max"]-grouped["mean"]], fmt="none", color="black", capsize=3)
        target = sizes.loc[sizes.split == split,"target_records"].iloc[0]
        ax.axvline(target, color="black", ls="--", label=f"Target {target:.0f}")
        ax.set(title=split, xlabel="Registros: media y rango entre semillas", yticks=y, yticklabels=labels)
        ax.legend(fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("¿Se conservan los tamaños históricos al mantener grupos indivisibles?")
    fig.tight_layout()
    save(fig, 0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7), sharey=True, layout="constrained")
    class_names = ["Vehicles", "Buildings", "Roads", "Rivers", "Heavy\nMachinery"]
    color_max = max(10., float(np.ceil(classes.groupby(["candidate_label", "split", "class_id"]).absolute_percentage_point_deviation.mean().max()/10)*10))
    for ax, split in zip(axes, ("train","val","test"), strict=True):
        frame = classes.loc[classes.split == split].groupby(["candidate_label","class_id"]).absolute_percentage_point_deviation.mean().unstack().reindex(order)
        im = ax.imshow(frame, aspect="auto", cmap="YlOrRd", vmin=0, vmax=color_max)
        for (i,j), value in np.ndenumerate(frame.to_numpy()):
            ax.text(j,i,f"{value:.1f}",ha="center",va="center",fontsize=8,color="white" if value > color_max*.55 else "black")
        ax.set(title=split, xticks=range(5), xticklabels=class_names, yticks=y, yticklabels=labels)
        ax.tick_params(axis="x", labelrotation=40)
    fig.colorbar(im, ax=axes, label="Desviación absoluta de composición global (pp), media de semillas", shrink=.8)
    fig.suptitle("¿Qué clases cambian su proporción de instancias en cada split?")
    save(fig, 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = runs.groupby("strategy").exact_duplicate_cross_split_count.max().reindex(["historical","random_content","cluster_aware"])
    bars = ax.bar(["Historical", "Random content", "Cluster-aware"], values, color=["#BB5448","#167D9A","#5F65AD"])
    ax.bar_label(bars, padding=4)
    ax.set(ylabel="Contenidos exactos presentes en varios splits", title="¿Se conserva la protección de identidad exacta?\nMáximo entre todas las configuraciones/semillas")
    ax.set_ylim(0,max(1,values.max())*1.2)
    save(fig, 2)

    for encoder, index in (("dinov2",3),("clip",4)):
        arrays = []
        for label in order:
            row = runs.loc[(runs.candidate_label == label) & (runs.seed == 0)].iloc[0]
            arrays.append(pd.read_parquet(directories[row.split_space_id]/f"cross_split_nn_{encoder}.parquet").cross_split_nn_similarity.to_numpy())
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.boxplot(arrays, orientation="horizontal", tick_labels=labels, showfliers=False, whis=(5,95), patch_artist=True,
                   boxprops={"facecolor":"#C9DEE5"},medianprops={"color":"#8A302A"})
        ax.invert_yaxis()
        ax.set(xlabel="Coseno original: caja Q1–Q3, mediana, bigotes p5–p95; seed 0",
               title=f"{encoder.upper()}: ¿qué tan similar es el vecino más cercano de otro split?\nUn valor por contenido; exactos excluidos del NN")
        ax.grid(axis="x",alpha=.25)
        fig.tight_layout()
        save(fig,index)

    fig, axes = plt.subplots(1,2,figsize=(14,7),sharey=True,layout="constrained")
    for ax, encoder in zip(axes,("dinov2","clip"),strict=True):
        frame = quantiles.loc[quantiles.encoder == encoder].groupby(["candidate_label","quantile"]).cross_split_fraction.mean().unstack().reindex(order)*100
        im = ax.imshow(frame,aspect="auto",cmap="YlOrRd",vmin=0,vmax=100)
        for (i,j),value in np.ndenumerate(frame.to_numpy()):
            label = "<0.1" if 0 < value < .1 else f"{value:.1f}"
            ax.text(j,i,label,ha="center",va="center",fontsize=8,color="black" if value < 65 else "white")
        ax.set(title=encoder.upper(),xticks=range(6),xticklabels=["10%","5%","2.5%","1%","0.5%","0.1%"],yticks=y,yticklabels=labels,xlabel="Cohorte superior de similitud; thresholds propios")
    fig.colorbar(im,ax=axes,label="% de pares cross-split, media entre semillas",shrink=.8)
    fig.suptitle("¿Cuántas relaciones de alta similitud siguen cruzando los splits?")
    save(fig,5)

    fig,ax=plt.subplots(figsize=(9,7))
    frame=temporal.groupby(["candidate_label","frame_delta_max"]).fraction.mean().unstack().reindex(order)*100
    im=ax.imshow(frame,aspect="auto",cmap="YlOrRd",vmin=0,vmax=100)
    for (i,j),value in np.ndenumerate(frame.to_numpy()):
        ax.text(j,i,f"{value:.1f}%",ha="center",va="center",fontsize=9)
    ax.set(xticks=range(4),xticklabels=["Δ ≤ 1","Δ ≤ 5","Δ ≤ 10","Δ ≤ 25"],yticks=y,yticklabels=labels,
           title="¿Qué proximidad temporal inferida permanece entre splits?\nMedia entre semillas; contenidos distintos de la misma secuencia")
    fig.colorbar(im,ax=ax,label="% de pares elegibles cross-split")
    fig.tight_layout()
    save(fig,6)

    fracture=pd.read_csv(comparison/"cluster_fracture.csv")
    fig,ax=plt.subplots(figsize=(13,5))
    x=np.arange(len(candidates))
    for offset,strategy,color in ((-.25,"historical","#BB5448"),(0,"random_content","#167D9A"),(.25,"cluster_aware","#5F65AD")):
        values=fracture.loc[fracture.strategy == strategy].groupby("candidate_label").fracture_rate.mean().reindex(candidates.candidate_label)*100
        ax.bar(x+offset,values,width=.25,label=strategy,color=color)
    ax.set(xticks=x,xticklabels=candidates.candidate_label,ylabel="% clusters fracturados (media entre semillas)",
           title="¿Los grupos elegidos permanecen indivisibles? Ruido excluido del denominador")
    ax.legend(loc="upper right",fontsize=9)
    ax.set_ylim(0,115)
    fig.tight_layout()
    save(fig,7)

    pareto=pd.read_csv(comparison/"pareto.csv")
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout="constrained")
    for ax,encoder in zip(axes,("dinov2","clip"),strict=True):
        for row in pareto.itertuples():
            ax.scatter(row.max_relative_record_deviation*100,getattr(row,f"{encoder}_top001_fraction")*100,
                       marker="o" if row.eligible else "x",s=80,color="#167D9A" if row.pareto else "#A0A0A0")
        ax.axvline(10,color="#BB5448",ls="--",label="Tolerancia previa: 10%")
        ax.set_xscale("symlog",linthresh=1)
        ax.set_xlim(-.05, max(15., pareto.max_relative_record_deviation.max()*150))
        ax.set(xlabel="Peor desviación relativa de tamaño (%) · escala symlog",ylabel="Peor % de pares top 0.1% cross-split",title=encoder.upper())
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("¿Qué compromisos sobreviven a las cinco semillas?\nAzul: Pareto multicriterio; ×: excluido por constraints; vista parcial de ocho criterios")
    fig.canvas.draw()
    for ax,encoder in zip(axes,("dinov2","clip"),strict=True):
        placed = []
        for row in pareto.sort_values([f"{encoder}_top001_fraction", "candidate_label"]).itertuples():
            point = (row.max_relative_record_deviation*100, getattr(row,f"{encoder}_top001_fraction")*100)
            pixel = ax.transData.transform(point)
            offset = 5
            while any(abs(pixel[0]+6-x) < 34 and abs(pixel[1]+offset*fig.dpi/72-y) < 17 for x,y in placed):
                offset += 12
            placed.append((pixel[0]+6,pixel[1]+offset*fig.dpi/72))
            ax.annotate(row.candidate_label,point,xytext=(4,offset),textcoords="offset points",fontsize=8,
                        arrowprops={"arrowstyle":"-", "color":"#777777", "lw":.6} if offset > 5 else None)
    save(fig,8)
    write_json(output/"report_metadata.json", {"comparison_metadata_sha256":file_sha256(comparison/"metadata.json"),
               "figure_count":len(FIGURES),"output_sha256":{p.relative_to(output).as_posix():file_sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "report_metadata.json"}})
