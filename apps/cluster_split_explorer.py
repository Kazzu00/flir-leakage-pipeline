"""Local read-only inspection. Launch from the repository root with Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from flir_pipeline.cli import _default_root
from flir_pipeline.explorer.data import (
    cluster_members,
    compare_partitions,
    filter_split,
    load_cluster,
    load_split,
    with_split,
)
from flir_pipeline.explorer.discovery import candidate_labels, discover_runs, sha256
from flir_pipeline.explorer.frames import FrameReader
from flir_pipeline.explorer.preview import PLAYBACK_NOTICE, cache_key, render_preview
from flir_pipeline.explorer.timeline import (
    comparison_figure,
    index_gaps,
    ordered_contents,
    preview_plans,
    timeline_figure,
)

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="FLIR · Cluster & Split Explorer", page_icon="🎞️", layout="wide")


@st.cache_data(ttl=30, max_entries=4, show_spinner="Descubriendo experimentos locales…")
def catalog(cluster_root, split_root, aliases_path):
    alias_issues = []
    try:
        aliases = candidate_labels(Path(aliases_path))
    except (OSError, ValueError, KeyError):
        aliases = {}
        alias_issues.append("Tabla opcional de alias inválida; se muestran las identidades de los runs.")
    clusters, c_issues = discover_runs(Path(cluster_root), aliases)
    splits, s_issues = discover_runs(Path(split_root), aliases)
    return clusters, splits, alias_issues + c_issues + s_issues


def run_stamp(run, names):
    return tuple((name, sha256(run.directory / name)) for name in names)


@st.cache_data(max_entries=6, show_spinner=False)
def cached_cluster(run, manifest, stamp):
    return load_cluster(run, manifest)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_split(run, manifest, stamp):
    return load_split(run, manifest)


def open_split(run, manifest):
    return cached_split(run, manifest, run_stamp(run, ("record_split_assignments.parquet", "source_groups.parquet", "quality.json")))


@st.cache_data(max_entries=8, show_spinner=False)
def gif_bytes(plan, rows, data_root, source_stamp, key):
    # source_stamp and key invalidate display-only cache, without persistent files.
    with FrameReader(Path(data_root), rows) as reader:
        return render_preview(plan, reader)


def image_gallery(contents, records, manifest, data_root, key):
    ordered = ordered_contents(contents)
    if ordered.empty:
        return
    pages = (len(ordered) + 11) // 12
    page = st.number_input("Página de galería · 12 contenidos", min_value=1, max_value=pages, value=1, key=f"page-{key}")
    subset = ordered.iloc[(page - 1) * 12:page * 12]
    columns = st.columns(4)
    if data_root is None:
        st.info("Configura FLIR_DATA_ROOT en .env para cargar imágenes de los ZIP originales.")
    else:
        with FrameReader(data_root, manifest) as reader:
            for position, row in enumerate(subset.itertuples()):
                with columns[position % 4]:
                    index = str(row.frame_index) if row.frame_index_valid else "sin orden inferido"
                    memberships = " / ".join(row.new_splits) or "sin partición"
                    try:
                        st.image(reader.image(row.representative_frame_id, width=320), width="stretch")
                    except (OSError, ValueError, KeyError):
                        st.warning("Imagen no disponible o hash diferente del manifest.")
                    st.caption(f"{row.sequence_id or 'Secuencia ambigua'} · índice {index} · {memberships}")
    with st.expander("Show technical metadata", expanded=False):
        fields = ["frame_id", "content_id", "possible_sequence", "possible_frame_index", "cluster_id", "original_split", "new_split"]
        st.dataframe(records.loc[records.content_id.isin(subset.content_id), fields], hide_index=True, width="stretch")


def show_sequences(members, records, manifest, data_root, clustering_id, cluster_id, context):
    st.subheader("Timeline · índices inferidos")
    st.pyplot(timeline_figure(members), width="stretch")
    unknown = members.loc[~(members.sequence_provenance_valid & members.frame_index_valid)]
    controls = st.columns(3)
    fps = controls[0].select_slider("Playback FPS", options=[2, 4, 8, 12], value=4)
    threshold = controls[1].number_input("Gap para segmentar playback · 0 desactiva", min_value=0, value=25)
    max_frames = controls[2].selectbox("Máximo de frames por preview", [60, 120, 240], index=1)
    st.caption(PLAYBACK_NOTICE)
    st.caption("Cada preview contiene un contenido exacto por posición. Los segmentos son sólo de playback; no son nuevos clústeres.")
    gaps = index_gaps(members)
    large = gaps.loc[gaps.index_gap.gt(threshold if threshold else 25)]
    if not large.empty:
        st.info(f"{len(large)} discontinuidades con Δ > {threshold if threshold else 25}. No implican tiempo físico ni segmentos continuos.")
        st.dataframe(large[["sequence_id", "previous_index", "frame_index", "index_gap"]], hide_index=True)
    plans = preview_plans(members, clustering_id, cluster_id, fps=fps,
                          gap_threshold=int(threshold), max_frames=max_frames)
    for sequence_key, group in ordered_contents(members.loc[members.sequence_provenance_valid]).groupby("sequence_key", sort=True):
        sequence_name = f"{group.source_archive.iloc[0]} / {group.sequence_id.iloc[0]}"
        sequence_plans = [p for p in plans if p.sequence_key == sequence_key]
        widget_key = f"{context}-{clustering_id}-{cluster_id}-{sequence_key}"
        with st.expander(f"Secuencia {sequence_name} · {len(group)} contenidos", expanded=True):
            st.subheader("Secuencia visual reconstruida")
            st.caption("Orden inferido de nombres. Pueden faltar frames; no es una reconstrucción del video original.")
            if sequence_plans:
                selected = st.selectbox("Segmento de playback", range(len(sequence_plans)),
                                        format_func=lambda i, ps=sequence_plans: f"{i + 1} · índices {ps[i].indices[0]}–{ps[i].indices[-1]} · {len(ps[i].frame_ids)} frames",
                                        key=f"segment-{widget_key}")
                plan = sequence_plans[selected]
                token = cache_key(plan)
                if st.button("Generar / reproducir preview", key=f"play-{widget_key}"):
                    st.session_state[f"playing-{widget_key}"] = token
                if st.session_state.get(f"playing-{widget_key}") == token and data_root is not None:
                    rows = manifest.loc[manifest.frame_id.isin(plan.frame_ids)]
                    try:
                        stamp = tuple((name, (data_root / name).stat().st_mtime_ns, (data_root / name).stat().st_size) for name in sorted(rows.source_archive.unique()))
                        with st.spinner("Leyendo ZIP en memoria…"):
                            gif = gif_bytes(plan, rows, str(data_root), stamp, token)
                        st.image(gif, caption=f"Reconstructed frame sequence · {sequence_name} · {fps} FPS de visualización", width=640)
                    except (OSError, ValueError, KeyError):
                        st.error("No se pudo producir el preview. Comprueba FLIR_DATA_ROOT y los hashes de las imágenes.")
            else:
                st.info("Sin consenso de índice: galería disponible, playback no ordenable.")
            st.markdown("**Galería de contenidos**")
            image_gallery(group, records, manifest, data_root, widget_key)
    if not unknown.empty:
        # Unknown sequence contents remain visible without a fabricated sequence.
        ambiguous = unknown.loc[~unknown.sequence_provenance_valid]
        st.warning(f"{len(unknown)} contenidos con secuencia/índice desconocido o conflictivo; excluidos del playback.")
        if not ambiguous.empty:
            image_gallery(ambiguous, records, manifest, data_root, f"unknown-{context}")


def inspect_members(members, cluster, records, manifest, data_root, split, context):
    if members.empty:
        st.info("No hay contenidos para esta selección.")
        return
    cluster_id = int(members.cluster_id.iloc[0])
    occurrences = records.loc[records.content_id.isin(members.content_id)]
    metrics = st.columns(4)
    metrics[0].metric("Contenidos únicos", len(members))
    metrics[1].metric("Ocurrencias históricas", len(occurrences))
    metrics[2].metric("Secuencias conocidas", members.loc[members.sequence_provenance_valid, "sequence_key"].nunique())
    metrics[3].metric("Membership", "Noise · −1" if cluster_id == -1 else f"Cluster {cluster_id}")
    hist = occurrences.original_split.value_counts()
    new = occurrences.new_split.value_counts()
    st.caption("Histórico · " + " · ".join(f"{s}: {hist.get(s, 0)}" for s in ("train", "val", "test")))
    st.caption("Partición seleccionada · " + " · ".join(f"{s}: {new.get(s, 0)}" for s in ("train", "val", "test")))
    ranges = members.loc[members.frame_index_valid].groupby("sequence_key").agg(sequence=("sequence_id", "first"), index_min=("frame_index", "min"), index_max=("frame_index", "max"), contents=("content_id", "size"))
    st.dataframe(ranges.reset_index(drop=True), hide_index=True)
    saved = cluster.summary.loc[cluster.summary.cluster_id.eq(cluster_id)]
    if not saved.empty:
        with st.expander("Métricas registradas del clúster completo"):
            st.dataframe(saved[[c for c in ("mean_intra_cosine", "median_intra_cosine", "dominant_sequence_fraction", "n_members") if c in saved]], hide_index=True)
            medoid = saved.medoid_content_id.iloc[0]
            st.caption("Medoid registrado · " + ("incluido en esta vista" if medoid in set(members.content_id) else "fuera del filtro de esta vista"))
            with st.expander("Medoid · technical metadata"):
                st.code(medoid)
    if split and "group_id" in members:
        with st.expander("Grupos y asignaciones existentes"):
            st.dataframe(members[["group_id", "group_type", "cluster_id", "new_splits"]], hide_index=True)
    show_sequences(members, occurrences, manifest, data_root, cluster.run.space_id, cluster_id, context)


def main():
    st.title("FLIR · Cluster & Split Explorer")
    st.caption("Inspección visual local · sólo lectura · secuencias e índices inferidos")
    st.caption(PLAYBACK_NOTICE)
    with st.sidebar:
        st.header("Experimentos existentes")
        with st.expander("Fuentes locales"):
            manifest_path = Path(st.text_input("Manifest", str(ROOT / "data/manifests/flir_canonical_candidate_v1.parquet")))
            cluster_root = st.text_input("Clustering artifacts", str(ROOT / "artifacts/clustering"))
            split_root = st.text_input("Split artifacts", str(ROOT / "artifacts/splitting"))
        if st.button("Actualizar discovery"):
            catalog.clear()
        clusters, splits, issues = catalog(cluster_root, split_root, str(ROOT / "reports/splitting/tables/clustering_candidates.csv"))
        st.caption(f"{len(clusters)} clustering runs · {len(splits)} split runs")
        if issues:
            with st.expander(f"{len(issues)} avisos de discovery"):
                st.write(issues)
    if not manifest_path.is_file():
        st.info("Selecciona un manifest local existente.")
        return
    manifest = pd.read_parquet(manifest_path)
    from flir_pipeline.data.identity import dataset_id_from_manifest

    dataset_id = dataset_id_from_manifest(manifest)
    clusters = [r for r in clusters if r.dataset_id == dataset_id]
    splits = sorted([r for r in splits if r.dataset_id == dataset_id], key=lambda r: ({"historical": 0, "random_content": 1, "cluster_aware": 2}[r.strategy], r.candidate, r.seed, r.space_id))
    if not clusters:
        st.info("No hay clustering runs completos para este manifest.")
        return
    with st.sidebar:
        split_run = st.selectbox("Split experiment", [None, *splits], index=1 if splits else 0,
                                 format_func=lambda r: r.label if r else "Sin overlay de split")
        available = clusters
        if split_run and split_run.strategy == "cluster_aware":
            available = [r for r in clusters if r.space_id == split_run.clustering_space_id]
            if not available:
                st.warning("El clustering fuente de esta partición no está disponible.")
                return
        encoder = st.selectbox("Encoder", sorted({r.encoder for r in available}))
        available = [r for r in available if r.encoder == encoder]
        representation = st.selectbox("Representation", sorted({r.representation for r in available}))
        available = [r for r in available if r.representation == representation]
        algorithm = st.selectbox("Algorithm", sorted({r.algorithm for r in available}))
        available = [r for r in available if r.algorithm == algorithm]
        selected = st.selectbox("Clustering experiment", sorted(available, key=lambda r: (not bool(r.candidate), r.label)), format_func=lambda r: r.label)
        view = st.radio("Vista", ["Cluster browser", "Browse by split", "Compare partitions", "Noise / singleton browser"])
    cluster = cached_cluster(selected, manifest, run_stamp(selected, ("content_index.parquet", "cluster_labels.npy", "cluster_summary.parquet", "metrics.json", "quality.json")))
    split = open_split(split_run, manifest) if split_run else None
    contents, records = with_split(cluster, split)
    overview = st.columns(2)
    overview[0].markdown(f"**{selected.encoder.upper()} · {selected.representation} · {selected.algorithm.upper()}**")
    overview[0].caption(f"{cluster.metrics['n_clusters_excluding_noise']} clústeres · noise {cluster.metrics['noise_fraction']:.1%}")
    overview[1].markdown(f"**{split_run.candidate or split_run.strategy if split_run else 'Sin partición seleccionada'}**")
    if split:
        counts = split.records.new_split.value_counts()
        overview[1].caption(f"{split.run.space_id} · registros train / validation / test: {counts.get('train', 0)} / {counts.get('val', 0)} / {counts.get('test', 0)}")
    st.divider()
    data_root = _default_root()
    if view == "Compare partitions":
        if len(splits) < 2:
            st.info("Se necesitan al menos dos particiones existentes.")
            return
        columns = st.columns(2)
        left = columns[0].selectbox("Partición izquierda", splits, format_func=lambda r: r.label)
        right = columns[1].selectbox("Partición derecha", splits, index=splits.index(split_run) if split_run and split_run != left else 1, format_func=lambda r: r.label)
        comparison = compare_partitions(cluster, open_split(left, manifest), open_split(right, manifest))
        comparison["sequence"] = [json.dumps([a, s]) if a and s else "Sin procedencia" for a, s in zip(comparison.source_archive, comparison.possible_sequence, strict=True)]
        sequence = st.selectbox("Sequence", sorted(comparison.sequence.unique()), format_func=lambda s: " / ".join(json.loads(s)) if s != "Sin procedencia" else s)
        rows = comparison.loc[comparison.sequence.eq(sequence)]
        st.pyplot(comparison_figure(rows, left.candidate or left.strategy, right.candidate or right.strategy))
        st.caption("Cada punto corresponde a una ocurrencia; las copias históricas conservan sus memberships. Las subfilas distinguen train/val/test. Cluster −1 permanece noise.")
        st.dataframe(rows[["possible_frame_index", "original_split", "left", "right", "left_cluster", "right_cluster"]].sort_values("possible_frame_index"), hide_index=True)
        with st.expander("Show technical metadata"):
            st.dataframe(rows, hide_index=True)
        return
    if view == "Noise / singleton browser":
        noise = contents.loc[contents.cluster_id.eq(-1)]
        if noise.empty:
            st.info("Este clustering no tiene noise.")
            return
        ordered = ordered_contents(noise)
        chosen = st.selectbox("Noise content / singleton", range(len(ordered)),
                              format_func=lambda i: f"{i + 1} · {ordered.iloc[i].sequence_id or 'sin secuencia'} · índice {ordered.iloc[i].frame_index} · {' / '.join(ordered.iloc[i].new_splits)}")
        members = ordered.iloc[[chosen]]
    elif view == "Browse by split":
        if split is None:
            st.info("Selecciona una partición existente en el panel lateral.")
            return
        name = st.selectbox("Split", ["train", "val", "test"], format_func=lambda s: {"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}[s])
        filtered = filter_split(contents, name)
        if filtered.empty:
            st.info("No hay contenidos asignados a esta partición.")
            return
        st.caption("Se muestran los contenidos con alguna ocurrencia en el split. En historical/random un clúster puede aparecer en varias particiones.")
        groups = filtered.groupby("cluster_id").size()
        selected_id = st.selectbox("Cluster / noise presente en el split", groups.index.tolist(), format_func=lambda i: f"{'Noise −1' if i == -1 else 'Cluster ' + str(i)} · {groups[i]} contenidos")
        members = cluster_members(filtered, selected_id)
    else:
        counts = contents.groupby("cluster_id").size()
        ids = sorted(counts.index, key=lambda i: (i == -1, i))
        selected_id = st.selectbox("cluster_id", ids, format_func=lambda i: f"{'Noise −1' if i == -1 else 'Cluster ' + str(i)} · {counts[i]} contenidos")
        members = cluster_members(contents, selected_id)
    inspect_members(members, cluster, records, manifest, data_root, split, view)


try:
    main()
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"La selección no pudo cargarse ({type(exc).__name__}). Comprueba el manifest, las fuentes y la integridad de los artifacts.")
    with st.expander("Diagnóstico local"):
        st.code(str(exc))
