"""Local run-level inspection; never invokes training or recomputes similarity."""

from pathlib import Path

import altair as alt
import streamlit as st

from flir_pipeline.detection.association import (
    COLORS,
    METRIC_LABELS,
    NOTICE,
    POPULATIONS,
    RESIDUAL_LABELS,
    STRATEGIES,
    STRATEGY_LABELS,
    load_evidence,
    load_registry,
    prespecified_table,
    select_view,
    strategy_summaries,
    view_label,
)

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="FLIR · Detector & Residual Similarity Explorer", page_icon="📊", layout="wide")
st.title("FLIR · Detector & Residual Similarity Explorer")
st.caption("Local, read-only · detector runs and frozen residual context")
st.info(NOTICE)

with st.sidebar:
    st.header("Local evidence")
    plan_path = st.text_input("Frozen plan directory", "artifacts/detection/protocol")
    artifact_path = st.text_input("Detector artifact directory", "artifacts/detection")
    st.caption("No remote machine connection. Refresh reads local artifacts only.")
    refresh = st.button("Refresh evidence")


@st.cache_data(ttl=30, max_entries=2, show_spinner="Verifying local evidence…")
def read_evidence(plan, artifacts):
    return load_evidence(Path(plan), Path(artifacts), split_root=Path(artifacts).parent/"splitting/runs")


def color_state(value):
    color = {"PENDING": "#eef1f5", "RUNNING": "#fff2cb", "COMPLETE": "#d7eee8", "INVALID": "#f9d9d9"}.get(value, "white")
    return f"background-color: {color}; color: #172531"


def association_view(evidence, registry):
    if not evidence.complete:
        st.subheader("Detector ↔ residual similarity")
        st.warning("PENDING COMPUTE" if evidence.state == "PENDING" else "PARTIAL · scientific association withheld")
        st.write("The complete controlled matrix is required. Small-pilot metrics are excluded.")
        st.dataframe(prespecified_table(evidence, registry), hide_index=True, width="stretch")
        return
    controls = st.columns(4)
    metric = controls[0].selectbox("Detector metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get)
    population = controls[1].selectbox("Population", list(POPULATIONS), format_func=POPULATIONS.get)
    residuals = ["dinov2_nn_mean", "clip_nn_mean", "dinov2_top001_pairs", "clip_top001_pairs", "temporal_at5", "class_deviation_pp"]
    residual = controls[2].selectbox("Residual metric", residuals, format_func=RESIDUAL_LABELS.get)
    strategy = controls[3].selectbox("Strategy", [None, *STRATEGIES], format_func=lambda v: "All" if v is None else STRATEGY_LABELS[v])
    st.subheader(view_label(registry, metric, residual, population))
    selected = select_view(evidence, metric, residual, population, strategy)
    plotted = selected.dropna(subset=["x", "y"]).copy()
    plotted["strategy_label"] = plotted.strategy.map(STRATEGY_LABELS)
    st.caption(f"{len(plotted)} individual runs; {len(selected)-len(plotted)} undefined pairs omitted. No seed averaging or fitted trend.")
    if plotted.empty:
        st.info("No defined detector/residual pairs for this view.")
    else:
        chart = alt.Chart(plotted).mark_point(filled=True, size=95, opacity=.8).encode(
            x=alt.X("x:Q", title=RESIDUAL_LABELS[residual], scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title=METRIC_LABELS[metric], scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("strategy_label:N", title="Strategy", scale=alt.Scale(domain=list(STRATEGY_LABELS.values()), range=list(COLORS))),
            shape=alt.Shape("strategy_label:N", title="Strategy", scale=alt.Scale(domain=list(STRATEGY_LABELS.values()), range=["circle", "square", "triangle-up", "diamond"])),
            tooltip=[alt.Tooltip("strategy_label:N", title="Strategy"), "split_seed:O", "detector_seed:O",
                     "split_space_id:N", "detector_run_id:N", alt.Tooltip("x:Q", title=RESIDUAL_LABELS[residual], format=".6g"),
                     alt.Tooltip("y:Q", title=METRIC_LABELS[metric], format=".6g")],
        ).properties(height=420).interactive()
        st.altair_chart(chart, width="stretch")
    st.caption("Coincident points may overlap; the table retains every run and both seeds.")
    columns = ["strategy", "split_seed", "detector_seed", "split_space_id", "detector_run_id", "class_name", "x", "y"]
    st.dataframe(selected[columns], hide_index=True, width="stretch")


@st.fragment(run_every=30)
def dashboard():
    try:
        evidence = read_evidence(str((ROOT/plan_path).resolve()), str((ROOT/artifact_path).resolve()))
        registry = load_registry(ROOT/"configs/detection/associations.yaml")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Evidence unavailable or invalid: {exc}")
        st.caption("No detector association is published when inputs cannot be verified.")
        return
    st.subheader("Controlled detector experiment")
    counts = st.columns(3)
    counts[0].metric("Completed", f"{evidence.fairness['completed_runs']} / {evidence.fairness['expected_runs']}")
    counts[1].metric("Evidence state", evidence.state)
    counts[2].metric("Scientific association rows", len(evidence.associations))
    st.write(" · ".join(STRATEGY_LABELS.values()))
    st.caption("Local evidence refreshes every 30 seconds. COMPLETE means the whole controlled matrix passes verification.")
    if evidence.issues:
        with st.expander("Validation issues", expanded=True):
            for issue in evidence.issues:
                st.error(issue)
    associations, context_tab, matrix_tab, summary_tab = st.tabs(["Associations", "Frozen residual context", "Experiment matrix", "Strategy summary"])
    with associations:
        association_view(evidence, registry)
    with context_tab:
        st.caption("Existing splitting metrics only. Top-0.1% counts are pairs, temporal Δ≤5 is an inferred-index fraction, class deviation is in pp.")
        st.dataframe(evidence.context.assign(strategy=evidence.context.strategy.map(STRATEGY_LABELS)), hide_index=True, width="stretch")
    with matrix_tab:
        st.subheader("Experiment matrix")
        matrix = evidence.matrix.copy()
        matrix["partition"] = matrix.strategy.map(STRATEGY_LABELS) + " · split seed " + matrix.split_seed.astype(str)
        order = matrix.assign(order=matrix.strategy.map({s: i for i, s in enumerate(STRATEGIES)})).sort_values(["order", "split_seed"]).partition.drop_duplicates()
        pivot = matrix.pivot(index="partition", columns="detector_seed", values="status").reindex(order)
        st.dataframe(pivot.style.map(color_state), width="stretch", height=610)
        st.caption("RUNNING is inferred from INITIALIZED/TRAINING/TRAINED artifacts. It does not prove a process is currently alive.")
        with st.expander("Cell identities and validation details"):
            st.dataframe(matrix.drop(columns="partition"), hide_index=True, width="stretch")
    with summary_tab:
        summaries = strategy_summaries(evidence)
        if not summaries:
            st.info("Strategy summaries require COMPLETE_CONTROLLED_COMPARISON. No winner or ranking is produced.")
        else:
            st.caption("Fixed strategy order. Mean/median/sample SD describe individual runs; residual means use unique frozen splits.")
            st.dataframe(summaries["strategy_summary"], hide_index=True, width="stretch")
            st.write("Detector-seed variation within each split")
            st.dataframe(summaries["training_seed_summary"].query("class_id == -1 and metric == 'map50_95'"), hide_index=True, width="stretch")
            st.write("Split-seed variation of detector-seed means")
            st.dataframe(summaries["split_seed_summary"].query("class_id == -1 and metric == 'map50_95'"), hide_index=True, width="stretch")
    with st.expander("Infrastructure pilots"):
        st.warning("NOT SCIENTIFIC COMPARISON")
        st.write(f"{evidence.excluded_pilots} small-pilot artifacts excluded. No pilot performance is shown as scientific evidence.")


if refresh:
    read_evidence.clear()
dashboard()
