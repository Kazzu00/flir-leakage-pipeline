# Inspect existing clusters and partitions

Both interfaces require a local canonical manifest and completed, matching
artifacts. Set `FLIR_DATA_ROOT` to the external original ZIPs. Neither interface
fits models, recomputes metrics or creates new partitions.

| Need | Interface and guide |
|---|---|
| Detailed galleries, comparisons, timelines, gaps and reconstructed playback | [Streamlit](../visualization/streamlit.md) |
| Global collection, cluster/sequence/split filters and saved PaCMAP layouts | [VIKUS](../visualization/vikus.md) |
| Controlled detector run matrix, frozen residual context and gated associations | [Detector & Residual Similarity](../visualization/detector_similarity.md) |

Run from the repository root:

```powershell
uv run --extra explorer streamlit run apps/cluster_split_explorer.py
```

For the existing **C10 — DINOv2 / PaCMAP / DBSCAN** candidate:

```powershell
uv run flir-pipeline explorer vikus-build --candidate C10 --seed 0 --name c10-seed0-local
uv run flir-pipeline explorer vikus-serve --bundle reports/explorer/vikus/c10-seed0-local
```

Streamlit uses `http://127.0.0.1:8501`; VIKUS uses `http://127.0.0.1:8765`.
The first VIKUS build downloads only the pinned public runtime; use `--offline`
with its existing cache. Its preview bundle remains local and ignored.

If discovery omits a run, check completion, dataset/space identity, source joins
and checksums. Do not bypass invalid or `.partial` artifacts. Missing reduced
coordinates require the exact saved source reduction, not a new fit.
Sequence ordering is inferred; requested playback FPS is a display parameter,
not verified capture timing. Human inspection does not automatically validate scenes.
