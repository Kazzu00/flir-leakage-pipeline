"""Small data contracts for existing runs and visualization-only playback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Run:
    directory: Path
    kind: str
    space_id: str
    dataset_id: str
    metadata: dict = field(compare=False, repr=False)
    encoder: str = ""
    representation: str = ""
    algorithm: str = ""
    strategy: str = ""
    clustering_space_id: str | None = None
    seed: int | None = None
    candidate: str = ""

    @property
    def label(self) -> str:
        if self.kind == "clustering_run":
            return f"{self.candidate + ' · ' if self.candidate else ''}{self.encoder} / {self.representation} / {self.algorithm} · {self.space_id}"
        return f"{self.candidate or self.strategy} · {self.strategy} · seed {self.seed} · {self.space_id}"


@dataclass
class ClusterData:
    run: Run
    contents: pd.DataFrame
    records: pd.DataFrame
    summary: pd.DataFrame
    metrics: dict


@dataclass
class SplitData:
    run: Run
    records: pd.DataFrame
    groups: pd.DataFrame


@dataclass(frozen=True)
class PreviewPlan:
    clustering_space_id: str
    cluster_id: int
    sequence_key: str
    frame_ids: tuple[str, ...]
    content_ids: tuple[str, ...]
    indices: tuple[int, ...]
    gap_threshold: int
    fps: int
    width: int = 640
