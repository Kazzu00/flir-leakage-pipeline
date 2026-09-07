"""Validated configuration primitives for future pipeline stages."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PipelineConfig(BaseModel):
    """Portable project paths without assuming a dataset is present."""

    model_config = ConfigDict(extra="forbid")

    data_root: Path | None = None
    artifacts_root: Path = Path("artifacts")
    model_root: Path | None = None
