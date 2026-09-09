"""Resolve Hugging Face model provenance without an additional network request."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelRevision:
    """Requested ref and immutable commit, if exposed by the loaded HF config.

    ``effective`` is the feature-space identity. A moving branch name alone is
    never evidence of an immutable revision; unresolved runs remain explicit.
    """

    requested: str | None
    resolved: str | None
    source: str

    @property
    def effective(self) -> str:
        """Use the resolved commit, falling back to the historical ref/unknown."""
        return self.resolved or self.requested or "unknown"

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible provenance; unresolved SHA is null, not invented."""
        return {
            "model_revision": self.effective,
            "requested_model_revision": self.requested,
            "resolved_model_revision": self.resolved,
            "model_revision_source": self.source,
        }


def resolve_model_revision(
    config: Any, requested: str | None, require_resolved: bool = False
) -> ModelRevision:
    """Capture the commit attached by Transformers to the actual loaded model.

    Accept a full requested SHA when that optional config attribute is absent.
    Offline loading uses the same path and never queries the Hub separately.
    Final research runs can fail closed with ``require_resolved=True``.
    """
    def commit(value: Any) -> str | None:
        return value.lower() if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) else None

    resolved = commit(getattr(config, "_commit_hash", None))
    pinned = commit(requested)
    if resolved and pinned and resolved != pinned:
        raise ValueError("Loaded model commit differs from the requested model_revision")
    source = "model.config._commit_hash" if resolved else "requested_commit" if pinned else "unresolved"
    resolved = resolved or pinned
    if resolved is None:
        message = (
            "Cannot resolve an immutable Hugging Face model revision. Set "
            "model_revision to a full commit SHA before a final experiment."
        )
        if require_resolved:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    return ModelRevision(requested, resolved, source)
