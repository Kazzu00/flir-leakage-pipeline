"""Exact Euclidean matrices with an explicit, bounded retention lifetime.

These are operational caches, not new representations or persisted artifacts.
The float64 pdist/squareform path is deliberately identical to the eager path:
changing kernels or reducing precision could change epsilon ties and medoids.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass(eq=False)
class EuclideanDistances:
    """Keep only vectors outside a scope; allocate on first matrix request.

    Nested scopes reuse one matrix and the outermost exit drops the cache even
    on exceptions. Callers must not retain yielded arrays outside their scope.
    Like clustering publication, this cache supports one sequential worker;
    it is not a shared concurrent cache. Source vectors must remain unchanged.
    """

    values: np.ndarray
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _retainers: int = field(default=0, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self)

    @property
    def is_materialized(self) -> bool:
        return self._matrix is not None

    @contextmanager
    def retain(self):
        """Permit reuse within this scope without allocating anything on entry."""
        self._retainers += 1
        try:
            yield
        finally:
            self._retainers -= 1
            if self._retainers == 0:
                self._matrix = None

    @contextmanager
    def materialize(self):
        with self.retain():
            if self._matrix is None:
                self._matrix = squareform(pdist(self.values.astype(np.float64), metric="euclidean"))
            yield self._matrix


Distances: TypeAlias = np.ndarray | EuclideanDistances


@contextmanager
def distance_matrix(source: Distances):
    """Also accept existing eager callers without copying their arrays."""
    if isinstance(source, EuclideanDistances):
        with source.materialize() as matrix:
            yield matrix
    else:
        yield source


@contextmanager
def retain_distances(*sources: Distances):
    """Reuse only explicitly requested spaces; release on success or failure."""
    with ExitStack() as stack:
        seen = set()
        for source in sources:
            if isinstance(source, EuclideanDistances) and id(source) not in seen:
                stack.enter_context(source.retain())
                seen.add(id(source))
        yield
