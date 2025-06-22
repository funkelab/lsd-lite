from numpy.typing import ArrayLike, NDArray
from typing import Callable, Sequence
import numpy as np


def eq(x: NDArray, y: NDArray) -> NDArray:
    """Affinity is 1 where labels match."""
    return np.equal(x, y)


def eq_no_bg(x: NDArray, y: NDArray) -> NDArray:
    """Affinity is 1 where labels match *and* both are non-zero."""
    return np.equal(x, y) & (x > 0) & (y > 0)


DIST_FUNCS: dict[str, Callable[[NDArray, NDArray], NDArray]] = {
    "equality": eq,
    "equality-no-bg": eq_no_bg,
}


def _compute_affs_single(
    arr: NDArray,
    offset: Sequence[int],
    dist_fn: Callable[[NDArray, NDArray], NDArray],
    pad: bool,
) -> NDArray:
    offset = np.asarray(offset, dtype=int)
    ndim_spatial = offset.size

    if pad:
        lead = arr.ndim - ndim_spatial
        pad_width = [(0, 0)] * lead + [(max(0, -o), max(0, o)) for o in offset]
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)

    shp = arr.shape[-ndim_spatial:]
    lower = tuple(
        slice(max(0, -o), min(shp[i], shp[i] - o)) for i, o in enumerate(offset)
    )
    upper = tuple(
        slice(max(0, o), min(shp[i], shp[i] + o)) for i, o in enumerate(offset)
    )

    while len(lower) < arr.ndim:
        lower = (slice(None),) + lower
        upper = (slice(None),) + upper

    return dist_fn(arr[lower], arr[upper])


def get_affs(
    arr: ArrayLike,
    neighborhood: Sequence[Sequence[int]],
    dist: str | Callable[[NDArray, NDArray], NDArray] = "equality",
    *,
    pad: bool = True,
) -> NDArray:
    dist_fn = DIST_FUNCS.get(dist, dist) if isinstance(dist, str) else dist
    if not callable(dist_fn):
        raise ValueError(f"Unknown distance function: {dist}")

    aff_list = [
        _compute_affs_single(np.asarray(arr), offset, dist_fn, pad)
        for offset in neighborhood
    ]
    return np.stack(aff_list, axis=0)
