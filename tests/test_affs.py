from lsd_lite import get_affs
import numpy as np
import pytest

NEIGHBORHOODS = {
    "1d": [[1], [0]],
    "2d": [[1, 0], [0, 1]],
    "3d_nearest": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "3d_long_range": [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [3, 0, 0],
        [0, 6, 0],
        [0, 0, 6],
        [6, 0, 0],
        [0, 9, 0],
        [0, 0, 9],
    ],
}

# expected sums for toy volume
EXPECTED_TOY_COUNTS = {
    "1d": [56, 64],
    "2d": [56, 56],
    "3d_nearest": [16, 56, 56],
    "3d_long_range": [16, 56, 56, 32, 32, 32, 32, 32, 32],
}


@pytest.mark.parametrize("nb_key", NEIGHBORHOODS.keys())
@pytest.mark.parametrize("dist_func", ["equality", "equality-no-bg"])
def test_real_shapes(real_labels, nb_key, dist_func):
    neigh = NEIGHBORHOODS[nb_key]
    affs = get_affs(real_labels, neigh, dist=dist_func, pad=True)
    assert affs.shape == (len(neigh), *real_labels.shape)
    assert affs.dtype == np.bool_


@pytest.mark.parametrize("nb_key", NEIGHBORHOODS.keys())
def test_toy_counts(toy_labels, nb_key):
    neigh = NEIGHBORHOODS[nb_key]
    expected = EXPECTED_TOY_COUNTS[nb_key]

    affs = get_affs(toy_labels, neigh, dist="equality", pad=True)
    got = [int(a.sum()) for a in affs]

    assert got == expected, f"{nb_key}: expected {expected}, got {got}"


@pytest.mark.parametrize("nb_key", NEIGHBORHOODS.keys())
def test_no_background_dist_func(toy_labels, nb_key):
    bg_mask = toy_labels == 0
    affs = get_affs(toy_labels, NEIGHBORHOODS[nb_key], dist="equality-no-bg", pad=True)
    assert (affs & bg_mask).sum() == 0


@pytest.mark.parametrize("nb_key", ["1d", "2d", "3d_nearest", "3d_long_range"])
@pytest.mark.parametrize("dist_func", ["equality", "equality-no-bg"])
def test_real_affs_non_empty(real_labels, nb_key, dist_func):
    affs = get_affs(real_labels, NEIGHBORHOODS[nb_key], dist=dist_func, pad=True)
    assert affs.any(), f"{nb_key} – {dist_func} produced all-zero affinities"
