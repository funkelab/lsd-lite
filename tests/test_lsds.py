import functools

import numpy as np
import pytest
from funlib.geometry import Coordinate
from scipy.ndimage import gaussian_filter

from lsd_lite import get_lsds
from lsd_lite.lsds import outer_product, upsample


def get_lsds_full_block(segmentation, sigma, voxel_size=None, labels=None, downsample=1):
    """Reference: this is the original per-label lsd computation over the whole block."""
    dims = segmentation.ndim
    if isinstance(sigma, (int, float)):
        sigma = (sigma,) * dims
    voxel_size = Coordinate((1,) * dims if voxel_size is None else voxel_size)
    if labels is None:
        labels = np.unique(segmentation)
    df = downsample
    sub_segmentation = segmentation[tuple(slice(None, None, df) for _ in range(dims))]
    sub_voxel_size = tuple(v * df for v in voxel_size)
    sub_sigma_voxel = tuple(s / v for s, v in zip(sigma, sub_voxel_size))
    grid = np.meshgrid(
        *[
            np.arange(0, sub_segmentation.shape[d] * sub_voxel_size[d], sub_voxel_size[d])
            for d in range(dims)
        ],
        indexing="ij",
    )
    coords = np.array(grid, dtype=np.float32)
    max_distance = np.array(sigma, dtype=np.float32)
    descriptors = np.zeros((10 if dims == 3 else 6,) + segmentation.shape, dtype=np.float32)
    aggregate = functools.partial(gaussian_filter, mode="constant", cval=0.0, truncate=3.0)
    rows, cols = np.triu_indices(dims)
    entries = sorted(
        (rows * dims + cols).tolist(),
        key=lambda x: x % (dims + 1) * (dims + 1) + x // (dims + 1),
    )
    for label in labels:
        if label == 0:
            continue
        sub_mask = sub_segmentation == label
        masked_coords = coords * sub_mask
        mass = aggregate(sub_mask.astype(np.float32), sigma=sub_sigma_voxel)
        mass_mask = mass == 0
        mass[mass_mask] = 1
        center_of_mass = (
            np.array([aggregate(masked_coords[d], sigma=sub_sigma_voxel) for d in range(dims)])
            / mass
        )
        mean_offset = center_of_mass - coords
        mean_offset = mean_offset / max_distance.reshape((-1,) + (1,) * dims) * 0.5 + 0.5
        coords_outer = outer_product(masked_coords)
        center_of_mass_outer = outer_product(center_of_mass)
        covariance = (
            np.array([aggregate(coords_outer[d], sub_sigma_voxel) for d in entries]) / mass
        )
        covariance -= center_of_mass_outer[entries]
        variance = np.maximum(covariance[:dims], 1e-3)
        pearson = covariance[dims:]
        for ind, entry in enumerate(entries[dims:]):
            i, j = entry // dims, entry % dims
            pearson[ind] /= 2 * np.sqrt(variance[i] * variance[j])
            pearson[ind] += 0.5
        for d in range(dims):
            variance[d] /= sigma[d] ** 2
        mass = mass * ~mass_mask
        if dims == 3:
            pearson = pearson[[0, 2, 1]]
        descriptor = np.concatenate((mean_offset, variance, pearson, mass[None, :]))
        descriptor = upsample(descriptor, df)
        descriptors += descriptor * (segmentation == label)[None, ...]
    np.clip(descriptors, 0.0, 1.0, out=descriptors)
    return descriptors


def voronoi_labels(shape, num_seeds, seed):
    """Random Voronoi cells: labels fill the block and touch its borders."""
    rng = np.random.default_rng(seed)
    seeds = np.stack([rng.integers(0, s, num_seeds) for s in shape], axis=1)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1)
    dist = ((grid[..., None, :] - seeds) ** 2).sum(-1)
    return (dist.argmin(-1) + 1).astype(np.uint64)


def speckled_labels(shape, num_blobs, seed):
    """Small isolated boxes in a sea of background, plus one spanning slab."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=np.uint64)
    for i in range(num_blobs):
        size = rng.integers(1, 6, len(shape))
        start = [rng.integers(0, s - z + 1) for s, z in zip(shape, size)]
        labels[tuple(slice(a, a + z) for a, z in zip(start, size))] = i + 1
    labels[-3:] = num_blobs + 1
    return labels


@pytest.mark.parametrize("downsample", [1, 2])
def test_bbox_matches_full_block_voronoi(downsample):
    labels = voronoi_labels((24, 32, 40), num_seeds=12, seed=0)
    ref = get_lsds_full_block(labels, sigma=5, downsample=downsample)
    out = get_lsds(labels, sigma=5, downsample=downsample)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("downsample", [1, 2])
def test_bbox_matches_full_block_speckles(downsample):
    labels = speckled_labels((32, 32, 32), num_blobs=15, seed=1)
    ref = get_lsds_full_block(labels, sigma=6, downsample=downsample)
    out = get_lsds(labels, sigma=6, downsample=downsample)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)


def test_bbox_matches_full_block_anisotropic():
    labels = voronoi_labels((16, 32, 32), num_seeds=8, seed=2)
    ref = get_lsds_full_block(labels, sigma=(8, 6, 6), voxel_size=(2, 1, 1), downsample=2)
    out = get_lsds(labels, sigma=(8, 6, 6), voxel_size=(2, 1, 1), downsample=2)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)


def test_bbox_matches_full_block_label_subset():
    labels = voronoi_labels((24, 24, 24), num_seeds=10, seed=3)
    subset = [2, 5, 7, 999]  # 999 is absent from the block
    ref = get_lsds_full_block(labels, sigma=4, labels=subset)
    out = get_lsds(labels, sigma=4, labels=subset)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)
    assert np.all(out[:, labels == 1] == 0)


def test_bbox_matches_full_block_2d(real_labels):
    section = real_labels[0, :256, :256]
    ref = get_lsds_full_block(section, sigma=10, downsample=2)
    out = get_lsds(section, sigma=10, downsample=2)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6)


def test_lsds(real_labels):
    lsds_2d = get_lsds(real_labels[0], sigma=10, downsample=2)
    lsds_3d = get_lsds(real_labels, sigma=10, downsample=4)

    assert lsds_2d.shape[0] == 6
    assert lsds_3d.shape[0] == 10


def test_sigmas(real_labels):
    for sigma in [5, 10, 15, 20]:
        _ = get_lsds(real_labels[0], sigma=sigma, downsample=2)


def test_empty_labels(empty_labels):
    """Test that get_lsds handles empty (all-background) segmentations."""
    lsds = get_lsds(empty_labels, sigma=5, downsample=1)

    # Should return zeros with correct shape
    assert lsds.shape == (10,) + empty_labels.shape
    assert np.all(lsds == 0)

