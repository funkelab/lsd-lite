from collections.abc import Sequence
from funlib.geometry import Coordinate
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import convolve, gaussian_filter
import functools
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_lsds(
    segmentation: np.ndarray,
    sigma: float | Sequence[float],
    voxel_size: Sequence[int] | None = None,
    labels: Sequence[int] | None = None,
    downsample: int = 1,
):
    """
    Compute local shape descriptors for the given segmentation.

    Args:

        segmentation (``np.array`` of ``int``):

            A label array to compute the local shape descriptors for.

        sigma (``tuple`` of ``float``):

            The radius to consider for the local shape descriptor.

        voxel_size (``tuple`` of ``int``, optional):

            The voxel size of ``segmentation``. Defaults to 1.

        labels (array-like of ``int``, optional):

            Restrict the computation to the given labels. Defaults to all
            labels inside the ``roi`` of ``segmentation``.

        downsample (``int``, optional):

            Compute the local shape descriptor on a downsampled volume for
            faster processing. Defaults to 1 (no downsampling).
    """

    assert all([(s // downsample) * downsample == s for s in segmentation.shape]), (
        f"Segmentation shape {segmentation.shape} must be divisible by "
        f"downsample factor {downsample}."
    )

    dims = len(segmentation.shape)
    if isinstance(sigma, (int, float)):
        sigma = (sigma,) * dims

    assert len(sigma) == dims, (
        f"Sigma {sigma} must have the same number of dimensions as "
        f"segmentation. shape: {segmentation.shape}."
    )

    if voxel_size is None:
        voxel_size = Coordinate((1,) * dims)
    else:
        voxel_size = Coordinate(voxel_size)

    assert voxel_size.dims == dims, (
        f"Voxel size {voxel_size} must have the same number of dimensions as "
        f"segmentation. shape: {segmentation.shape}."
    )

    if labels is None:
        labels = np.unique(segmentation)

    # prepare full-res descriptor volumes for roi
    # channels = 10 if dims == 3 else 6
    # descriptors = np.zeros((channels,) + segmentation.shape, dtype=np.float32)

    # get sub-sampled shape, roi, voxel size and sigma
    df = downsample
    logger.debug("Downsampling segmentation %s with factor %f", segmentation.shape, df)

    sub_segmentation = segmentation[tuple(slice(None, None, df) for _ in range(dims))]

    sub_shape = sub_segmentation.shape
    sub_voxel_size = tuple(v * df for v in voxel_size)
    sub_sigma_voxel = tuple(s / v for s, v in zip(sigma, sub_voxel_size))

    grid = np.meshgrid(
        *[
            np.arange(0, sub_shape[dim] * sub_voxel_size[dim], sub_voxel_size[dim])
            for dim in range(dims)
        ],
        indexing="ij",
    )
    coords = np.array(grid, dtype=np.float32)

    # normalize stats
    # get max possible mean offset for normalization
    # farthest voxel in context is 3*sigma away, but due to Gaussian
    # weighting, sigma itself is probably a better upper bound
    max_distance = np.array([s for s in sigma], dtype=np.float32)

    # for all labels
    label_descriptors = []
    for label in labels:
        if label == 0:
            continue

        sub_mask: np.ndarray = sub_segmentation == label
        masked_coords = coords * sub_mask

        aggregate = functools.partial(
            gaussian_filter,
            mode="constant",
            cval=0.0,
            truncate=3.0,
        )

        # simply a mask convolved with a Gaussian
        mass = aggregate(
            sub_mask.astype(np.float32),
            sigma=sub_sigma_voxel,
        )
        mass[mass == 0] = 1

        # offsets (meshgrid convolved with Gaussian, divided by mass, minus
        # meshgrid)
        center_of_mass = (
            np.array(
                [
                    aggregate(
                        masked_coords[d],
                        sigma=sub_sigma_voxel,
                    )
                    for d in range(dims)
                ]
            )
            / mass
        )
        mean_offset = center_of_mass - coords
        mean_offset = (
            mean_offset / max_distance.reshape((-1,) + (1,) * dims) * 0.5 + 0.5
        )

        # covariance
        coords_outer = outer_product(masked_coords)
        center_of_mass_outer = outer_product(center_of_mass)

        diag_entries = [i * dims + i for i in range(dims)]
        off_entries = [i * dims + j for i in range(dims) for j in range(i + 1, dims)]
        entries = diag_entries + off_entries

        covariance = (
            np.array([aggregate(coords_outer[d], sub_sigma_voxel) for d in entries])
            / mass
        )
        covariance -= center_of_mass_outer[entries]

        variance = covariance[:dims]
        variance[...] = np.maximum(variance, 1e-3)  # floor for stability

        # off-diagonals pearson
        k = 0
        for i in range(dims):
            for j in range(i + 1, dims):
                covariance[dims + k] /= np.sqrt(variance[i] * variance[j])
                k += 1

        # diagonals, normalize by sigma
        for i in range(dims):
            covariance[i] /= sigma[i] * sigma[i]

        descriptor = np.concatenate((mean_offset, covariance, mass[None, :]))
        descriptor = upsample(descriptor, df)
        label_descriptors.append(descriptor * (segmentation == label)[None, ...])

    descriptors = np.sum(np.array(label_descriptors), axis=0)

    n_mean = dims
    n_variance = dims
    n_pearsons = dims * (dims - 1) // 2

    pearsons_slice = slice(n_mean + n_variance, n_mean + n_variance + n_pearsons)

    # rescale pearsons
    if n_pearsons:
        descriptors[pearsons_slice] = descriptors[pearsons_slice] * 0.5 + 0.5
        # reset background
        descriptors[pearsons_slice] *= segmentation[(slice(None),)] != 0

    # clip outliers
    np.clip(descriptors, 0.0, 1.0, out=descriptors)

    return descriptors


def outer_product(array):
    """Computes the unique values of the outer products of the first dimension
    of ``array``. If ``array`` has shape ``(k, d, h, w)``, for example, the
    output will be of shape ``(k*(k+1)/2, d, h, w)``.
    """

    k = array.shape[0]
    outer = np.einsum("i...,j...->ij...", array, array)
    return outer.reshape((k**2,) + array.shape[1:])


def upsample(array, f):
    shape = array.shape
    stride = array.strides

    if len(array.shape) == 4:
        sh = (shape[0], shape[1], f, shape[2], f, shape[3], f)
        st = (stride[0], stride[1], 0, stride[2], 0, stride[3], 0)
    else:
        sh = (shape[0], shape[1], f, shape[2], f)
        st = (stride[0], stride[1], 0, stride[2], 0)

    view = as_strided(array, sh, st)

    ll = [shape[0]]
    [ll.append(shape[i + 1] * f) for i, j in enumerate(shape[1:])]

    return view.reshape(ll)


def deriv_based_covariance(sub_voxel_size, mass, sub_sigma_voxel):
    """
    Instead of using an inner product to compute covariance, we could
    take derivatives. Might be more efficient but seems harder to normalize
    appropriately between small and large objects.
    """
    k_y = np.zeros((3, 1), dtype=np.float32)
    k_y[0, 0] = sub_voxel_size[0]
    k_y[2, 0] = -sub_voxel_size[0]

    k_x = np.zeros((1, 3), dtype=np.float32)
    k_x[0, 0] = sub_voxel_size[1]
    k_x[0, 2] = -sub_voxel_size[1]

    # first derivatives
    d_y = convolve(mass, k_y, mode="constant")
    d_x = convolve(mass, k_x, mode="constant")

    # second derivatives
    d_yy = convolve(d_y, k_y, mode="constant")
    d_xx = convolve(d_x, k_x, mode="constant")
    d_yx = convolve(d_y, k_x, mode="constant")

    norm = 1
    d_y *= norm * sub_sigma_voxel[0]
    d_x *= norm * sub_sigma_voxel[1]

    d_yy *= norm * sub_sigma_voxel[0] ** 2
    d_xx *= norm * sub_sigma_voxel[1] ** 2
    d_yx *= norm * sub_sigma_voxel[0] * sub_sigma_voxel[1]

    _mean_offset = np.stack([d_y, d_x]) * 0.5 + 0.5
    _covariance = np.stack([d_yy, d_xx, d_yx]) * 0.5 + 0.5
