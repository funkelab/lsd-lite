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

    segmentation = segmentation[tuple(slice(None, None, df) for _ in range(dims))]

    sub_shape = segmentation.shape
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

        mask: np.ndarray = segmentation == label
        masked_coords = coords * mask

        aggregate = functools.partial(
            gaussian_filter,
            mode="constant",
            cval=0.0,
            truncate=3.0,
        )

        # simply a mask convolved with a Gaussian
        mass = aggregate(
            mask.astype(np.float32),
            sigma=sub_sigma_voxel,
        )

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
        mean_offset *= mask

        # covariance
        coords_outer = outer_product(masked_coords)
        center_of_mass_outer = outer_product(center_of_mass)

        # get indices of upper triangle of covariance matrix
        rows, cols = np.triu_indices(dims)
        entries = (rows * dims + cols).tolist()

        # sort them s.t. the diagonal entries come first. the first `dims` are
        # the diagonals
        entries = sorted(
            entries, key=lambda x: x % (dims + 1) * (dims + 1) + x // (dims + 1)
        )
        covariance = (
            np.array([aggregate(coords_outer[d], sub_sigma_voxel) for d in entries])
            / mass
        )
        covariance -= center_of_mass_outer[entries]

        for ind, entry in enumerate(entries):
            x, y = entry // dims, entry % dims
            covariance[ind] /= sigma[x] * sigma[y]

        descriptor = np.concatenate((mean_offset, covariance, mass[None, :]))

        mask = mask[None][[0] * descriptor.shape[0], ...]
        masked_descriptor = np.zeros_like(descriptor)
        masked_descriptor[mask] = descriptor[mask]
        label_descriptors.append(masked_descriptor)

    descriptors = np.sum(np.array(label_descriptors), axis=0)
    # clip outliers
    np.clip(descriptors, 0.0, 1.0, out=descriptors)

    return upsample(descriptors, df)


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
