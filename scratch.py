from lsd.train import LsdExtractor  # "Old"
from lsd_lite import get_lsds  # "New"
from skimage import data

from dacapo_toolbox.vis.preview import cube
from funlib.persistence import Array
from scipy.ndimage import label


segmentation = label(
    data.binary_blobs(
        length=8, blob_size_fraction=32 / 64, volume_fraction=0.5, n_dim=3
    )
)[0]

sigma = 5
downsample_lsds = 4

extractor = LsdExtractor(
    sigma=(sigma,) * 3,
    downsample=1,
)
extractor_downsampled = LsdExtractor(
    sigma=(sigma,) * 3,
    downsample=downsample_lsds,
)

lsds_old = extractor.get_descriptors(
    segmentation=segmentation,
    voxel_size=(1,) * 3,
)

lsds_old_downsampled = extractor_downsampled.get_descriptors(
    segmentation=segmentation,
    voxel_size=(1,) * 3,
)

lsds_new = get_lsds(
    segmentation=segmentation,
    sigma=(sigma,) * 3,
    voxel_size=(1,) * 3,
    downsample=1,
)

lsds_new_downsampled = get_lsds(
    segmentation=segmentation,
    sigma=(sigma,) * 3,
    voxel_size=(1,) * 3,
    downsample=downsample_lsds,
)

for arr, name in zip(
    [lsds_old, lsds_old_downsampled, lsds_new, lsds_new_downsampled],
    ["lsds_old", "lsds_old_downsampled", "lsds_new", "lsds_new_downsampled"],
):
    print(name)
    for (a, b), name in zip([(0, 3), (3, 6), (6, 9), (9, 10)], ["offsets", "variance", "pearson", "mass"]):
        print(f"  {name}: {arr[a:b].min()} {arr[a:b].max()} {arr[a:b].mean()}")

for start, stop in [(0, 3), (3, 6), (6, 9), (9, 10)]:
    cube(
        {
            "segmentation": Array(segmentation),
            "lsds_old": Array(
                lsds_old[start:stop],
                voxel_size=(1, 1, 1),
            ),
            "lsds_old_downsampled": Array(
                lsds_old_downsampled[start:stop],
                voxel_size=(1, 1, 1),
            ),
            "lsds_new": Array(
                lsds_new[start:stop],
                voxel_size=(1, 1, 1),
            ),
            "lsds_new_downsampled": Array(
                lsds_new_downsampled[start:stop],
                voxel_size=(1, 1, 1),
            ),
        },
        array_types={
            "segmentation": "labels",
            "lsds_old": "raw",
            "lsds_old_downsampled": "raw",
            "lsds_new": "raw",
            "lsds_new_downsampled": "raw",
        },
        filename=f"scratch-{start}-{stop}.png",
        title=f"lsds {start}:{stop}",
    )
