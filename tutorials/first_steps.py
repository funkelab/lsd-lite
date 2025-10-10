# %% [markdown]
# # First Steps with LSD-Lite

# %%
from scipy.ndimage import gaussian_filter, maximum_filter, label
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import numpy as np


# %%
def create_random_segmentation(size, seed):
    np.random.seed(seed)
    peaks = np.random.random(size).astype(np.float32)
    peaks = gaussian_filter(peaks, sigma=5.0)
    max_filtered = maximum_filter(peaks, 10)
    maxima = max_filtered == peaks
    seeds, n = label(maxima)
    print(f"Creating segmentation with {n} segments")
    return watershed(1.0 - peaks, seeds).astype(np.uint64)


# %%
# 2d
segmentation = create_random_segmentation((64,) * 2, seed=42)

# %%
plt.imshow(segmentation)

# %%
from lsd_lite import get_affs

affs = get_affs(segmentation, neighborhood=[[1, 0], [0, 1]])

plt.imshow(affs.mean(axis=0))

# %%
affs = get_affs(
    segmentation, neighborhood=[[1, 0], [0, 1], [3, 0], [0, 3], [6, 0], [0, 6]]
)

plt.imshow(affs.mean(axis=0))


# %%
def view(data):
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(10, 7))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.imshow(data[i])
        ax.set_title(f"Channel {i + 1}")

    plt.tight_layout()
    plt.show()


# %%
from lsd_lite import get_lsds

lsds = get_lsds(segmentation, sigma=10)

view(lsds)

# %%
# larger sigma
lsds = get_lsds(segmentation, sigma=20)

view(lsds)

# %%
# smaller sigma
lsds = get_lsds(segmentation, sigma=5)

view(lsds)

# %%
# 3d
segmentation = create_random_segmentation((64,) * 3, seed=42)

# view 10th section
plt.imshow(segmentation[10])

# %%
affs = get_affs(
    segmentation,
    neighborhood=[
        [1, 0, 0],
        [0, 1, 0],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [6, 0, 0],
        [0, 6, 0],
        [0, 0, 6],
    ],
)

# view mid range of section 10 in rgb
plt.imshow(affs[3:6, 10].T.astype(np.float32))

# %%
# add downsample
lsds = get_lsds(segmentation, sigma=10, downsample=2)

# view offsets in rgb
plt.imshow(lsds[0:3, 10].T)

# %%
# view diagonals
plt.imshow(lsds[3:6, 10].T)

# %%
# view off-diagonals
plt.imshow(lsds[6:9, 10].T)

# %%
# view size
plt.imshow(lsds[9, 10])
