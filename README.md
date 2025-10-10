# lsd-lite
Simple LSD and affinity graph computation

Install: `pip install lsd-lite`

Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/funkelab/lsd-lite/blob/main/lsd_lite_tutorial.ipynb)

Example:

```py
from lsd_lite import get_affs, get_lsds

# 2d short range affs example
affs = get_affs(segmentation, neighborhood=[[1,0],[0,1]])

# 2d long range affs example
affs = get_affs(
    segmentation,
    neighborhood=[
      [1,0],
      [0,1],
      [3,0],
      [0,3],
      [6,0],
      [0,6]
    ]
)

# 3d long range adds example
affs = get_affs(
    segmentation,
    neighborhood=[
      [1,0,0],
      [0,1,0],
      [3,0,0],
      [0,3,0],
      [0,0,3],
      [6,0,0],
      [0,6,0],
      [0,0,6]
    ]
)

# lsds example
lsds = get_lsds(segmentation, sigma=10, downsample=2)
```
