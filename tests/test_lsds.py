from lsd_lite import get_lsds


def test_lsds(real_labels):
    lsds_2d = get_lsds(real_labels[0], sigma=10, downsample=2)
    lsds_3d = get_lsds(real_labels, sigma=10, downsample=4)

    assert lsds_2d.shape[0] == 6
    assert lsds_3d.shape[0] == 10


def test_sigmas(real_labels):
    for sigma in [5, 10, 15, 20]:
        _ = get_lsds(real_labels[0], sigma=sigma, downsample=2)
