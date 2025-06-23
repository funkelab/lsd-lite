from pathlib import Path
import numpy as np
import pytest


@pytest.fixture(scope="session")
def toy_labels() -> np.ndarray:
    """
    4×4×4 toy volume:
      * background -> 0
      * label 1 -> first z section
      * label 2 -> last z section
    """
    labels = np.zeros((4, 4, 4), dtype=np.uint64)
    labels[0] = 1
    labels[-1] = 2
    return labels


@pytest.fixture(scope="session")
def real_labels() -> np.ndarray:
    root = Path(__file__).resolve().parent.parent
    data = np.load(root / "example_data" / "cremi.npz")["labels"]
    return data[:20]  # first 20 sections
