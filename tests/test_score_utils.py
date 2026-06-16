from __future__ import annotations

import numpy as np
import pytest

from physicscourt.pipeline.score_utils import first_sustained_crossing


def test_first_sustained_crossing_returns_start_of_run() -> None:
    values = np.array([-0.2, 0.6, 0.4, 0.7, 0.8, 0.9, 0.1], dtype=np.float32)

    assert first_sustained_crossing(values, threshold=0.5, min_consecutive=3) == 3


def test_first_sustained_crossing_returns_none_without_long_enough_run() -> None:
    values = np.array([0.7, 0.8, 0.1, 0.7, 0.8], dtype=np.float32)

    assert first_sustained_crossing(values, threshold=0.5, min_consecutive=3) is None


def test_first_sustained_crossing_can_require_prior_below_run() -> None:
    values = np.array([0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float32)

    assert first_sustained_crossing(values, threshold=0.5, min_consecutive=3, min_prior_below=3) == 6


def test_first_sustained_crossing_rejects_nonpositive_run_length() -> None:
    with pytest.raises(ValueError, match="min_consecutive"):
        first_sustained_crossing(np.array([1.0], dtype=np.float32), threshold=0.5, min_consecutive=0)
