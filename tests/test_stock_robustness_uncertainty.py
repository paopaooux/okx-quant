import numpy as np
import pandas as pd
import pytest

from scripts.analysis.stock_robustness_uncertainty import block_indices, bootstrap_pair


def test_blocks_are_contiguous_and_reproducible():
    a = block_indices(16, repetitions=20)
    np.testing.assert_equal(a, block_indices(16, repetitions=20))
    assert a.shape == (20, 16)
    assert a.min() >= 0 and a.max() < 16
    assert np.all(np.diff(a[:, :7]) == 1)


def test_identical_paired_paths_have_zero_difference():
    curve = pd.Series(np.exp(np.arange(16)*.001), index=pd.date_range("2026-07-01", periods=16))
    result = bootstrap_pair(curve, curve, repetitions=50)
    assert result["delta_ci_low"] == result["delta_ci_high"] == result["paired_delta_pp"] == 0
    with pytest.raises(ValueError):
        bootstrap_pair(curve.iloc[1:], curve, repetitions=50)
