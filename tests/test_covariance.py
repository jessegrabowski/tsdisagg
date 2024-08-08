import numpy as np
from numpy.testing import assert_allclose

from tsdisagg.ts_disagg import build_chao_lin_covariance


def test_chow_lin_covariance():
    expected = np.array(
        [[10.25641, 9.74359, 9.25641], [9.74359, 10.25641, 9.74359], [9.25641, 9.74359, 10.25641]]
    )

    CL_cov = build_chao_lin_covariance(0.95, 1.0, n=3)
    assert_allclose(CL_cov, expected)
