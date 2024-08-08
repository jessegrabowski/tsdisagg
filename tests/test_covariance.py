import numpy as np
from numpy.testing import assert_allclose

from tsdisagg.ts_disagg import build_chao_lin_covariance, build_litterman_covariance


def test_chow_lin_covariance():
    expected = np.array(
        [[10.25641, 9.74359, 9.25641], [9.74359, 10.25641, 9.74359], [9.25641, 9.74359, 10.25641]]
    )

    CL_cov = build_chao_lin_covariance(0.95, 1.0, n=3)
    assert_allclose(CL_cov, expected)


def test_litterman_covariance():
    # From tempdisagg:::CalcQ_Lit, rho = 0.95
    expected = np.array(
        [
            [1.000000, 1.950000, 2.852500, 3.709875, 4.524381],
            [1.950000, 4.802500, 7.512375, 10.086756, 12.532418],
            [2.852500, 7.512375, 12.939256, 18.094793, 22.992554],
            [3.709875, 10.086756, 18.094793, 26.702429, 34.879682],
            [4.524381, 12.532418, 22.992554, 34.879682, 47.172454],
        ]
    )
    Lit_cov = build_litterman_covariance(0.95, 1.0, 5)
    assert_allclose(Lit_cov, expected)
