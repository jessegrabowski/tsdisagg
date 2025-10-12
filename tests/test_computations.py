import numpy as np
import pandas as pd

from numpy.testing import assert_allclose

from tsdisagg.ts_disagg import (
    GLS_beta_hat,
    build_chao_lin_covariance,
    build_conversion_matrix,
    build_litterman_covariance,
)


def test_chow_lin_covariance():
    expected = np.array(
        [
            [10.25641, 9.74359, 9.25641],
            [9.74359, 10.25641, 9.74359],
            [9.25641, 9.74359, 10.25641],
        ]
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


def test_GLM_estimator():
    # From tempdisagg:::CalcGLS, rho = 0.95
    expected = np.array([908.6679748, 0.9772889])

    low_freq_data = pd.read_csv("tests/data/AL_Annual_Data_Shorter.csv", parse_dates=True, index_col="period").dropna()
    low_freq_data.index.freq = low_freq_data.index.inferred_freq
    high_freq_data = pd.read_csv(
        "tests/data/AL_Quarterly_Data_Modified.csv",
        parse_dates=True,
        index_col="period",
    ).dropna()
    high_freq_data.index.freq = high_freq_data.index.inferred_freq

    C = build_conversion_matrix(low_freq_data, high_freq_data, 4, "last")

    # This particular data isn't aligned, so tempdisagg drops the first low-frequency observation
    C = C[1:, :]
    y = low_freq_data.values[1:]
    X = high_freq_data.assign(intercept=1)[["intercept", "Value"]].values

    Sigma = build_chao_lin_covariance(0.95, 1.0, n=X.shape[0])
    params = GLS_beta_hat(Sigma, y, X, C).ravel()

    np.testing.assert_allclose(params, expected)
