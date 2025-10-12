import warnings

from typing import Literal, cast

import numpy as np
import pandas as pd

from scipy import linalg, stats
from scipy.optimize import OptimizeResult, minimize

from tsdisagg.time_conversion import (
    FREQ_CONVERSION_FACTORS,
    auto_step_down_base_freq,
    get_frequency_name,
    make_companion_index,
    make_names_from_frequencies,
    validate_freqs,
)

AGG_FUNC = Literal["sum", "mean", "first", "last"]
METHOD = Literal["denton", "denton-cholette", "chow-lin", "litterman"]


def _get_C_index_and_fill(
    idx: np.ndarray[int], agg_func: AGG_FUNC, time_conversion_factor: int
) -> tuple[np.ndarray[int] | int, float]:
    if agg_func in ["sum", "first", "last"]:
        fill_value = 1.0
    elif agg_func == "mean":
        fill_value = 1 / time_conversion_factor
    else:
        raise ValueError("Invalid method")

    if len(idx) != time_conversion_factor:
        fill_value = 0.0

    if agg_func == "first":
        idx = idx[0]
    elif agg_func == "last":
        idx = idx[-1]

    return idx, fill_value


def build_conversion_matrix(
    low_freq_df: pd.Series | pd.DataFrame,
    high_freq_df: pd.Series | pd.DataFrame,
    time_conversion_factor: int,
    agg_func: AGG_FUNC,
):
    high_freq_df = high_freq_df.copy()
    if isinstance(high_freq_df, pd.Series):
        high_freq_df = high_freq_df.to_frame()

    n_low, n_high = low_freq_df.shape[0], high_freq_df.shape[0]

    low_index, high_index = low_freq_df.index, high_freq_df.index
    low_freq, _high_freq = low_index.freq, high_index.freq

    low_freq_period = "Y" if low_freq.name.startswith("Y") or low_freq.name.startswith("BY") else "Q"
    high_freq_df["low_freq_period"] = high_freq_df.index.to_period(freq=low_freq_period)
    period_to_row_idx = {period: idx for idx, period in enumerate(low_freq_df.index.to_period(low_freq_period))}

    C = np.zeros((n_low, n_high))
    grouped = high_freq_df.groupby("low_freq_period")

    for low_freq_period, _group in grouped:
        row_idx = period_to_row_idx.get(low_freq_period)
        if row_idx is not None:
            idx = cast(
                np.ndarray[int],
                np.flatnonzero(high_freq_df.low_freq_period == low_freq_period),
            )
            idx, fill_value = _get_C_index_and_fill(idx, agg_func, time_conversion_factor)
            C[row_idx, idx] = fill_value

    return C


def log_likelihood(nl, CΣCT, ul):
    sign, log_det = np.linalg.slogdet(CΣCT)

    return -nl / 2 * np.log(2 * np.pi) - 0.5 * (log_det + ul.T @ np.linalg.solve(CΣCT, ul))


def build_difference_matrix(n, h=0):
    Δ = np.eye(n)
    Δ[np.where(np.eye(n, k=-1))] = -1
    return np.linalg.matrix_power(Δ, h)


def build_distribution_matrix(Σ, C):
    return np.linalg.solve(np.linalg.multi_dot([C, Σ, C.T]), C @ Σ).T


def build_chao_lin_covariance(rho, sigma_e_sq, n):
    iota = np.arange(n)[:, None].repeat(n, axis=1)
    power_matrix = np.abs(iota - iota.T)

    Σ_CL = rho**power_matrix
    Σ_CL *= sigma_e_sq / (1 - rho**2)

    return Σ_CL


def build_litterman_covariance(rho, sigma_e_sq, n):
    Δ = build_difference_matrix(n, h=1)
    H_rho = np.eye(n, k=-1) * -rho + np.eye(n)
    return sigma_e_sq * np.linalg.solve(Δ.T @ H_rho.T @ H_rho @ Δ, np.eye(n))


def GLS_beta_hat(Σ, y, X, C):
    CΣCT = np.linalg.multi_dot([C, Σ, C.T])
    CX = C @ X
    XTCT = X.T @ C.T

    lu, piv = linalg.lu_factor(CΣCT)
    Z1 = linalg.lu_solve((lu, piv), CX)
    Z2 = linalg.lu_solve((lu, piv), y)

    A = XTCT @ Z1
    B = XTCT @ Z2

    return np.linalg.solve(A, B)


def f_minimize(params, y, X, C, f_cov):
    n, k = X.shape
    nl = y.shape[0]

    ρ, sigma_e_sq = params

    # TODO: This correction is just pure magic. It makes the results match, but I haven't been able to figure out
    #  why it's needed (it changes the scaling factor of the CL covariance from sigma ** 2 / (1 - rho **2) to
    #  sigma ** 2 / (1 - rho)
    #  I think it has to do with the fact that I'm estimating sigma, whereas timedisagg just computes it via the RSS?
    sigma_e_sq = (1 + ρ) * sigma_e_sq

    Σ = f_cov(ρ, sigma_e_sq, n)
    β = GLS_beta_hat(Σ, y, X, C)

    p = X @ β
    ul = y - C @ p
    CΣCT = np.linalg.multi_dot([C, Σ, C.T])
    return -log_likelihood(nl, CΣCT, ul)


def build_denton_covariance(n, C, X, h=1, criterion="proportional"):  # noqa: ARG001
    Δ = build_difference_matrix(n, h)
    if criterion == "proportional":
        Δ = Δ @ np.diag(1 / X.ravel() / X.mean())
    return np.linalg.solve(Δ.T @ Δ, np.eye(n))


def build_denton_charlotte_distribution_matrix(n, nl, C, X, h=1, criterion="proportional"):
    # Here is the Charlotte correction: slice off the top h rows of the difference matrix
    Δ = build_difference_matrix(n, h)[h:, :]
    if criterion == "proportional":
        Δ = Δ @ np.diag(1 / X.ravel() / X.mean())
    W_1 = np.r_[np.c_[Δ.T @ Δ, C.T], np.c_[C, np.zeros((nl, nl))]]
    W_2 = np.r_[np.c_[Δ.T @ Δ, np.zeros((n, nl))], np.c_[C, np.eye(nl)]]
    W = np.linalg.solve(W_1, W_2)

    w_theta = W[:n, n:]
    W[n:, n:]

    return w_theta


def _compute_D_and_p(method, y, X, C, n, nl, k, h, criterion, optimizer_kwargs, verbose):
    result = None
    if method == "denton":
        assert k == 1
        Σ = build_denton_covariance(n, C, X.values, h, criterion)
        D = build_distribution_matrix(Σ, C)
        p = X.values.ravel()

    elif method == "denton-cholette":
        assert k == 1
        D = build_denton_charlotte_distribution_matrix(n, nl, C, X.values, h, criterion)
        p = X.values.ravel()

    else:
        if optimizer_kwargs is None:
            optimizer_kwargs = {"method": "nelder-mead"}
        if optimizer_kwargs and "method" not in optimizer_kwargs:
            optimizer_kwargs.update({"method": "nelder-mead"})

        if method == "chow-lin":
            f_cov = build_chao_lin_covariance
        elif method == "litterman":
            f_cov = build_litterman_covariance
        else:
            raise ValueError(f"Method {method} not supported.")

        # betas are unbounded, bound rho between 0 and 1 and sigma between 0 and +inf
        bounds = [(1e-5, 1 - 1e-5), (1e-5, None)]

        x0 = np.full(2, 0.8)
        result = minimize(
            f_minimize,
            x0=x0,
            args=(y.values, X.values, C, f_cov),
            bounds=bounds,
            **optimizer_kwargs,
        )

        ρ, sigma_e_sq = result.x
        Σ = f_cov(ρ, sigma_e_sq, n)
        Σ_inv_X = np.linalg.solve(Σ, X.values)

        β = GLS_beta_hat(Σ, y.values, X.values, C)
        std_β = np.sqrt(np.diagonal(np.linalg.inv(X.values.T @ Σ_inv_X)))

        if verbose:
            print_regression_report(y, X, np.r_[β, ρ, sigma_e_sq], std_β, C, method)

        p = X.values @ β
        D = build_distribution_matrix(Σ, C)

    return D, p, result


def print_regression_report(y, X, params, std_β, C, method):
    print(f"Dependent Variable: {y.name}")
    print(f"GLS Estimates using {method.title()}'s covariance matrix")
    print(f"N = {X.shape[0]}\t\tdf = {X.shape[0] - len(params)}")
    N, k = X.shape
    deg_f = N - len(params)
    t_dist = stats.t(df=deg_f)

    ul = y.dropna().values - C @ X.values @ params[:-2]
    r2 = 1 - np.var(ul) / (y.dropna() - y.mean()).var()
    adj_r2 = 1 - (1 - r2) * (N - 1) / (N - k - 1)

    print(f"Adj r2 = {adj_r2:0.4f}")
    print("")

    print(f'{"Variable":<15}{"coef":>10}{"sd err":>15}{"t":>15}{"P > |t|":>15}{"[0.025":>15}{"0.975]":>15}')
    print("-" * 100)
    for i, var in enumerate(X.columns):
        t_05 = t_dist.ppf(1 - 0.05 / 2)
        ci_low = params[i] - std_β[i] * t_05
        ci_high = params[i] + std_β[i] * t_05

        t_stat = params[i] / std_β[i]
        p_value = t_dist.sf(np.abs(t_stat))
        print(
            f"{var:<15}{params[i]:>10.4f}{std_β[i]:>15.4f}{t_stat:>15.4f}{p_value:>15.4f}{ci_low:>15.4f}{ci_high:>15.4f}"
        )
    print("")
    print(f'{"rho":<15}{params[-2]:>10.4f}')
    print(f'{"sigma.sq":<15}{params[-1]:>10.4f}')


def prepare_input_dataframes(low_freq_df, high_freq_df, target_freq, method):  # noqa: PLR0912, C901
    low_freq_df_out = low_freq_df.copy()

    if not isinstance(low_freq_df.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise TypeError("No datetime index found on the dataframe passed as argument to low_freq_df.")

    if low_freq_df.isna().any().any():
        raise ValueError("low_freq_df has missing values.")

    if high_freq_df is not None:
        if not isinstance(high_freq_df.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise ValueError("No datetime index found on the dataframe passed as argument to high_freq_df.")

        if high_freq_df.isna().any().any():
            raise ValueError("high_freq_df has missing values.")

        if high_freq_df.index[0] > low_freq_df.index[0]:
            raise ValueError(
                f"Start date found on high frequency data {high_freq_df.index[0].strftime('%Y-%m-%d')} is after start "
                f"date found on low frequency data {low_freq_df.index[0].strftime('%Y-%m-%d')}. Interpolation is not "
                f"possible in this case, because there is no observed high frequency data associated with the first "
                f"{(low_freq_df.index < high_freq_df.index[0]).sum()} low-frequency observations. "
                f"Align the start date of these two dataframes and try again."
            )

        high_freq_df_out = high_freq_df.copy()
    else:
        high_freq_df_out = high_freq_df

    low_freq = low_freq_df_out.index.freq or low_freq_df.index.inferred_freq
    if not low_freq:
        raise ValueError("Low frequency dataframe does not have a valid time index with frequency information")

    if high_freq_df_out is None and target_freq is None:
        high_freq = auto_step_down_base_freq(low_freq)
    elif high_freq_df_out is None and target_freq is not None:
        high_freq = target_freq
    elif high_freq_df_out is not None and target_freq is not None:
        if high_freq_df_out.index.inferred_freq != target_freq:
            raise ValueError(
                "User provided target_freq does not match frequency information found on indicator data "
                "high_freq_df."
            )
        high_freq = target_freq
    else:
        high_freq = high_freq_df_out.index.inferred_freq
        if not high_freq:
            raise ValueError("Indicator data high_freq_df does not have a valid time index with frequency information")

    validate_freqs(low_freq, high_freq)

    high_name = get_frequency_name(high_freq)
    low_name = get_frequency_name(low_freq)
    time_conversion_factor = FREQ_CONVERSION_FACTORS[low_name][high_name]

    var_name, low_freq_name, high_freq_name = make_names_from_frequencies(low_freq_df_out, high_freq)

    if isinstance(low_freq_df_out, pd.Series):
        low_freq_df_out.name = low_freq_name
    elif isinstance(low_freq_df_out, pd.DataFrame):
        low_freq_df_out.rename(columns={var_name: low_freq_name}, inplace=True)

    if high_freq_df_out is None and method in ["denton", "denton-cholette"]:
        high_freq_idx = make_companion_index(low_freq_df_out, target_freq=high_freq)
        high_freq_df_out = pd.Series(1, index=high_freq_idx, name=high_freq_name)

    elif high_freq_df_out is None:
        raise ValueError(
            'high_freq_df can only be None for methods "denton" and "denton-cholette", otherwise a '
            "dataframe of high-frequency indicators must be provided."
        )

    low_freq_df_out.index.freq = low_freq_df_out.index.inferred_freq
    high_freq_df_out.index.freq = high_freq_df_out.index.inferred_freq

    df = pd.merge(low_freq_df_out, high_freq_df_out, left_index=True, right_index=True, how="outer")
    return df, low_freq_df_out, high_freq_df_out, time_conversion_factor


def disaggregate_series(
    low_freq_df,
    high_freq_df=None,
    target_freq=None,
    target_column=None,
    agg_func: AGG_FUNC = "sum",
    method: METHOD = "denton-cholette",
    criterion="proportional",
    h=1,
    optimizer_kwargs=None,
    verbose=True,
    return_optim_res=False,
) -> pd.DataFrame | tuple[pd.DataFrame, OptimizeResult]:
    """
    Transform a low frequency time series into a higher frequency series, preserving certain aggregate statistics.

    Parameters
    ----------
    low_freq_df: DataFrame
        Low frequency time series to be converted
    high_freq_df: Dataframe, Optional
        High frequency companion data. The user can optionally supply data that is believed to be correlated with the
        low frequency data. This data will be used to interpolate the missing high frequency observations.
    target_freq: str, Optional
        The desired output frequency for the low frequency data. If None, the frequency of the low frequency data will
        be "stepped down" once, e.g. yearly to quarterly, or quarterly monthly. Should be a valid pandas frequency
        string; see the pandas documentation for details.
    target_column: str, default "None"
        If low_freq_df has multiple columns, the name of the column to be converted. If None, the first column in the
        dataframe will be used.
    agg_func: str, default "sum"
        Function which the interpolated high frequency data should respect when aggregating back to the original low
        frequency. One of: sum, mean, first, last.
    criterion: str, default "proportional"
        Objective function used by methods "denton" and "denton-cholette" to minimize the sum of square deviations from
        the low frequency data and the interpolated high frequency data. One of "proportional" (deviation) or
        "additive" (absolute deviation).
    method: str, default "denton"
        Method of interpolation to use. Currently, "denton", "denton-cholette", "chow-lin", and "litterman" are
        supported.
            Denton: Univariate method, naive interpolation that preserves the statistic specificed in the agg_func
                    argument only. The first several observation in the produced series will have unacceptably high
                    variance, so "denton-cholette" is recommended in all cases.
            Denton-Cholette: As Denton, but with an h-differences correction applied to reduce variance in the first
                    few observations.
            Chow-Lin: Multivariate regression-based method which allows for multiple high-frequency indicator series
                    to be provided. For details, see [1]
            Litterman: Multivariate regression-based method which allows for multiple high-frequency indicator series
                    to be provided. For details, see [1].
    h: int, default 1
        Number of differences to use when applying the Cholette correction to the Denton method. The Cholette method
        corrects for high variance in the first few observations when using the Denton method. A higher h will result
        in a more aggressive correction.
    optimizer_kwargs: dict, Optional
        A dictionary of keyword arguments to be passed to scipy.optimize.minimize. For full details, consult the
        docstring of that function. Ignoired if method is "denton" or "denton-cholette", as these do not use numerical
        optimization.
    verbose: bool, default True
        Whether to print regression results. Ignored if method is "denton" or "denton-cholette", as these have no
        regression results to report.
    return_optim_res: bool, default False
        Whether to return the optimization results from scipy.optimize.minimize. Ignored if mode is "denton" or
        "denton-cholette"

    Returns
    -------
    high_freq_df: Series
        A Pandas Series object containing the interpolated high frequency data.

    result: OptimizeResult
        Optimization result returned by scipy.optimize.minimize. Only returned if return_optimizer_result is True
    """
    if isinstance(low_freq_df, pd.Series):
        low_freq_df = low_freq_df.to_frame()

    if method not in ["denton", "denton-cholette", "chow-lin", "litterman"]:
        raise ValueError(f"Method should be one of 'denton', 'denton-cholette', 'chow-lin', 'litterman'. Got {method}.")
    if criterion not in ["proportional", "additive"]:
        raise ValueError(f"Criterion should be one of 'proportional', 'additive'. Got {criterion}")
    if agg_func not in ["mean", "sum", "first", "last"]:
        raise ValueError(f"agg_func should be one of 'mean', 'sum', 'first', 'last'. Got {agg_func}")

    target_column = target_column or low_freq_df.columns[0]
    target_idx = np.flatnonzero(low_freq_df.columns == target_column)[0]

    df, low_freq_df, high_freq_df, time_conversion_factor = prepare_input_dataframes(
        low_freq_df, high_freq_df, target_freq, method
    )

    C = build_conversion_matrix(low_freq_df, high_freq_df, time_conversion_factor, agg_func)
    drop_rows = np.all(C == 0, axis=1)
    if any(drop_rows):
        dropped = low_freq_df.index.strftime("%Y-%m-%d")[drop_rows]
        warnings.warn(
            f'Insufficent high-frequency data to decompose the following dates: {", ".join(dropped)}',
            UserWarning,
            stacklevel=2,
        )

    y = df.iloc[:, target_idx].dropna().loc[~drop_rows]
    C = C[~drop_rows, :]
    X = df.drop(columns=df.columns[target_idx])

    n, k = X.shape
    nl = y.shape[0]

    D, p, result = _compute_D_and_p(method, y, X, C, n, nl, k, h, criterion, optimizer_kwargs, verbose)

    ul = y - C @ p
    y_hat = p + D @ ul

    output = pd.Series(y_hat, index=df.index, name=target_column)
    output.index.freq = output.index.inferred_freq

    if return_optim_res and result is not None:
        return output, result

    return output
