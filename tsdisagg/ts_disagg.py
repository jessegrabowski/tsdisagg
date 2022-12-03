from scipy import linalg, stats
from scipy.optimize import minimize
import pandas as pd
import numpy as np

from tsdisagg.time_conversion import make_companion_index, auto_step_down_base_freq, validate_freqs, \
    make_names_from_frequencies, align_and_merge_dfs, get_frequency_name, FREQ_CONVERSION_FACTORS


def build_conversion_matrix(n, nl, i_len, C_mask = None, agg_func='sum'):
    excess = n - i_len * nl
    extra_periods = int(np.ceil(excess / i_len))

    C_mask = C_mask or np.full(nl, True)

    if agg_func == 'sum':
        i = np.ones((i_len, 1))
    elif agg_func == 'mean':
        i = np.ones((i_len, 1)) / i_len
    elif agg_func == 'first':
        i = np.zeros((i_len, 1))
        i[0] = 1
    elif agg_func == 'last':
        i = np.zeros((i_len, 1))
        i[-1] = 1

    C = linalg.kron(np.eye(nl + extra_periods), i.T)
    C = C[C_mask, :n]

    return C


def log_likelihood(nl, CΣCT, ul):
    sign, log_det = np.linalg.slogdet(CΣCT)

    return -nl / 2 * np.log(2 * np.pi) - 0.5 * (log_det + ul.T @ np.linalg.solve(CΣCT, ul))


def build_difference_matrix(n, h=0):
    Δ = np.eye(n)
    Δ[np.where(np.eye(n, k=-1))] = -1
    return np.linalg.matrix_power(Δ, h)


def build_distribution_matrix(Σ, C):
    return np.linalg.solve(C @ Σ @ C.T, C @ Σ).T


def build_chao_lin_covariance(rho, sigma_e_sq, n):
    row = rho ** np.arange(n)
    Σ_CL = np.r_[[np.r_[np.zeros(n - i), row[:i]] for i in range(n, 0, -1)]]
    Σ_CL += Σ_CL.T - np.eye(n)
    Σ_CL *= sigma_e_sq / (1 - rho)
    return Σ_CL


def build_litterman_covariance(rho, sigma_e_sq, n):
    Δ = build_difference_matrix(n, h=1)
    H_rho = np.eye(n, k=-1) * -rho + np.eye(n)
    Σ_L = sigma_e_sq * np.linalg.solve(Δ.T @ H_rho.T @ H_rho @ Δ, np.eye(n))
    return Σ_L


def GLS_beta_hat(Σ, y, X, C, nl):
    CΣCT_inv = np.linalg.solve(C @ Σ @ C.T, np.eye(nl))
    A = X.T @ C.T @ CΣCT_inv @ C @ X
    B = X.T @ C.T @ CΣCT_inv @ y
    β = np.linalg.solve(A, B)

    return β


def f_minimize(params, y, X, C, f_cov):
    n, k = X.shape
    nl = y.shape[0]

    ρ, sigma_e_sq = params
    Σ = f_cov(ρ, sigma_e_sq, n)
    β = GLS_beta_hat(Σ, y, X, C, nl)

    p = X @ β
    ul = y - C @ p
    CΣCT = C @ Σ @ C.T
    return -log_likelihood(nl, CΣCT, ul)


def build_denton_covariance(n, C, X, h=1, criterion='proportional'):
    Δ = build_difference_matrix(n, h)
    if criterion == 'proportional':
        Δ = Δ @ np.diag(1 / X.ravel() / X.mean())
    Σ_D = np.linalg.solve(Δ.T @ Δ, np.eye(n))

    return Σ_D


def build_denton_charlotte_distribution_matrix(n, nl, C, X, h=1, criterion='proportional'):
    # Here is the Charlotte correction: slice off the top h rows of the difference matrix
    Δ = build_difference_matrix(n, h)[h:, :]
    if criterion == 'proportional':
        Δ = Δ @ np.diag(1 / X.ravel() / X.mean())
    W_1 = np.r_[np.c_[Δ.T @ Δ, C.T], np.c_[C, np.zeros((nl, nl))]]
    W_2 = np.r_[np.c_[Δ.T @ Δ, np.zeros((n, nl))], np.c_[C, np.eye(nl)]]
    W = np.linalg.solve(W_1, W_2)

    w_theta = W[:n, n:]
    w_gamma = W[n:, n:]

    return w_theta


def print_regression_report(y, X, params, std_β, C, method):
    print(f'Dependent Variable: {y.name}')
    print(f"GLS Estimates using {method.title()}'s covariance matrix")
    print(f'N = {X.shape[0]}\t\tdf = {X.shape[0] - len(params)}')
    N, k = X.shape
    deg_f = N - len(params)
    t_dist = stats.t(df=deg_f)

    ul = y.dropna().values - C @ X.values @ params[:-2]
    r2 = 1 - np.var(ul) / (y.dropna() - y.mean()).var()
    adj_r2 = 1 - (1 - r2) * (N - 1) / (N - k - 1)

    print(f'Adj r2 = {adj_r2:0.4f}')
    print('')

    print(f'{"Variable":<15}{"coef":>10}{"sd err":>15}{"t":>15}{"P > |t|":>15}{"[0.025":>15}{"0.975]":>15}')
    print('-' * 100)
    for i, var in enumerate(X.columns):
        t_05 = t_dist.ppf(1 - 0.05 / 2)
        ci_low = params[i] - std_β[i] * t_05
        ci_high = params[i] + std_β[i] * t_05

        t_stat = params[i] / std_β[i]
        p_value = t_dist.sf(np.abs(t_stat))
        print(
            f'{var:<15}{params[i]:>10.4f}{std_β[i]:>15.4f}{t_stat:>15.4f}{p_value:>15.4f}{ci_low:>15.4f}{ci_high:>15.4f}')
    print('')
    print(f'{"rho":<15}{params[-2]:>10.4f}')
    print(f'{"sigma.sq":<15}{params[-1]:>10.4f}')


def prepare_input_dataframes(df1, df2, target_freq, method):
    df1_out = df1.copy()

    if not isinstance(df1.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise ValueError('No datetime index found on the dataframe passed as argument to df1.')
    if df2 is not None:
        if not isinstance(df2.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise ValueError('No datetime index found on the dataframe passed as argument to df1.')
        df2_out = df2.copy()
    else:
        df2_out = df2

    low_freq = df1_out.index.freq or df1_out.index.inferred_freq
    if not low_freq:
        raise ValueError('Dataframe df1 does not have a valid time index with frequency information')

    if df2_out is None and target_freq is None:
        high_freq = auto_step_down_base_freq(low_freq)
    elif df2_out is None and target_freq is not None:
        high_freq = target_freq
    elif df2_out is not None and target_freq is not None:
        if df2_out.index.inferred_freq != target_freq:
            raise ValueError('User provided target_freq does not match frequency information found on indicator data '
                             'df2.')
        high_freq = target_freq
    else:
        high_freq = df2_out.index.inferred_freq
        if not high_freq:
            raise ValueError('Indicator data df2 does not have a valid time index with frequency information')

    validate_freqs(low_freq, high_freq)

    high_name = get_frequency_name(high_freq)
    low_name = get_frequency_name(low_freq)
    time_conversion_factor = FREQ_CONVERSION_FACTORS[low_name][high_name]

    var_name, low_freq_name, high_freq_name = make_names_from_frequencies(df1_out, high_freq)

    if isinstance(df1_out, pd.Series):
        df1_out.name = low_freq_name
    elif isinstance(df1_out, pd.DataFrame):
        df1_out.rename(columns={var_name: low_freq_name}, inplace=True)

    if df2_out is None and method in ['denton', 'denton-cholette']:
        high_freq_idx = make_companion_index(df1_out, target_freq=high_freq)
        df2_out = pd.Series(1, index=high_freq_idx, name=high_freq_name)

    elif df2_out is None:
        raise ValueError('df2 can only be None for methods "denton" and "denton-cholette", otherwise a dataframe of'
                         'high-frequency indicators must be provided.')
    df, C_mask = align_and_merge_dfs(df1_out, df2_out)

    return df, C_mask, time_conversion_factor


def disaggregate_series(low_freq_df,
                        high_freq_df=None,
                        target_freq=None,
                        target_column=None,
                        agg_func='sum',
                        method='denton-cholette',
                        criterion='proportional',
                        h=1,
                        optimizer_kwargs=None,
                        verbose=True):

    """
    Transform a low frequency time series into a higher frequency series, preserving certain statistics aggregate
    statistics.

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

    Returns
    -------
    high_freq_df: Series
        A Pandas Series object containing the interpolated high frequency data.
    """

    assert method in ['denton', 'denton-cholette', 'chow-lin', 'litterman']
    assert criterion in ['proportional', 'additive']
    assert agg_func in ['mean', 'sum', 'first', 'last']

    target_column = target_column or low_freq_df.columns[0]
    target_idx = np.flatnonzero(low_freq_df.columns == target_column)[0]

    df, C_mask, time_conversion_factor = prepare_input_dataframes(low_freq_df, high_freq_df, target_freq, method)

    y = df.iloc[:, target_idx].dropna().values
    X = df.drop(columns=df.columns[target_idx]).values

    n, k = X.shape
    nl = y.shape[0]

    C = build_conversion_matrix(n, nl, time_conversion_factor, agg_func=agg_func, C_mask=C_mask)
    if method == 'denton':
        assert k == 1
        Σ = build_denton_covariance(n, C, X, h, criterion)
        D = build_distribution_matrix(Σ, C)
        p = X.ravel()

    elif method == 'denton-cholette':
        assert k == 1
        D = build_denton_charlotte_distribution_matrix(n, nl, C, X, h, criterion)
        p = X.ravel()

    else:
        if optimizer_kwargs is None:
            optimizer_kwargs = {'method': 'nelder-mead'}
        if optimizer_kwargs and 'method' not in optimizer_kwargs.keys():
            optimizer_kwargs.update({'method': 'nelder-mead'})

        if method == 'chow-lin':
            f_cov = build_chao_lin_covariance
        elif method == 'litterman':
            f_cov = build_litterman_covariance
        else:
            raise ValueError(f'Method {method} not supported.')

        # betas are unbounded, bound rho between 0 and 1 and sigma between 0 and +inf
        bounds = [(1e-5, 1 - 1e-5), (1e-5, None)]

        x0 = np.full(2, 0.8)
        result = minimize(f_minimize, x0=x0, args=(y, X, C, f_cov), bounds=bounds, **optimizer_kwargs)

        ρ, sigma_e_sq = result.x
        Σ = f_cov(ρ, sigma_e_sq, n)
        Σ_inv = np.linalg.solve(Σ, np.eye(n))

        β = GLS_beta_hat(Σ, y, X, C, nl)
        std_β = np.sqrt(np.diagonal(np.linalg.solve(X.T @ Σ_inv @ X, np.eye(k))))

        if verbose:
            print_regression_report(df.iloc[:, 0], df.iloc[:, 1:], np.r_[β, ρ, sigma_e_sq], std_β, C, method)

        p = X @ β
        D = build_distribution_matrix(Σ, C)

    ul = y - C @ p
    y_hat = p + D @ ul

    output = pd.Series(y_hat, index=df.index, name=target_column)
    output.index.freq = output.index.inferred_freq
    return output
