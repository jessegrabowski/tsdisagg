import calendar
import unittest
from typing import Callable, List, Tuple

import pandas as pd
from hypothesis import given
from hypothesis.strategies import SearchStrategy, composite, integers

from tsdisagg.time_conversion import (
    FREQ_CONVERSION_FACTORS,
    MONTHS,
    get_frequency_names,
    handle_endpoint_differences,
    make_companion_index,
)
from tsdisagg.ts_disagg import build_conversion_matrix, prepare_input_dataframes


@composite
def freq(
    draw: Callable[[SearchStrategy[int]], int], base: str, suffix_list: List[str]
) -> Tuple[str, str, str]:

    bases = [f"{base}E", f"B{base}E", f"{base}S", f"B{base}S"]
    suffixes = [f"-{x}" for x in suffix_list] + [""]

    n_bases = len(bases) - 1
    n_suffixes = len(suffixes) - 1

    base_idx = draw(integers(min_value=0, max_value=n_bases))
    suffix_idx = draw(integers(min_value=0, max_value=n_suffixes))

    year = draw(integers(min_value=1900, max_value=2000))
    month = draw(integers(min_value=1, max_value=12))
    day = draw(integers(min_value=1, max_value=calendar.monthrange(year, month)[1]))

    start_date = f"{year}-{month}-{day}"

    base_freq = bases[base_idx]
    suffix = suffixes[suffix_idx]

    return base_freq, suffix, start_date


class TestPandasIndex(unittest.TestCase):
    @given(freq(base="Y", suffix_list=MONTHS))
    def test_dataframe_merge(self, params):
        base, suffix, start_date = params
        freq = base + suffix
        target_freq = base.replace("Y", "Q") + suffix

        low_freq_df = pd.Series(
            1, index=pd.date_range(start_date, freq=freq, periods=20), name="test"
        )
        index = make_companion_index(low_freq_df, target_freq)
        low_freq_name, high_freq_name = get_frequency_names(low_freq_df, target_freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(low_freq_df, high_freq_df, left_index=True, right_index=True, how="outer")

        df, _, _ = prepare_input_dataframes(low_freq_df, None, target_freq, "denton")
        self.assertEqual(df.shape[0], result.shape[0])

    @given(freq(base="Y", suffix_list=MONTHS))
    def test_dataframe_merge_A_to_M(self, params):
        base, suffix, start_date = params
        freq = base + suffix
        target_freq = base.replace("Y", "M")

        low_freq_df = pd.Series(
            1, index=pd.date_range(start_date, freq=freq, periods=20), name="test"
        )
        index = make_companion_index(low_freq_df, target_freq)
        low_freq_name, high_freq_name = get_frequency_names(low_freq_df, target_freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(low_freq_df, high_freq_df, left_index=True, right_index=True, how="outer")

        df, _, _ = prepare_input_dataframes(low_freq_df, None, target_freq, "denton")

        self.assertEqual(df.shape[0], result.shape[0])

    def test_dataframe_merge_Q_to_M(self):
        freq = "QS"
        target_freq = "MS"
        start_date = "1900-01-01"

        low_freq_df = pd.Series(
            1, index=pd.date_range(start_date, freq=freq, periods=20), name="test"
        ).iloc[:-2]

        index = make_companion_index(low_freq_df, target_freq)
        low_freq_name, high_freq_name = get_frequency_names(low_freq_df, target_freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(low_freq_df, high_freq_df, left_index=True, right_index=True, how="outer")

        df, C_mask, factor = prepare_input_dataframes(low_freq_df, None, target_freq, "denton")
        self.assertEqual(df.shape[0], result.shape[0])


"""
 In my test case, low-frequency quarterly data starts on 1995/6/1, and ends on 2001/12/1; high-frequency
 monthly data starts on 1995/3/1, and ends on 2001/12/1. The program will show up the following error:
"""


@given(freq(base="Y", suffix_list=MONTHS), integers(min_value=1, max_value=5))
def test_handle_endpoint_differences_Y_to_M(params, offset):
    base, suffix, lf_start_date = params
    freq = base + suffix
    target_freq = base.replace("Y", "M")
    FREQ_CONVERSION_FACTORS["yearly"]["monthly"]

    hf_delta = pd.DateOffset(months=offset)
    # op = np.add if op == 0 else np.subtract
    hf_start_date = (pd.Timestamp(lf_start_date) + hf_delta).strftime("%Y-%m-%d")
    end_date = (pd.Timestamp(lf_start_date) + pd.DateOffset(years=20)).strftime(
        "%Y-%m-%d"
    )

    low_freq_df = pd.Series(
        1,
        index=pd.date_range(start=lf_start_date, end=end_date, freq=freq),
        name="low_freq",
    )
    high_freq_df = pd.Series(
        1,
        index=pd.date_range(start=hf_start_date, end=end_date, freq=target_freq),
        name="high_freq",
    )

    df = pd.merge(
        low_freq_df, high_freq_df, left_index=True, right_index=True, how="outer"
    )
    print(df.head(20).to_string())

    # n = high_freq_df.shape[0]
    # nl = low_freq_df.shape[0]
    #
    C_mask = handle_endpoint_differences(low_freq_df, high_freq_df)
    print((~C_mask).sum())

    # print(C_mask)
    # n_masked = (~C_mask).sum()
    #
    # excess = n - time_conversion_factor * nl
    # extra_periods = int(np.ceil(excess / time_conversion_factor))
    # print(extra_periods, n_masked)
    # assert n_masked == extra_periods


def test_build_conversion_matrix(params):
    freq = "QS-OCT"
    target_freq = "MS"
    lf_start_date = "1995-06-01"
    hf_start_date = "1995-03-01"
    end_date = "2001-12-01"

    low_freq_df = pd.Series(
        1, index=pd.date_range(lf_start_date, end_date, freq=freq), name="low_freq"
    )
    high_freq_df = pd.Series(
        1,
        index=pd.date_range(hf_start_date, end_date, freq=target_freq),
        name="low_freq",
    )

    df, C_mask, time_conversion_factor = prepare_input_dataframes(
        low_freq_df, high_freq_df, target_freq, "denton"
    )

    y = df.iloc[:, 0].dropna().values
    X = df.drop(columns=df.columns[0]).values
    n, k = X.shape
    nl = y.shape[0]

    build_conversion_matrix(
        n, nl, time_conversion_factor, agg_func="sum", C_mask=C_mask
    )


if __name__ == "__main__":
    unittest.main()
