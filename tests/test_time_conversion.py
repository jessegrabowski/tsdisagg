import unittest
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis.strategies import SearchStrategy, composite, integers

from tsdisagg.time_conversion import (
    MONTHS,
    auto_step_down_base_freq,
    get_frequency_names,
    make_companion_index,
)


@composite
def freq(
    draw: Callable[[SearchStrategy[int]], int], base: str, suffix_list: List[str]
) -> Tuple[str, str, bool]:

    bases = [f"{base}E", f"B{base}E", f"{base}S", f"B{base}S"]
    suffixes = [f"-{x}" for x in suffix_list]

    n_bases = len(bases) - 1
    n_suffixes = len(suffixes) - 1

    base_idx = draw(integers(min_value=0, max_value=n_bases))
    suffix_idx = draw(integers(min_value=0, max_value=n_suffixes))
    to_df = bool(draw(integers(0, 1)))

    return bases[base_idx], suffixes[suffix_idx], to_df


class TestCompanionIndex(unittest.TestCase):
    @given(freq(base="Y", suffix_list=MONTHS))
    def test_yearly_dataframe_to_monthly(self, params):
        base, suffix, test_df = params
        freq = base + suffix

        df = pd.Series(1, index=pd.date_range("1900-01-01", "1902-01-01", freq=freq), name="test")
        if test_df:
            df = df.to_frame()

        T = df.shape[0]

        target_freq = base.replace("Y", "M")
        index = make_companion_index(df, target_freq)
        low_freq_name, high_freq_name = get_frequency_names(df, target_freq)

        self.assertEqual(target_freq, index.freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(df, high_freq_df, left_index=True, right_index=True, how="outer")

        self.assertEqual(result.shape[0], T * 12)

        block_matrix = result.values[:, 0].reshape(T, 12)
        self.assertEqual(np.all(np.isnan(block_matrix).sum(axis=1) == 11), True)

    @given(freq(base="Y", suffix_list=MONTHS))
    def test_yearly_to_quarterly(self, params):
        base, suffix, test_df = params
        freq = base + suffix

        df = pd.Series(1, index=pd.date_range("1900-01-01", "1902-01-01", freq=freq), name="test")
        if test_df:
            df = df.to_frame()

        T = df.shape[0]

        target_freq = base.replace("Y", "Q") + suffix
        index = make_companion_index(df, target_freq)
        self.assertEqual(target_freq, index.freq)

        low_freq_name, high_freq_name = get_frequency_names(df, target_freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(df, high_freq_df, left_index=True, right_index=True, how="outer")
        self.assertEqual(result.shape[0], T * 4)

        block_matrix = result.values[:, 0].reshape(T, 4)
        self.assertEqual(np.all(np.isnan(block_matrix).sum(axis=1) == 3), True)

    @given(freq(base="Q", suffix_list=MONTHS))
    def test_quarterly_to_monthly(self, params):
        base, suffix, test_df = params
        freq = base + suffix

        df = pd.Series(1, index=pd.date_range("1900-01-01", "1902-01-01", freq=freq), name="test")
        if test_df:
            df = df.to_frame()

        T = df.shape[0]
        target_freq = base.replace("Q", "M")
        index = make_companion_index(df, target_freq)
        self.assertEqual(target_freq, index.freq)

        low_freq_name, high_freq_name = get_frequency_names(df, target_freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(df, high_freq_df, left_index=True, right_index=True, how="outer")

        self.assertEqual(result.shape[0], T * 3)

        block_matrix = result.values[:, 0].reshape(T, 3)
        self.assertEqual(np.all(np.isnan(block_matrix).sum(axis=1) == 2), True)


class TestUtilities(unittest.TestCase):
    def test_step_down_frequency(self):
        freq = "YS-JAN"

        auto_freq = auto_step_down_base_freq(freq)
        self.assertEqual(auto_freq, "QS-JAN")

        freq = "BQE-MAR"
        auto_freq = auto_step_down_base_freq(freq)
        self.assertEqual(auto_freq, "BME")

        freq = "MS"
        self.assertRaises(NotImplementedError, auto_step_down_base_freq, freq)


if __name__ == "__main__":
    unittest.main()
