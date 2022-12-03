import unittest
from tsdisagg.ts_disagg import prepare_input_dataframes
from tsdisagg.time_conversion import MONTHS, get_frequency_names, make_companion_index

from hypothesis.strategies import composite, SearchStrategy, integers
from hypothesis import given

import pandas as pd

from typing import Callable, List, Tuple
import calendar


@composite
def freq(draw: Callable[[SearchStrategy[int]], int],
         base: str,
         suffix_list: List[str]) -> Tuple[str, str, str]:

    bases = [f'B{base}', f'{base}S', f'B{base}S']
    suffixes = [f'-{x}' for x in suffix_list] + ['']

    n_bases = len(bases) - 1
    n_suffixes = len(suffixes) - 1

    base_idx = draw(integers(min_value=0, max_value=n_bases))
    suffix_idx = draw(integers(min_value=0, max_value=n_suffixes))

    year = draw(integers(min_value=1900, max_value=2000))
    month = draw(integers(min_value=1, max_value=12))
    day = draw(integers(min_value=1, max_value=calendar.monthrange(year, month)[1]))

    start_date = f'{year}-{month}-{day}'

    base_freq = bases[base_idx]
    suffix = suffixes[suffix_idx]

    return base_freq, suffix, start_date


class TestPandasIndex(unittest.TestCase):

    @given(freq(base='A', suffix_list=MONTHS))
    def test_dataframe_merge(self, params):
        base, suffix, start_date = params
        freq = base + suffix
        target_freq = base.replace('A', 'Q') + suffix

        low_freq_df = pd.Series(1, index=pd.date_range(start_date, freq=freq, periods=20), name='test')
        index = make_companion_index(low_freq_df, target_freq)
        low_freq_name, high_freq_name = get_frequency_names(low_freq_df, target_freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(low_freq_df, high_freq_df, left_index=True, right_index=True, how='outer')

        df, _, _ = prepare_input_dataframes(low_freq_df, None, target_freq, 'denton')
        self.assertEqual(df.shape[0], result.shape[0])

    @given(freq(base='A', suffix_list=MONTHS))
    def test_dataframe_merge_A_to_M(self, params):
        base, suffix, start_date = params
        freq = base + suffix
        target_freq = base.replace('A', 'M')

        low_freq_df = pd.Series(1, index=pd.date_range(start_date, freq=freq, periods=20), name='test')
        index = make_companion_index(low_freq_df, target_freq)
        low_freq_name, high_freq_name = get_frequency_names(low_freq_df, target_freq)

        high_freq_df = pd.Series(1, index=index, name=high_freq_name)
        result = pd.merge(low_freq_df, high_freq_df, left_index=True, right_index=True, how='outer')

        df, _, _ = prepare_input_dataframes(low_freq_df, None, target_freq, 'denton')

        self.assertEqual(df.shape[0], result.shape[0])




if __name__ == '__main__':
    unittest.main()
