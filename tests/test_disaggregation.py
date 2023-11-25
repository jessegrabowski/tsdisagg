import calendar
import unittest
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
from hypothesis import given
from hypothesis.strategies import SearchStrategy, composite, integers

from tsdisagg import disaggregate_series
from tsdisagg.time_conversion import MONTHS


class DisaggregationTests(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)
        self.exports_m = pd.read_csv("tests/data/exports_m.csv", index_col=0)
        self.exports_m.index = pd.date_range(
            start="1972-01-01", freq="MS", periods=self.exports_m.shape[0]
        )
        self.exports_m.columns = ["exports"]

        self.sales_a = pd.read_csv("tests/data/sales_a.csv", index_col=0)
        self.sales_a.index = pd.date_range(
            start="1975-01-01", freq="YS", periods=self.sales_a.shape[0]
        )
        self.sales_a.columns = ["sales"]

        self.exports_q = pd.read_csv("tests/data/exports_q.csv", index_col=0)
        self.exports_q.index = pd.date_range(
            start="1972-01-01", freq="QS-OCT", periods=self.exports_q.shape[0]
        )
        self.exports_q.columns = ["exports"]

        self.imports_q = pd.read_csv("tests/data/imports_q.csv", index_col=0)
        self.imports_q.index = pd.date_range(
            start="1972-01-01", freq="QS-OCT", periods=self.exports_q.shape[0]
        )
        self.imports_q.columns = ["imports"]

    def test_chow_lin(self):
        expected = pd.read_csv("tests/data/R_output_chow_lin.csv", index_col=0)
        expected.index = self.exports_q.index
        expected.columns = ["sales"]

        sales_q_chow_lin = disaggregate_series(
            self.sales_a,
            self.exports_q.assign(constant=1),
            method="chow-lin",
            agg_func="sum",
            optimizer_kwargs={"method": "powell"},
            verbose=False,
        )

        self.assertEqual(expected, sales_q_chow_lin.to_frame())

    def test_chow_lin_two_indicator(self):
        expected = pd.read_csv("tests/data/R_output_chow_lin_two_indicator.csv", index_col=0)
        expected.index = self.exports_q.index
        expected.columns = ["sales"]

        df2 = self.exports_q.merge(self.imports_q, left_index=True, right_index=True)

        sales_q_chow_lin = disaggregate_series(
            self.sales_a,
            df2.resample("QS-OCT").first().assign(constant=1),
            method="chow-lin",
            agg_func="sum",
            optimizer_kwargs={"method": "l-bfgs-b"},
            verbose=True,
        )

        self.assertEqual(expected, sales_q_chow_lin.to_frame())

    def test_denton(self):
        expected = pd.read_csv("tests/data/R_output_denton.csv", index_col=0)
        expected.index = pd.date_range(start="1975-01-01", freq="QS-OCT", periods=144)
        expected.columns = ["sales"]

        sales_q_denton = disaggregate_series(
            self.sales_a, method="denton", agg_func="sum", optimizer_kwargs={"method": "powell"}
        )
        self.assertEqual(expected, sales_q_denton.to_frame())

    def test_denton_cholette_w_constant(self):
        expected = pd.read_csv("tests/data/R_output_denton_cholette.csv", index_col=0)
        expected.index = pd.date_range(start="1975-01-01", freq="QS-OCT", periods=144)
        expected.columns = ["sales"]

        sales_q_dc = disaggregate_series(
            self.sales_a,
            method="denton-cholette",
            agg_func="sum",
            optimizer_kwargs={"method": "powell"},
        )

        self.assertEqual(expected, sales_q_dc.to_frame())

    def test_denton_cholette_w_indicator(self):
        expected = pd.read_csv("tests/data/R_output_denton_cholette_w_indicator.csv", index_col=0)
        expected.index = self.exports_q.index
        expected.columns = ["sales"]

        sales_q_dc = disaggregate_series(
            self.sales_a,
            high_freq_df=self.exports_q,
            method="denton-cholette",
            agg_func="sum",
            optimizer_kwargs={"method": "powell"},
            verbose=False,
        )

        self.assertEqual(expected, sales_q_dc.to_frame())

    def test_litterman_A_to_M(self):
        expected = pd.read_csv("tests/data/R_output_litterman_A_to_M.csv", index_col=0)
        expected.index = self.exports_m.index
        expected.columns = ["sales"]

        sales_m_litterman = disaggregate_series(
            self.sales_a,
            high_freq_df=self.exports_m.assign(Constant=1),
            method="litterman",
            agg_func="sum",
            optimizer_kwargs={"method": "nelder-mead"},
            verbose=False,
        )

        self.assertEqual(expected, sales_m_litterman.to_frame())


@composite
def freq(
    draw: Callable[[SearchStrategy[int]], int], base: str, suffix_list: List[str]
) -> Tuple[str, str, str]:
    bases = [f"{base}"] if base == "A" else [f"B{base}", f"{base}S", f"B{base}S"]
    suffixes = [""] if base == "M" else [f"-{x}" for x in suffix_list] + [""]

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


if __name__ == "__main__":
    unittest.main()
