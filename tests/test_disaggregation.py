import unittest

from collections.abc import Callable

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from hypothesis import given
from hypothesis.strategies import SearchStrategy, composite, integers

from tsdisagg import disaggregate_series
from tsdisagg.time_conversion import FREQ_CONVERSION_FACTORS, MONTHS, get_frequency_name
from tsdisagg.ts_disagg import build_conversion_matrix


def generate_random_index_pair(
    low_freq="YS",
    high_freq="QS",
    low_periods=5,
    high_periods=None,
    start=None,
    extra_high_freq_start=0,
    extra_high_freq_end=0,
):
    """
    Generate a random (low_frequency, high_frequency) DatetimeIndex tuple.

    Parameters
    ----------
    low_freq : str, default 'A'
        The frequency for the low-frequency DatetimeIndex (e.g., 'A' for yearly).
    high_freq : str, default 'Q'
        The frequency for the high-frequency DatetimeIndex (e.g., 'Q' for quarterly).
    low_periods : int, default 5
        Number of periods for the low-frequency DatetimeIndex.
    high_periods : int, optional
        Number of periods for the high-frequency DatetimeIndex. If None, it will be calculated based on the
        low-frequency range.
    start : str or pd.Timestamp, optional
        The start date for the index. If None, a random start date within a reasonable range will be chosen.

    Returns
    -------
    tuple of (pd.DatetimeIndex, pd.DatetimeIndex)
        A tuple containing the low-frequency and high-frequency DatetimeIndexes.
    """
    if start is None:
        rng = np.random.default_rng()
        start = pd.Timestamp.now().normalize() - pd.DateOffset(years=rng.integers(0, 30))

    low_freq_index = pd.date_range(start=start, periods=low_periods, freq=low_freq)
    high_freq = pd._libs.tslibs.to_offset(high_freq)

    high_start = low_freq_index[0] + high_freq * extra_high_freq_start
    high_end = low_freq_index[-1] + high_freq * extra_high_freq_end

    if high_periods is None:
        high_periods = len(pd.date_range(start=high_start, end=high_end, freq=high_freq))

    high_freq_index = pd.date_range(start=high_start, periods=max(0, high_periods), freq=high_freq)

    return (
        pd.Series(1.0, index=low_freq_index, name="low_freq"),
        pd.Series(1.0, index=high_freq_index, name="high_freq"),
    )


@composite
def frequencies(draw: Callable[[SearchStrategy[int]], int]) -> tuple[str, str]:
    base_choices = ["Y", "Q", "M"]
    prefixes = ["", "B"]
    suffixes = ["E", "S"]

    low_base_idx = draw(integers(min_value=0, max_value=1))
    high_base_idx = draw(integers(min_value=low_base_idx + 1, max_value=2))
    prefix_idx = draw(integers(min_value=0, max_value=1))
    suffix_idx = draw(integers(min_value=0, max_value=1))

    month_idx = draw(integers(min_value=0, max_value=11))
    month = MONTHS[month_idx]

    low_base = base_choices[low_base_idx]
    high_base = base_choices[high_base_idx]

    low_freq = prefixes[prefix_idx] + low_base + suffixes[suffix_idx]
    high_freq = prefixes[prefix_idx] + high_base + suffixes[suffix_idx]

    if low_base in ["Y", "Q"]:
        low_freq += "-" + month
    if high_base == "Q":
        high_freq += "-" + month

    return low_freq, high_freq


@given(frequencies())
@pytest.mark.parametrize("agg_func", ["sum", "mean", "first", "last"])
def test_build_C_matrix(agg_func, frequencies):
    low_freq, high_freq = frequencies
    df_low, df_high = generate_random_index_pair(low_freq, high_freq, extra_high_freq_start=0, extra_high_freq_end=0)

    high_name = get_frequency_name(high_freq)
    low_name = get_frequency_name(low_freq)
    time_conversion_factor = FREQ_CONVERSION_FACTORS[low_name][high_name]

    C = build_conversion_matrix(df_low, df_high, time_conversion_factor, agg_func)

    assert C.shape[0] == df_low.shape[0]
    assert C.shape[1] == df_high.shape[0]

    if agg_func == "sum":
        np.testing.assert_allclose(C.sum(axis=1).max(), time_conversion_factor)
    else:
        np.testing.assert_allclose(C.sum(axis=1).max(), 1.0)

    high_agg = C @ df_high.values

    high_with_info = df_high.to_frame().assign(
        year=lambda x: x.index.year,
        quarter=lambda x: x.index.quarter,
        month=lambda x: x.index.month,
    )

    if low_name == "yearly":
        groups = high_with_info.groupby("year")  # .high_freq.agg(agg_func).values
        group_size = groups.size()
        full_size = 4 if high_name == "quarterly" else 12
        full_size_mask = group_size.values == full_size
        expected_result = groups.high_freq.agg(agg_func).values
        np.testing.assert_allclose(high_agg[full_size_mask], expected_result[full_size_mask])

    elif low_name == "quarterly":
        groups = high_with_info.groupby(["year", "quarter"])  # .high_freq.agg(agg_func).values
        group_size = groups.size()
        full_size_mask = group_size.values == 3
        expected_result = groups.high_freq.agg(agg_func).values
        np.testing.assert_allclose(high_agg[full_size_mask], expected_result[full_size_mask])


class DisaggregationTests(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)
        self.exports_m = pd.read_csv("tests/data/exports_m.csv", index_col=0)
        self.exports_m.index = pd.date_range(start="1972-01-01", freq="MS", periods=self.exports_m.shape[0])
        self.exports_m.columns = ["exports"]

        self.sales_a = pd.read_csv("tests/data/sales_a.csv", index_col=0)
        self.sales_a.index = pd.date_range(start="1975-01-01", freq="YS", periods=self.sales_a.shape[0])
        self.sales_a.columns = ["sales"]

        self.exports_q = pd.read_csv("tests/data/exports_q.csv", index_col=0)
        self.exports_q.index = pd.date_range(start="1972-01-01", freq="QS-OCT", periods=self.exports_q.shape[0])
        self.exports_q.columns = ["exports"]

        self.imports_q = pd.read_csv("tests/data/imports_q.csv", index_col=0)
        self.imports_q.index = pd.date_range(start="1972-01-01", freq="QS-OCT", periods=self.exports_q.shape[0])
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

    def test_chow_lin_backcasting_error(self):
        # Test case for quarterly to monthly disaggregation backcasting error, issue #6
        expected = pd.read_csv("tests/data/R_Output_chow-lin_QtoM_2.csv")

        low_freq_data = pd.read_csv("tests/data/AL_Quarterly_Data_Modified.csv")
        high_freq_data = pd.read_csv("tests/data/AL_Monthly_Data_Modified_Shorter.csv")

        low_freq_data.index = pd.to_datetime(low_freq_data["period"])
        high_freq_data.index = pd.to_datetime(high_freq_data["period"])

        low_freq_data = low_freq_data.dropna()
        high_freq_data = high_freq_data.dropna()

        low_freq_data = low_freq_data.drop(["period"], axis=1)
        high_freq_data = high_freq_data.drop(["period"], axis=1)

        low_freq_data.index.freq = low_freq_data.index.inferred_freq
        high_freq_data.index.freq = high_freq_data.index.inferred_freq

        expected.index = high_freq_data.index
        expected.columns = ["Value"]

        m_chow_lin = disaggregate_series(
            low_freq_data,
            high_freq_data.assign(intercept=1),
            method="chow-lin",
            agg_func="first",
            optimizer_kwargs={"method": "powell"},
        )

        assert np.all(expected.index == m_chow_lin.index)
        np.testing.assert_allclose(expected.values.ravel(), m_chow_lin.values, rtol=1e-3)

    def test_chow_lin_backcasting_error_YtoQ(self):
        expected = pd.read_csv("tests/data/AL_A_to_Q_expected.csv")
        expected["index"] = (
            expected["index"]
            .str.replace(" Q", "-")
            .map(lambda x: pd.Period(year=int(x.split("-")[0]), quarter=int(x.split("-")[-1]), freq="Q").start_time)
        )

        expected = expected.set_index("index").resample("QS-DEC").last()
        expected.index = expected.index + expected.index.freq

        low_freq_data = pd.read_csv(
            "tests/data/AL_Annual_Data_Shorter.csv",
            parse_dates=True,
            index_col="period",
        ).dropna()
        low_freq_data.index.freq = low_freq_data.index.inferred_freq
        high_freq_data = pd.read_csv(
            "tests/data/AL_Quarterly_Data_Modified.csv",
            parse_dates=True,
            index_col="period",
        ).dropna()
        high_freq_data.index.freq = high_freq_data.index.inferred_freq

        q_chow_lin, res = disaggregate_series(
            low_freq_data,
            high_freq_data.assign(intercept=1),
            method="chow-lin",
            agg_func="first",
            optimizer_kwargs={"method": "powell"},
            return_optim_res=True,
        )

        assert res.success
        assert np.all(expected.index == q_chow_lin.index)
        np.testing.assert_allclose(expected.values.ravel(), q_chow_lin.values.ravel(), rtol=1e-3)

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

    def test_chow_lin_no_freq(self):
        expected = pd.read_csv("tests/data/R_output_chow_lin_two_indicator.csv", index_col=0)
        expected.index = self.exports_q.index
        expected.columns = ["sales"]

        df2 = self.exports_q.merge(self.imports_q, left_index=True, right_index=True)

        sales_nofreq = self.sales_a.copy()
        sales_nofreq.index.freq = None
        assert sales_nofreq.index.freq is None

        sales_q_chow_lin = disaggregate_series(
            sales_nofreq,
            df2.resample("QS-OCT").first().assign(constant=1),
            method="chow-lin",
            agg_func="sum",
            optimizer_kwargs={"method": "l-bfgs-b"},
            verbose=True,
        )

        self.assertEqual(expected, sales_q_chow_lin.to_frame())
        assert sales_q_chow_lin.index.freq == "QS-OCT"

    def test_denton(self):
        expected = pd.read_csv("tests/data/R_output_denton.csv", index_col=0)
        expected.index = pd.date_range(start="1975-01-01", freq="QS-OCT", periods=144)
        expected.columns = ["sales"]

        sales_q_denton = disaggregate_series(
            self.sales_a,
            method="denton",
            agg_func="sum",
            optimizer_kwargs={"method": "powell"},
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


def test_invalid_dataframe_warnings():
    with pytest.raises(
        TypeError,
        match="No datetime index found on the dataframe passed as argument to low_freq_df",
    ):
        disaggregate_series(
            pd.DataFrame({"data": [1, 2, 3]}),
            pd.DataFrame({"data": [1, 2, 3]}),
            method="denton",
            agg_func="sum",
        )

    with pytest.raises(
        ValueError,
        match="No datetime index found on the dataframe passed as argument to high_freq_df",
    ):
        disaggregate_series(
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            pd.DataFrame({"data": [1, 2, 3]}),
            method="denton",
            agg_func="sum",
        )

    with pytest.raises(ValueError, match="low_freq_df has missing values"):
        disaggregate_series(
            pd.DataFrame({"data": [1, np.nan, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            method="denton",
            agg_func="sum",
        )

    with pytest.raises(ValueError, match="high_freq_df has missing values"):
        disaggregate_series(
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            pd.DataFrame({"data": [1, np.nan, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            method="denton",
            agg_func="sum",
        )

    with pytest.raises(
        ValueError,
        match="Start date found on high frequency data 2020-01-01 is after start date "
        "found on low frequency data 1999-01-01.",
    ):
        disaggregate_series(
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("1999-01-01", periods=3, freq="D")),
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            method="denton",
            agg_func="sum",
        )

    with pytest.raises(
        ValueError,
        match="User provided target_freq does not match frequency information found on " "indicator data.",
    ):
        disaggregate_series(
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")),
            method="denton",
            agg_func="sum",
            target_freq="M",
        )

    with pytest.raises(
        ValueError,
        match="Indicator data high_freq_df does not have a valid time index with " "frequency information",
    ):
        disaggregate_series(
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="ME")),
            pd.DataFrame(
                {"data": [1, 2, 3]},
                index=pd.to_datetime(["2020-01-01", "2020-03-04", "2020-12-06"]),
            ),
            method="denton",
            agg_func="sum",
        )

    with pytest.raises(ValueError, match='high_freq_df can only be None for methods "denton" and "denton-cholette"'):
        disaggregate_series(
            pd.DataFrame({"data": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="QE")),
            None,
            method="litterman",
            agg_func="sum",
        )


if __name__ == "__main__":
    unittest.main()
