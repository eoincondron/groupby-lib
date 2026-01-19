"""
Unit tests for Polars integration with cumulative and rolling GroupBy methods.

This module tests that cumulative and rolling methods return Polars objects when
Polars objects are passed as values. Tests cover different data types (ints, bools,
floats, timestamps, durations) and cases with null entries.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from groupby_lib import GroupBy


class TestPolarsCumulativeMethods:
    """Test cumulative methods (cumsum, cummin, cummax, cumcount) with Polars."""

    @pytest.mark.parametrize("method", ["cumsum", "cummin", "cummax"])
    @pytest.mark.parametrize(
        "dtype,values",
        [
            (pl.Int64, [1, 2, 3, 4, 5, 6, 7, 8]),
            (pl.Float64, [1.5, 2.3, 3.1, 4.8, 5.2, 6.9, 7.1, 8.4]),
            (pl.Boolean, [True, False, True, True, False, True, False, True]),
        ],
    )
    def test_cumulative_basic_types(self, method, dtype, values):
        """Test cumulative methods with basic numeric and boolean types."""
        # Skip boolean for cummin/cummax as they don't make sense
        if dtype == pl.Boolean and method in ("cummin", "cummax"):
            pytest.skip("cummin/cummax not meaningful for boolean")

        # Create test data
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])
        values_series = pl.Series("values", values, dtype=dtype)

        # Calculate with Polars and pandas
        gb = GroupBy(groups)
        result = getattr(gb, method)(values_series)
        result_pd = getattr(gb, method)(values_series.to_pandas())

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series), f"Expected pl.Series, got {type(result)}"
        assert result.name == values_series.name
        assert len(result) == len(values_series)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize("method", ["cumsum", "cummin", "cummax"])
    def test_cumulative_with_nulls(self, method):
        """Test cumulative methods with null values."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])
        values = pl.Series(
            "values", [1.0, None, 3.0, 4.0, None, 6.0, 7.0, None], dtype=pl.Float64
        )

        gb = GroupBy(groups)
        result = getattr(gb, method)(values)
        result_pd = getattr(gb, method)(values.to_pandas())

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name

        assert result_pd.equals(result.to_pandas())

    def test_cumulative_datetime(self):
        """Test cumulative min/max with datetime types."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])

        # Use pandas to create datetime values, then convert to Polars
        dates_pd = pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-02 12:00:00",
            "2024-01-03 00:00:00",
            "2024-01-04 12:00:00",
            "2024-01-05 00:00:00",
            "2024-01-06 12:00:00",
            "2024-01-07 00:00:00",
            "2024-01-08 12:00:00",
        ])
        values_series = pl.from_pandas(pd.Series(dates_pd, name="values"))

        gb = GroupBy(groups)

        # cummin and cummax make sense for datetimes
        for method in ["cummin", "cummax"]:
            result = getattr(gb, method)(values_series)
            result_pd = getattr(gb, method)(values_series.to_pandas())

            # Verify result is a Polars Series
            assert isinstance(result, pl.Series)
            assert result.name == values_series.name
            assert len(result) == len(values_series)

            # Compare with pandas equivalent
            assert result_pd.equals(result.to_pandas())

    def test_cumulative_datetime_tz_aware(self):
        """Test cumulative min/max with timezone-aware datetime types."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])

        # Create timezone-aware datetime values
        dates_pd = pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-02 12:00:00",
            "2024-01-03 00:00:00",
            "2024-01-04 12:00:00",
            "2024-01-05 00:00:00",
            "2024-01-06 12:00:00",
            "2024-01-07 00:00:00",
            "2024-01-08 12:00:00",
        ]).tz_localize("UTC")
        values_series = pl.from_pandas(pd.Series(dates_pd, name="values"))

        gb = GroupBy(groups)

        # cummin and cummax make sense for datetimes
        for method in ["cummin", "cummax"]:
            result = getattr(gb, method)(values_series)
            result_pd = getattr(gb, method)(values_series.to_pandas())

            # Verify result is a Polars Series
            assert isinstance(result, pl.Series)
            assert result.name == values_series.name
            assert len(result) == len(values_series)

            # Compare with pandas equivalent
            assert result_pd.equals(result.to_pandas())

    def test_cumulative_duration(self):
        """Test cumulative min/max with duration types."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])

        # Create timedelta values using pandas, then convert to Polars
        durations_pd = pd.to_timedelta([
            "1h", "2h", "30m", "1h30m", "45m", "2h15m", "1h", "3h"
        ])
        values_series = pl.from_pandas(pd.Series(durations_pd, name="values"))

        gb = GroupBy(groups)

        # cummin and cummax make sense for durations
        for method in ["cummin", "cummax"]:
            result = getattr(gb, method)(values_series)
            result_pd = getattr(gb, method)(values_series.to_pandas())

            # Verify result is a Polars Series
            assert isinstance(result, pl.Series)
            assert result.name == values_series.name
            assert len(result) == len(values_series)

            # Compare with pandas equivalent
            assert result_pd.equals(result.to_pandas())

    def test_cumcount(self):
        """Test cumcount method with Polars."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])

        gb = GroupBy(groups)
        result = gb.cumcount()

        # cumcount returns pandas Series (not Polars) as it doesn't take values
        assert isinstance(result, pd.Series)
        assert len(result) == len(groups)

        # Verify correct counting within groups
        expected = pd.Series([0, 1, 0, 1, 2, 3, 2, 3])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_cumulative_polars_dataframe(self):
        """Test cumulative methods with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B", "A", "A", "B", "B"],
                "val1": [1, 2, 3, 4, 5, 6, 7, 8],
                "val2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )

        gb = GroupBy(df["group"])

        for method in ["cumsum", "cummin", "cummax"]:
            result = getattr(gb, method)(df.select(["val1", "val2"]))
            result_pd = getattr(gb, method)(df.select(["val1", "val2"]).to_pandas())

            # Verify result is a Polars DataFrame
            assert isinstance(result, pl.DataFrame)
            assert result.columns == ["val1", "val2"]
            assert len(result) == len(df)

            # Compare with pandas equivalent
            pd.testing.assert_frame_equal(result_pd, result.to_pandas(), check_dtype=False)


class TestPolarsRollingMethods:
    """Test rolling methods (rolling_sum, rolling_mean, rolling_min, rolling_max) with Polars."""

    @pytest.mark.parametrize(
        "method", ["rolling_sum", "rolling_mean", "rolling_min", "rolling_max"]
    )
    @pytest.mark.parametrize(
        "dtype,values",
        [
            (pl.Int64, [1, 2, 3, 4, 5, 6, 7, 8]),
            (pl.Float64, [1.5, 2.3, 3.1, 4.8, 5.2, 6.9, 7.1, 8.4]),
        ],
    )
    def test_rolling_basic_types(self, method, dtype, values):
        """Test rolling methods with basic numeric types."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])
        values_series = pl.Series("values", values, dtype=dtype)

        gb = GroupBy(groups)
        result = getattr(gb, method)(values_series, window=2)
        result_pd = getattr(gb, method)(values_series.to_pandas(), window=2)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values_series.name
        assert len(result) == len(values_series)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize(
        "method", ["rolling_sum", "rolling_mean", "rolling_min", "rolling_max"]
    )
    def test_rolling_with_nulls(self, method):
        """Test rolling methods with null values."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A", "B", "B"])
        values = pl.Series(
            "values", [1.0, None, 3.0, 4.0, None, 6.0, 7.0, None], dtype=pl.Float64
        )

        gb = GroupBy(groups)
        result = getattr(gb, method)(values, window=2)
        result_pd = getattr(gb, method)(values.to_pandas(), window=2)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize(
        "method", ["rolling_sum", "rolling_mean", "rolling_min", "rolling_max"]
    )
    def test_rolling_with_min_periods(self, method):
        """Test rolling methods with min_periods parameter."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1, 2, 3, 4, 5, 6], dtype=pl.Int64)

        gb = GroupBy(groups)
        result = getattr(gb, method)(values, window=3, min_periods=2)
        result_pd = getattr(gb, method)(values.to_pandas(), window=3, min_periods=2)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    def test_rolling_polars_dataframe(self):
        """Test rolling methods with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B", "A", "A", "B", "B"],
                "val1": [1, 2, 3, 4, 5, 6, 7, 8],
                "val2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )

        gb = GroupBy(df["group"])

        for method in ["rolling_sum", "rolling_mean"]:
            result = getattr(gb, method)(df.select(["val1", "val2"]), window=2)
            result_pd = getattr(gb, method)(df.select(["val1", "val2"]).to_pandas(), window=2)

            # Verify result is a Polars DataFrame
            assert isinstance(result, pl.DataFrame)
            assert result.columns == ["val1", "val2"]
            assert len(result) == len(df)

            # Compare with pandas equivalent
            pd.testing.assert_frame_equal(result_pd, result.to_pandas(), check_dtype=False)

    @pytest.mark.parametrize(
        "method", ["rolling_min", "rolling_max"]
    )
    def test_rolling_datetime(self, method):
        """Test rolling min/max with datetime types."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A"])

        # Use pandas to create datetime values, then convert to Polars
        dates_pd = pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ])
        values = pl.from_pandas(pd.Series(dates_pd, name="values"))

        gb = GroupBy(groups)
        result = getattr(gb, method)(values, window=2)
        result_pd = getattr(gb, method)(values.to_pandas(), window=2)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())


class TestPolarsEMA:
    """Test exponential moving average with Polars."""

    def test_ema_basic_polars_series(self):
        """Test EMA with Polars Series."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=pl.Float64)

        gb = GroupBy(groups)
        result = gb.ema(values, alpha=0.5)
        result_pd = gb.ema(values.to_pandas(), alpha=0.5)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    def test_ema_with_nulls(self):
        """Test EMA with null values."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series(
            "values", [1.0, None, 3.0, 10.0, None, 30.0], dtype=pl.Float64
        )

        gb = GroupBy(groups)
        result = gb.ema(values, alpha=0.5)
        result_pd = gb.ema(values.to_pandas(), alpha=0.5)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    def test_ema_polars_dataframe(self):
        """Test EMA with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "val1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "val2": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            }
        )

        gb = GroupBy(df["group"])
        result = gb.ema(df.select(["val1", "val2"]), alpha=0.5)
        result_pd = gb.ema(df.select(["val1", "val2"]).to_pandas(), alpha=0.5)

        # Verify result is a Polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["val1", "val2"]
        assert len(result) == len(df)

        # Compare with pandas equivalent
        pd.testing.assert_frame_equal(result_pd, result.to_pandas(), check_dtype=False)

    @pytest.mark.parametrize(
        "dtype,values",
        [
            (pl.Int64, [1, 2, 3, 4, 5, 6]),
            (pl.Float32, [1.5, 2.3, 3.1, 4.8, 5.2, 6.9]),
        ],
    )
    def test_ema_different_dtypes(self, dtype, values):
        """Test EMA with different numeric types."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values_series = pl.Series("values", values, dtype=dtype)

        gb = GroupBy(groups)
        result = gb.ema(values_series, alpha=0.5)
        result_pd = gb.ema(values_series.to_pandas(), alpha=0.5)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert len(result) == len(values_series)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    def test_ema_time_weighted_tz_aware(self):
        """Test EMA with timezone-aware timestamps."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=pl.Float64)

        # Create timezone-aware timestamps using pandas, then convert to Polars
        times_pd = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        times = pl.from_pandas(pd.Series(times_pd, name="times"))

        gb = GroupBy(groups)
        result = gb.ema(values, halflife="2h", times=times)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

    def test_ema_time_weighted_tz_aware_different_timezones(self):
        """Test EMA with different timezone-aware timestamps."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=pl.Float64)

        gb = GroupBy(groups)

        # Test with different timezones
        for tz in ["UTC", "US/Eastern", "Europe/London"]:
            times_pd = pd.date_range("2024-01-01", periods=6, freq="1h", tz=tz)
            times = pl.from_pandas(pd.Series(times_pd, name="times"))

            result = gb.ema(values, halflife="2h", times=times)

            # Verify result is a Polars Series
            assert isinstance(result, pl.Series)
            assert result.name == values.name
            assert len(result) == len(values)


class TestPolarsTransformReductions:
    """Test reduction methods with transform=True and Polars."""

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "min", "max", "first", "last"]
    )
    @pytest.mark.parametrize(
        "dtype,values",
        [
            (pl.Int64, [1, 2, 3, 4, 5, 6]),
            (pl.Float64, [1.5, 2.3, 3.1, 4.8, 5.2, 6.9]),
        ],
    )
    def test_transform_basic_types(self, method, dtype, values):
        """Test transform reductions with basic numeric types."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values_series = pl.Series("values", values, dtype=dtype)

        gb = GroupBy(groups)
        result = getattr(gb, method)(values_series, transform=True)
        result_pd = getattr(gb, method)(values_series.to_pandas(), transform=True)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values_series.name
        # Transform should return same length as input
        assert len(result) == len(values_series)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize(
        "method", ["var", "std"]
    )
    def test_transform_variance_std(self, method):
        """Test transform for variance and std with Polars."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=pl.Float64)

        gb = GroupBy(groups)
        result = getattr(gb, method)(values, transform=True)
        result_pd = getattr(gb, method)(values.to_pandas(), transform=True)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "min", "max"]
    )
    def test_transform_with_nulls(self, method):
        """Test transform reductions with null values."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1.0, None, 3.0, 10.0, None, 30.0], dtype=pl.Float64)

        gb = GroupBy(groups)
        result = getattr(gb, method)(values, transform=True)
        result_pd = getattr(gb, method)(values.to_pandas(), transform=True)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "min", "max"]
    )
    def test_transform_dataframe(self, method):
        """Test transform reductions with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "val1": [1, 2, 3, 4, 5, 6],
                "val2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )

        gb = GroupBy(df["group"])
        result = getattr(gb, method)(df.select(["val1", "val2"]), transform=True)
        result_pd = getattr(gb, method)(df.select(["val1", "val2"]).to_pandas(), transform=True)

        # Verify result is a Polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["val1", "val2"]
        assert len(result) == len(df)

        # Compare with pandas equivalent
        pd.testing.assert_frame_equal(result_pd, result.to_pandas(), check_dtype=False)

    def test_transform_multiple_groups(self):
        """Test transform with multiple groups."""
        groups = pl.Series("group", ["A", "A", "B", "B", "C", "C", "A", "B"])
        values = pl.Series("values", [1, 2, 3, 4, 5, 6, 7, 8], dtype=pl.Int64)

        gb = GroupBy(groups)
        result = gb.sum(values, transform=True)
        result_pd = gb.sum(values.to_pandas(), transform=True)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

        # Verify values are correct (each element should be the group sum)
        result_list = result.to_list()
        # Group A (indices 0,1,6): sum = 1+2+7 = 10
        assert result_list[0] == 10
        assert result_list[1] == 10
        assert result_list[6] == 10
        # Group B (indices 2,3,7): sum = 3+4+8 = 15
        assert result_list[2] == 15
        assert result_list[3] == 15
        assert result_list[7] == 15
        # Group C (indices 4,5): sum = 5+6 = 11
        assert result_list[4] == 11
        assert result_list[5] == 11

    @pytest.mark.parametrize(
        "method", ["min", "max"]
    )
    def test_transform_datetime(self, method):
        """Test transform min/max with datetime types."""
        groups = pl.Series("group", ["A", "A", "B", "B", "A", "A"])

        # Use pandas to create datetime values, then convert to Polars
        dates_pd = pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ])
        values = pl.from_pandas(pd.Series(dates_pd, name="values"))

        gb = GroupBy(groups)
        result = getattr(gb, method)(values, transform=True)
        result_pd = getattr(gb, method)(values.to_pandas(), transform=True)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values.name
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    def test_transform_with_mask(self):
        """Test transform with boolean mask."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1, 2, 3, 4, 5, 6], dtype=pl.Int64)
        mask = np.array([True, True, False, True, False, True])

        gb = GroupBy(groups)
        result = gb.sum(values, transform=True, mask=mask)
        result_pd = gb.sum(values.to_pandas(), transform=True, mask=mask)

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())


class TestPolarsShiftDiff:
    """Test shift and diff methods with Polars."""

    @pytest.mark.parametrize("method", ["shift", "diff"])
    @pytest.mark.parametrize(
        "dtype,values",
        [
            (pl.Int64, [1, 2, 3, 4, 5, 6]),
            (pl.Float64, [1.5, 2.3, 3.1, 4.8, 5.2, 6.9]),
        ],
    )
    def test_shift_diff_basic(self, method, dtype, values):
        """Test shift and diff methods with basic types."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values_series = pl.Series("values", values, dtype=dtype)

        gb = GroupBy(groups)
        result = getattr(gb, method)(values_series)
        result_pd = getattr(gb, method)(values_series.to_pandas())

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert result.name == values_series.name
        assert len(result) == len(values_series)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize("method", ["shift", "diff"])
    def test_shift_diff_with_nulls(self, method):
        """Test shift and diff with null values."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1.0, None, 3.0, 10.0, None, 30.0], dtype=pl.Float64)

        gb = GroupBy(groups)
        result = getattr(gb, method)(values)
        result_pd = getattr(gb, method)(values.to_pandas())

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

    @pytest.mark.parametrize("method", ["shift", "diff"])
    def test_shift_diff_dataframe(self, method):
        """Test shift and diff with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "val1": [1, 2, 3, 4, 5, 6],
                "val2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )

        gb = GroupBy(df["group"])
        result = getattr(gb, method)(df.select(["val1", "val2"]))
        result_pd = getattr(gb, method)(df.select(["val1", "val2"]).to_pandas())

        # Verify result is a Polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["val1", "val2"]
        assert len(result) == len(df)

        # Compare with pandas equivalent
        pd.testing.assert_frame_equal(result_pd, result.to_pandas(), check_dtype=False)

    def test_diff_datetime(self):
        """Test diff with datetime types."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])

        # Use pandas to create datetime values, then convert to Polars
        dates_pd = pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ])
        values = pl.from_pandas(pd.Series(dates_pd, name="values"))

        gb = GroupBy(groups)
        result = gb.diff(values)
        result_pd = gb.diff(values.to_pandas())

        # Verify result is a Polars Series
        assert isinstance(result, pl.Series)
        assert len(result) == len(values)

        # Compare with pandas equivalent
        assert result_pd.equals(result.to_pandas())

        # First element of each group should be null
        assert pd.isna(result.to_pandas()[0])
        assert pd.isna(result.to_pandas()[3])


class TestPolarsEdgeCases:
    """Test edge cases and special scenarios with Polars."""

    def test_empty_groups(self):
        """Test with empty groups."""
        groups = pl.Series("group", [], dtype=pl.Utf8)
        values = pl.Series("values", [], dtype=pl.Float64)

        gb = GroupBy(groups)
        result = gb.cumsum(values)

        assert isinstance(result, pl.Series)
        assert len(result) == 0

    def test_single_group(self):
        """Test with single group."""
        groups = pl.Series("group", ["A"] * 6)
        values = pl.Series("values", [1, 2, 3, 4, 5, 6], dtype=pl.Int64)

        gb = GroupBy(groups)

        # Test cumulative
        result_cumsum = gb.cumsum(values)
        result_cumsum_pd = gb.cumsum(values.to_pandas())
        assert isinstance(result_cumsum, pl.Series)
        assert result_cumsum_pd.equals(result_cumsum.to_pandas())
        assert result_cumsum.to_pandas().tolist() == [1, 3, 6, 10, 15, 21]

        # Test rolling
        result_rolling = gb.rolling_sum(values, window=2)
        result_rolling_pd = gb.rolling_sum(values.to_pandas(), window=2)
        assert isinstance(result_rolling, pl.Series)
        assert result_rolling_pd.equals(result_rolling.to_pandas())

    def test_all_nulls(self):
        """Test with all null values."""
        groups = pl.Series("group", ["A", "A", "B", "B"])
        values = pl.Series("values", [None, None, None, None], dtype=pl.Float64)

        gb = GroupBy(groups)
        result = gb.cumsum(values)

        assert isinstance(result, pl.Series)
        # cumsum with skip_na=True (default) treats nulls as 0, so result is 0.0 not null
        # This is consistent with pandas behavior
        result_list = result.to_list()
        assert all(v == 0.0 or pd.isna(v) for v in result_list)

    def test_mixed_null_patterns(self):
        """Test with various null patterns."""
        groups = pl.Series("group", ["A", "A", "A", "B", "B", "B"])
        values = pl.Series("values", [1.0, None, 3.0, None, 5.0, None], dtype=pl.Float64)

        gb = GroupBy(groups)

        # Test cumsum with skip_na
        result = gb.cumsum(values, skip_na=True)
        result_pd = gb.cumsum(values.to_pandas(), skip_na=True)
        assert isinstance(result, pl.Series)
        assert result_pd.equals(result.to_pandas())

        # Test rolling
        result_rolling = gb.rolling_sum(values, window=2)
        result_rolling_pd = gb.rolling_sum(values.to_pandas(), window=2)
        assert isinstance(result_rolling, pl.Series)
        assert result_rolling_pd.equals(result_rolling.to_pandas())
