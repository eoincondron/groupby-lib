"""
Comprehensive tests for timezone-aware timestamp handling in groupby operations.

This test module covers:
1. Timezone-aware timestamps as group keys (monotonic and non-monotonic)
2. Timezone-aware timestamps as input values for aggregations
3. Mixed timezone scenarios
4. Edge cases (NaT, different timezones, DST transitions)
"""

import numpy as np
import pandas as pd
import pytest

from groupby_lib.groupby import SeriesGroupBy


class TestTimezoneAwareGroupKeys:
    """Tests for timezone-aware timestamps used as grouping keys."""

    def test_monotonic_tz_aware_group_keys_us_eastern(self):
        """Test monotonic timezone-aware group keys with US/Eastern timezone."""
        # Create monotonic timezone-aware dates as keys
        dates = pd.date_range('2020-01-01', periods=10, freq='D', tz='US/Eastern')
        values = pd.Series(range(10))

        gb = SeriesGroupBy(values, by=dates)
        result = gb.sum()

        # Each date appears once, so sum should equal the value
        assert len(result) == 10
        assert result.index.tz is not None
        assert 'US/Eastern' in str(result.index.tz)
        pd.testing.assert_series_equal(result, values.set_axis(dates))

    def test_monotonic_tz_aware_group_keys_utc(self):
        """Test monotonic timezone-aware group keys with UTC timezone."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D', tz='UTC')
        values = pd.Series(range(10))

        gb = SeriesGroupBy(values, by=dates)
        result = gb.sum()

        assert len(result) == 10
        assert result.index.tz is not None
        assert str(result.index.tz) == 'UTC'

    def test_monotonic_tz_aware_group_keys_multiple_timezones(self):
        """Test monotonic group keys with different timezones."""
        timezones = ['UTC', 'US/Pacific', 'Europe/London', 'Asia/Tokyo']

        for tz in timezones:
            dates = pd.date_range('2020-01-01', periods=5, freq='D', tz=tz)
            values = pd.Series([1, 2, 3, 4, 5])

            gb = SeriesGroupBy(values, by=dates)
            result = gb.sum()

            assert len(result) == 5, f"Failed for timezone: {tz}"
            assert result.index.tz is not None, f"Lost timezone info for: {tz}"
            pd.testing.assert_series_equal(result, values.set_axis(dates))

    def test_non_monotonic_tz_aware_group_keys(self):
        """Test non-monotonic timezone-aware group keys."""
        # Create dates that repeat (non-monotonic)
        dates = pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', '2020-01-01',
            '2020-01-03', '2020-01-02', '2020-01-01'
        ], tz='US/Eastern')
        values = pd.Series([10, 20, 30, 40, 50, 60])

        gb = SeriesGroupBy(values, by=dates)
        result = gb.sum()

        # Should aggregate by unique dates
        expected = pd.Series(
            [100, 70, 40],  # sum of values for each date
            index=pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'],
                                   tz='US/Eastern')
        )
        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())

    def test_tz_aware_group_keys_with_duplicates(self):
        """Test timezone-aware group keys with many duplicates."""
        dates = pd.DatetimeIndex(['2020-01-01'] * 5 + ['2020-01-02'] * 5, tz='UTC')
        values = pd.Series(range(10))

        gb = SeriesGroupBy(values, by=dates)
        result = gb.sum()

        expected = pd.Series(
            [0+1+2+3+4, 5+6+7+8+9],
            index=pd.DatetimeIndex(['2020-01-01', '2020-01-02'], tz='UTC')
        )
        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())

    def test_tz_aware_group_keys_with_nat(self):
        """Test timezone-aware group keys with NaT (Not a Time) values."""
        dates = pd.DatetimeIndex([
            '2020-01-01', pd.NaT, '2020-01-02', pd.NaT, '2020-01-01'
        ], tz='US/Pacific')
        values = pd.Series([10, 20, 30, 40, 50])

        gb = SeriesGroupBy(values, by=dates)
        result = gb.sum()

        # NaT should be grouped together
        assert len(result) == 3  # 2020-01-01, 2020-01-02, NaT
        assert pd.isna(result.index).sum() == 1  # One NaT in index

    def test_tz_aware_hourly_group_keys(self):
        """Test timezone-aware group keys with hourly frequency."""
        dates = pd.date_range('2020-01-01', periods=24, freq='H', tz='Europe/London')
        values = pd.Series(range(24))

        gb = SeriesGroupBy(values, by=dates)
        result = gb.mean()

        assert len(result) == 24
        pd.testing.assert_series_equal(result, values.set_axis(dates).astype(float))

    def test_tz_aware_group_keys_across_dst_transition(self):
        """Test timezone-aware group keys across DST transition."""
        # US/Eastern has DST transition in March
        dates = pd.date_range('2020-03-07', periods=10, freq='D', tz='US/Eastern')
        values = pd.Series(range(10))

        gb = SeriesGroupBy(values, by=dates)
        result = gb.sum()

        # Should handle DST transition correctly
        assert len(result) == 10
        assert result.index.tz is not None


class TestTimezoneAwareValues:
    """Tests for timezone-aware timestamps as aggregation values."""

    def test_sum_tz_aware_values(self):
        """Test sum aggregation with timezone-aware timestamp values."""
        groups = pd.Series(['A', 'B', 'A', 'B', 'A'])
        # Timezone-aware timestamps as values to aggregate
        values = pd.Series(pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', '2020-01-03',
            '2020-01-04', '2020-01-05'
        ], tz='UTC'))

        gb = SeriesGroupBy(values, by=groups)
        result = gb.sum()

        # Sum of timestamps should work (adds nanoseconds)
        assert len(result) == 2
        assert result.dtype.kind == 'M'  # datetime
        assert 'UTC' in str(result.dtype)

    def test_min_max_tz_aware_values(self):
        """Test min/max aggregations with timezone-aware timestamp values."""
        groups = pd.Series(['A', 'B', 'A', 'B', 'A'])
        values = pd.Series(pd.DatetimeIndex([
            '2020-01-05', '2020-01-02', '2020-01-01',
            '2020-01-04', '2020-01-03'
        ], tz='US/Pacific'))

        gb = SeriesGroupBy(values, by=groups)

        # Test min
        result_min = gb.min()
        assert result_min['A'] == pd.Timestamp('2020-01-01', tz='US/Pacific')
        assert result_min['B'] == pd.Timestamp('2020-01-02', tz='US/Pacific')

        # Test max
        result_max = gb.max()
        assert result_max['A'] == pd.Timestamp('2020-01-05', tz='US/Pacific')
        assert result_max['B'] == pd.Timestamp('2020-01-04', tz='US/Pacific')

    def test_first_last_tz_aware_values(self):
        """Test first/last aggregations with timezone-aware timestamp values."""
        groups = pd.Series(['A', 'B', 'A', 'B'])
        values = pd.Series(pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'
        ], tz='Europe/London'))

        gb = SeriesGroupBy(values, by=groups)

        result_first = gb.first()
        assert result_first['A'] == pd.Timestamp('2020-01-01', tz='Europe/London')
        assert result_first['B'] == pd.Timestamp('2020-01-02', tz='Europe/London')

        result_last = gb.last()
        assert result_last['A'] == pd.Timestamp('2020-01-03', tz='Europe/London')
        assert result_last['B'] == pd.Timestamp('2020-01-04', tz='Europe/London')

    def test_mean_tz_aware_values(self):
        """Test mean aggregation with timezone-aware timestamp values."""
        groups = pd.Series(['A', 'A', 'B', 'B'])
        values = pd.Series(pd.DatetimeIndex([
            '2020-01-01', '2020-01-03', '2020-01-02', '2020-01-04'
        ], tz='Asia/Tokyo'))

        gb = SeriesGroupBy(values, by=groups)
        result = gb.mean()

        # Mean should be the midpoint
        expected_a = pd.Timestamp('2020-01-02', tz='Asia/Tokyo')
        expected_b = pd.Timestamp('2020-01-03', tz='Asia/Tokyo')

        assert result['A'] == expected_a
        assert result['B'] == expected_b

    def test_tz_aware_values_with_nat(self):
        """Test aggregations with timezone-aware values containing NaT."""
        groups = pd.Series(['A', 'B', 'A', 'B', 'A'])
        values = pd.Series(pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', pd.NaT, '2020-01-04', '2020-01-05'
        ], tz='UTC'))

        gb = SeriesGroupBy(values, by=groups)

        # Min should skip NaT
        result_min = gb.min()
        assert result_min['A'] == pd.Timestamp('2020-01-01', tz='UTC')

        # Count should count non-NaT values
        result_count = gb.count()
        assert result_count['A'] == 2  # Has one NaT
        assert result_count['B'] == 2


class TestMixedTimezoneScenarios:
    """Tests for mixed timezone scenarios and edge cases."""

    def test_both_keys_and_values_tz_aware_same_tz(self):
        """Test with both group keys and values as timezone-aware (same timezone)."""
        dates = pd.date_range('2020-01-01', periods=6, freq='D', tz='UTC')
        # Use dates as both keys (by 2-day period) and values
        groups = dates.floor('2D')  # Group by 2-day periods
        values = pd.Series(dates)

        gb = SeriesGroupBy(values, by=groups)
        result = gb.first()

        assert len(result) == 3
        assert result.index.tz is not None
        assert result.dtype.kind == 'M'

    def test_both_keys_and_values_tz_aware_different_tz(self):
        """Test with group keys and values in different timezones."""
        # Keys in US/Eastern
        keys = pd.date_range('2020-01-01', periods=4, freq='D', tz='US/Eastern')
        # Values in UTC
        values = pd.Series(pd.date_range('2020-01-01', periods=4, freq='D', tz='UTC'))

        gb = SeriesGroupBy(values, by=keys)
        result = gb.first()

        # Keys preserve their timezone
        assert 'US/Eastern' in str(result.index.tz)
        # Values preserve their timezone
        assert 'UTC' in str(result.dtype)

    def test_tz_naive_keys_tz_aware_values(self):
        """Test timezone-naive keys with timezone-aware values."""
        keys = pd.date_range('2020-01-01', periods=4, freq='D')  # No timezone
        values = pd.Series(pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'
        ], tz='UTC'))

        gb = SeriesGroupBy(values, by=keys)
        result = gb.first()

        # Values should preserve timezone
        assert 'UTC' in str(result.dtype)

    def test_tz_aware_keys_tz_naive_values(self):
        """Test timezone-aware keys with timezone-naive values."""
        keys = pd.date_range('2020-01-01', periods=4, freq='D', tz='UTC')
        values = pd.Series(pd.date_range('2020-01-01', periods=4, freq='D'))  # No timezone

        gb = SeriesGroupBy(values, by=keys)
        result = gb.first()

        # Keys should preserve timezone in index
        assert result.index.tz is not None
        assert str(result.index.tz) == 'UTC'

    def test_string_keys_tz_aware_values(self):
        """Test string group keys with timezone-aware timestamp values."""
        keys = pd.Series(['A', 'B', 'A', 'B'])
        values = pd.Series(pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'
        ], tz='US/Pacific'))

        gb = SeriesGroupBy(values, by=keys)
        result = gb.mean()

        assert len(result) == 2
        assert 'US/Pacific' in str(result.dtype)


class TestTimezoneAwareEdgeCases:
    """Tests for edge cases with timezone-aware timestamps."""

    def test_empty_tz_aware_data(self):
        """Test with empty timezone-aware data."""
        keys = pd.DatetimeIndex([], tz='UTC')
        values = pd.Series([], dtype='float64')

        gb = SeriesGroupBy(values, by=keys)
        result = gb.sum()

        assert len(result) == 0
        assert result.index.tz is not None

    def test_single_tz_aware_group(self):
        """Test with single timezone-aware group."""
        keys = pd.DatetimeIndex(['2020-01-01'] * 5, tz='UTC')
        values = pd.Series(range(5))

        gb = SeriesGroupBy(values, by=keys)
        result = gb.sum()

        assert len(result) == 1
        assert result.iloc[0] == 10
        assert result.index.tz is not None

    def test_large_tz_aware_dataset(self):
        """Test with large timezone-aware dataset."""
        n = 10000
        dates = pd.date_range('2020-01-01', periods=100, freq='D', tz='UTC')
        keys = np.random.choice(dates, n)
        values = pd.Series(np.random.randn(n))

        gb = SeriesGroupBy(values, by=keys)
        result = gb.mean()

        assert len(result) <= 100
        assert result.index.tz is not None

    def test_tz_aware_with_numeric_values(self):
        """Test timezone-aware group keys with regular numeric values."""
        keys = pd.date_range('2020-01-01', periods=5, freq='D', tz='UTC')
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        gb = SeriesGroupBy(values, by=keys)
        result = gb.sum()

        # Should work normally with numeric values
        assert len(result) == 5
        pd.testing.assert_series_equal(result, values.set_axis(keys))

    def test_timedelta_vs_timestamp_distinction(self):
        """Test that timedeltas are handled separately from timestamps."""
        groups = pd.Series(['A', 'B', 'A', 'B'])

        # Test with timedelta
        timedeltas = pd.Series(pd.to_timedelta(['1 day', '2 days', '3 days', '4 days']))
        gb = SeriesGroupBy(timedeltas, by=groups)
        result_td = gb.sum()
        assert result_td.dtype.kind == 'm'  # timedelta

        # Test with timestamp
        timestamps = pd.Series(pd.DatetimeIndex([
            '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'
        ], tz='UTC'))
        gb = SeriesGroupBy(timestamps, by=groups)
        result_ts = gb.sum()
        assert result_ts.dtype.kind == 'M'  # datetime
        assert 'UTC' in str(result_ts.dtype)
