import numpy as np
import pandas as pd
import pytest

from groupby_lib import GroupBy
from groupby_lib.emas import ema, ema_grouped


class TestGroupByEma:
    """Test suite for GroupBy.ema method."""

    def test_basic_ema_alpha(self):
        """Test basic grouped EMA with alpha parameter."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        # Compute expected using ema_grouped
        expected = ema_grouped(gb.group_ikey, ngroups=2, values=values, alpha=0.5)
        expected_series = pd.Series(expected, index=values.index, name=values.name)

        pd.testing.assert_series_equal(result, expected_series)

    def test_basic_ema_halflife(self):
        """Test basic grouped EMA with halflife parameter."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        gb = GroupBy(key)

        result = gb.ema(values, halflife=2)

        # Compute expected using ema_grouped
        expected = ema_grouped(gb.group_ikey, ngroups=2, values=values, halflife=2)
        expected_series = pd.Series(expected, index=values.index, name=values.name)

        pd.testing.assert_series_equal(result, expected_series)

    def test_ema_with_mask(self):
        """Test EMA with boolean mask."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        mask = np.array([True, True, False, True, True, False])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5, mask=mask)

        expected_masked = gb.ema(values.where(mask), alpha=0.5, mask=mask)
        pd.testing.assert_series_equal(result, expected_masked)

    def test_ema_preserves_index(self):
        """Test that EMA preserves original pandas index."""
        key = pd.Series([1, 1, 2, 2], index=[10, 20, 30, 40])
        values = pd.Series([1.0, 2.0, 10.0, 20.0], index=[10, 20, 30, 40])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        assert result.index.tolist() == [10, 20, 30, 40]

    def test_ema_preserves_series_name(self):
        """Test that EMA preserves Series name."""
        key = pd.Series([1, 1, 2, 2])
        values = pd.Series([1.0, 2.0, 10.0, 20.0], name="price")
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        assert result.name == "price"

    def test_ema_multiple_values(self):
        """Test EMA with multiple value columns."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "col2": [5.0, 6.0, 7.0, 15.0, 25.0, 35.0],
            }
        )
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        # Result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]

        # Check each column
        for col in ["col1", "col2"]:
            expected = ema_grouped(
                gb.group_ikey, ngroups=2, values=values[col], alpha=0.5
            )
            np.testing.assert_array_almost_equal(result[col].values, expected)

    def test_ema_dict_values(self):
        """Test EMA with dictionary of values."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = {
            "a": pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
            "b": pd.Series([5.0, 6.0, 7.0, 15.0, 25.0, 35.0]),
        }
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        # Result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"a", "b"}

        # Check each column
        for col, val_series in values.items():
            expected = ema_grouped(
                gb.group_ikey, ngroups=2, values=val_series, alpha=0.5
            )
            np.testing.assert_array_almost_equal(result[col].values, expected)

    def test_ema_time_weighted(self):
        """Test time-weighted EMA."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        times = pd.date_range("2024-01-01", periods=6, freq="1h")
        gb = GroupBy(key)

        result = gb.ema(values, halflife="2h", times=times)

        # Compute expected using ema_grouped
        expected = ema_grouped(
            gb.group_ikey, ngroups=2, values=values, halflife="2h", times=times
        )
        expected_series = pd.Series(expected, index=values.index, name=values.name)

        pd.testing.assert_series_equal(result, expected_series)

    def test_ema_time_weighted_with_mask(self):
        """Test time-weighted EMA with mask."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        times = pd.date_range("2024-01-01", periods=6, freq="1h")
        mask = np.array([True, True, False, True, False, True])
        gb = GroupBy(key)

        result = gb.ema(values, halflife="2h", times=times, mask=mask)
        expected_masked = gb.ema(
            values.where(mask), halflife="2h", times=times, mask=mask
        )
        pd.testing.assert_series_equal(result, expected_masked)

    def test_ema_time_weighted_tz_aware(self):
        """Test time-weighted EMA with timezone-aware timestamps."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        # Create timezone-aware timestamps
        times = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        gb = GroupBy(key)

        result = gb.ema(values, halflife="2h", times=times)

        # Compute expected using ema_grouped
        expected = ema_grouped(
            gb.group_ikey, ngroups=2, values=values, halflife="2h", times=times
        )
        expected_series = pd.Series(expected, index=values.index, name=values.name)

        pd.testing.assert_series_equal(result, expected_series)

    def test_ema_time_weighted_tz_aware_different_timezones(self):
        """Test time-weighted EMA with different timezone-aware timestamps."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        gb = GroupBy(key)

        # Test with different timezones
        for tz in ["US/Eastern", "Europe/London", "Asia/Tokyo"]:
            times = pd.date_range("2024-01-01", periods=6, freq="1h", tz=tz)

            result = gb.ema(values, halflife="2h", times=times)

            # Compute expected using ema_grouped
            expected = ema_grouped(
                gb.group_ikey, ngroups=2, values=values, halflife="2h", times=times
            )
            expected_series = pd.Series(expected, index=values.index, name=values.name)

            pd.testing.assert_series_equal(result, expected_series)

    def test_ema_time_weighted_tz_aware_with_mask(self):
        """Test time-weighted EMA with timezone-aware timestamps and mask."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        times = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        mask = np.array([True, True, False, True, False, True])
        gb = GroupBy(key)

        result = gb.ema(values, halflife="2h", times=times, mask=mask)
        expected_masked = gb.ema(
            values.where(mask), halflife="2h", times=times, mask=mask
        )
        pd.testing.assert_series_equal(result, expected_masked)

    def test_ema_time_weighted_tz_aware_irregular_intervals(self):
        """Test time-weighted EMA with timezone-aware timestamps at irregular intervals."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

        # Create irregular time intervals
        base_time = pd.Timestamp("2024-01-01", tz="UTC")
        times = pd.Series([
            base_time,
            base_time + pd.Timedelta(minutes=15),
            base_time + pd.Timedelta(hours=2),
            base_time + pd.Timedelta(minutes=5),
            base_time + pd.Timedelta(hours=1),
            base_time + pd.Timedelta(hours=3),
        ])
        gb = GroupBy(key)

        result = gb.ema(values, halflife="1h", times=times)

        # Compute expected using ema_grouped
        expected = ema_grouped(
            gb.group_ikey, ngroups=2, values=values, halflife="1h", times=times
        )
        expected_series = pd.Series(expected, index=values.index, name=values.name)

        pd.testing.assert_series_equal(result, expected_series)

    def test_ema_time_weighted_tz_aware_vs_naive_equivalence(self):
        """Test that TZ-aware and TZ-naive timestamps give equivalent results."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        gb = GroupBy(key)

        # Create TZ-naive timestamps
        times_naive = pd.date_range("2024-01-01", periods=6, freq="1h")
        result_naive = gb.ema(values, halflife="2h", times=times_naive)

        # Create TZ-aware timestamps (same absolute times)
        times_aware = times_naive.tz_localize("UTC")
        result_aware = gb.ema(values, halflife="2h", times=times_aware)

        # Results should be equivalent
        pd.testing.assert_series_equal(result_naive, result_aware)

    def test_ema_single_group(self):
        """Test EMA with only one group."""
        key = pd.Series([1, 1, 1, 1, 1])
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        # Should match regular ema for single group
        expected = ema(values.values, alpha=0.5)
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_ema_interleaved_groups(self):
        """Test EMA with interleaved groups."""
        key = pd.Series([1, 2, 1, 2, 1, 2])
        values = pd.Series([1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.3)

        # Each group's EMA should be independent
        group1_values = values.iloc[[0, 2, 4]].values
        group2_values = values.iloc[[1, 3, 5]].values

        group1_ema = ema(group1_values, alpha=0.3)
        group2_ema = ema(group2_values, alpha=0.3)

        # Check group 1 positions
        np.testing.assert_almost_equal(result.iloc[0], group1_ema[0])
        np.testing.assert_almost_equal(result.iloc[2], group1_ema[1])
        np.testing.assert_almost_equal(result.iloc[4], group1_ema[2])

        # Check group 2 positions
        np.testing.assert_almost_equal(result.iloc[1], group2_ema[0])
        np.testing.assert_almost_equal(result.iloc[3], group2_ema[1])
        np.testing.assert_almost_equal(result.iloc[5], group2_ema[2])

    def test_ema_nan_values(self):
        """Test EMA with NaN values."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1.0, np.nan, 3.0, 10.0, np.nan, 30.0])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        # NaN should propagate the last valid EMA value
        assert not result.isnull().any()

    def test_ema_array_input(self):
        """Test EMA with numpy array input."""
        key = np.array([1, 1, 1, 2, 2, 2])
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5)

        # Result should still be a Series (with default range index)
        assert isinstance(result, pd.Series)
        assert len(result) == len(values)

    def test_ema_validation_errors(self):
        """Test that validation errors are raised properly."""
        key = pd.Series([1, 1, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        gb = GroupBy(key)

        # Missing both alpha and halflife
        with pytest.raises(
            ValueError, match="one of alpha or halflife must be provided"
        ):
            gb.ema(values)

        # Both alpha and halflife provided
        with pytest.raises(ValueError, match="only one of alpha or halflife"):
            gb.ema(values, alpha=0.5, halflife=2)

        # Alpha out of range
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            gb.ema(values, alpha=1.5)

        # Negative halflife
        with pytest.raises(ValueError, match="Halflife must be positive"):
            gb.ema(values, halflife=-1)

    def test_ema_times_validation(self):
        """Test validation with times parameter."""
        key = pd.Series([1, 1, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        times = pd.date_range("2024-01-01", periods=4, freq="1h")
        gb = GroupBy(key)

        # Times without halflife
        with pytest.raises(
            ValueError, match="halflife must be provided when times are given"
        ):
            gb.ema(values, alpha=0.5, times=times)

        # Times length mismatch
        short_times = pd.date_range("2024-01-01", periods=3, freq="1h")
        with pytest.raises(
            ValueError,
            match="group_key, values, times must have equal length. " \
            "Got lengths: {'group_key': 4, 'values': 4, 'times': 3",
        ):
            gb.ema(values, halflife="1h", times=short_times)

    def test_ema_decorator_allows_key_as_first_arg(self):
        """Test that @groupby_method decorator allows passing key as first arg."""
        key = pd.Series([1, 1, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 4.0])

        # Should be able to call directly without creating GroupBy object
        result = GroupBy.ema(key, values, alpha=0.5)

        # Should match calling through GroupBy object
        gb = GroupBy(key)
        expected = gb.ema(values, alpha=0.5)

        pd.testing.assert_series_equal(result, expected)

    def test_ema_empty_groups(self):
        """Test EMA with groups that have no data."""
        key = pd.Series([1, 1, 2, 2])
        values = pd.Series([1.0, 2.0, 10.0, 20.0])
        mask = np.array([True, True, False, False])
        gb = GroupBy(key)

        result = gb.ema(values, alpha=0.5, mask=mask)

        # Group 2 should be all NaN
        assert result.isna()[2]
        assert result.isna()[3]

        # Group 1 should have values
        assert not result.isna()[0]
        assert not result.isna()[1]

    def test_ema_output_type_consistency(self):
        """Test that output type matches input type."""
        key = pd.Series([1, 1, 2, 2])
        gb = GroupBy(key)

        # Single Series -> Series
        values_series = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = gb.ema(values_series, alpha=0.5)
        assert isinstance(result, pd.Series)

        # DataFrame -> DataFrame
        values_df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = gb.ema(values_df, alpha=0.5)
        assert isinstance(result, pd.DataFrame)

        # Dict -> DataFrame
        values_dict = {
            "a": pd.Series([1.0, 2.0, 3.0, 4.0]),
            "b": pd.Series([5.0, 6.0, 7.0, 8.0]),
        }
        result = gb.ema(values_dict, alpha=0.5)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("use_mask", [False, True], ids=["unmasked", "masked"])
    @pytest.mark.parametrize("include_nans", [False, True], ids=["no-nans", "nans"])
    @pytest.mark.parametrize("use_times", [False, True], ids=["not timed", "timed"])
    def test_ema_comparison_with_pandas_ewm(self, use_mask, include_nans, use_times):
        """Test that results match pandas ewm for grouped alpha-based EMA."""
        np.random.seed(147)
        n = 1000
        key = pd.Series(np.random.randint(1, 6, n))
        values = pd.Series(np.random.rand(n))
        if include_nans:
            values = values.where(values > 0.1)

        if use_times:
            base_times = pd.Series(pd.Timestamp.now(), values.index)
            offsets = pd.to_timedelta(np.cumsum(np.random.randint(1, 3, n)), unit="s")
            times = base_times + offsets
            kwargs = dict(times=times, halflife="1s")
        else:
            kwargs = dict(alpha=0.5)

        mask = values > 0.2 if use_mask else None

        # Our implementation (adjust=True)
        result = GroupBy(key).ema(values, **kwargs, index_by_groups=True, mask=mask)

        # Pandas ewm (adjust=True)
        if use_mask:
            values = values.where(mask)
        expected = values.groupby(key).ewm(**kwargs, adjust=True).mean()

        pd.testing.assert_series_equal(result, expected)
