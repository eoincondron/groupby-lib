import numpy as np
import pandas as pd
import pytest

from groupby_lib.emas import ema, ema_grouped


class TestEmaGrouped:
    """Test suite for ema_grouped function."""

    def test_basic_grouped_ema_alpha(self):
        """Test basic grouped EMA with alpha parameter."""
        groups = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

        result = ema_grouped(groups, 2, values, alpha=0.5)

        # Compute expected: EMA for each group separately
        group0_ema = ema(values[:3], alpha=0.5)
        group1_ema = ema(values[3:], alpha=0.5)
        expected = np.concatenate([group0_ema, group1_ema])

        np.testing.assert_array_almost_equal(result, expected)

    def test_basic_grouped_ema_halflife(self):
        """Test basic grouped EMA with halflife parameter."""
        groups = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

        result = ema_grouped(groups, 2, values, halflife=2)

        # Compute expected: EMA for each group separately
        group0_ema = ema(values[:3], halflife=2)
        group1_ema = ema(values[3:], halflife=2)
        expected = np.concatenate([group0_ema, group1_ema])

        np.testing.assert_array_almost_equal(result, expected)

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        groups = pd.Series([0, 0, 0, 1, 1, 1], index=[10, 20, 30, 40, 50, 60])
        values = pd.Series(
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=[10, 20, 30, 40, 50, 60]
        )

        result = ema_grouped(groups, 2, values, alpha=0.5)

        # Result should be a Series with the same index
        assert isinstance(result, pd.Series)
        assert result.index.tolist() == [10, 20, 30, 40, 50, 60]

        # Values should match numpy array result
        groups_arr = groups.values
        values_arr = values.values
        expected = ema_grouped(groups_arr, 2, values_arr, alpha=0.5)
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_single_group(self):
        """Test with only one group."""
        groups = np.array([0, 0, 0, 0, 0])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = ema_grouped(groups, 1, values, alpha=0.5)
        expected = ema(values, alpha=0.5)

        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_groups(self):
        """Test with multiple interleaved groups."""
        groups = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        values = np.array([1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0])

        result = ema_grouped(groups, 3, values, alpha=0.3)

        # Each group should have EMA computed independently
        # Group 0: [1.0, 2.0, 3.0] at indices [0, 3, 6]
        # Group 1: [10.0, 20.0, 30.0] at indices [1, 4, 7]
        # Group 2: [100.0, 200.0, 300.0] at indices [2, 5, 8]

        group0_ema = ema(np.array([1.0, 2.0, 3.0]), alpha=0.3)
        group1_ema = ema(np.array([10.0, 20.0, 30.0]), alpha=0.3)
        group2_ema = ema(np.array([100.0, 200.0, 300.0]), alpha=0.3)

        np.testing.assert_almost_equal(result[0], group0_ema[0])
        np.testing.assert_almost_equal(result[3], group0_ema[1])
        np.testing.assert_almost_equal(result[6], group0_ema[2])

        np.testing.assert_almost_equal(result[1], group1_ema[0])
        np.testing.assert_almost_equal(result[4], group1_ema[1])
        np.testing.assert_almost_equal(result[7], group1_ema[2])

    def test_nan_handling(self):
        """Test handling of NaN values."""
        groups = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([1.0, np.nan, 3.0, 10.0, np.nan, 30.0])

        result = ema_grouped(groups, 2, values, alpha=0.5)

        # NaN should propagate the last valid EMA value
        assert not np.isnan(result[0])  # First value
        assert not np.isnan(result[1])  # Should carry forward from result[0]
        assert not np.isnan(result[2])  # Valid value

        assert not np.isnan(result[3])  # First value in group 1
        assert not np.isnan(result[4])  # Should carry forward from result[3]
        assert not np.isnan(result[5])  # Valid value

    def test_time_weighted_ema(self):
        """Test time-weighted EMA with times parameter."""
        groups = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        times = pd.date_range("2024-01-01", periods=6, freq="1h")

        result = ema_grouped(groups, 2, values, halflife="2h", times=times)

        # Result should have same length as input
        assert len(result) == len(values)
        assert not np.any(np.isnan(result))

    def test_time_weighted_different_spacing(self):
        """Test time-weighted EMA with irregular time spacing."""
        groups = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        # Irregular spacing
        times = pd.to_datetime(
            [
                "2024-01-01 00:00",
                "2024-01-01 01:00",
                "2024-01-01 03:00",  # 2-hour gap
                "2024-01-01 04:00",
                "2024-01-01 05:00",
                "2024-01-01 08:00",  # 3-hour gap
            ]
        )

        result = ema_grouped(groups, 2, values, halflife="1h", times=times)

        assert len(result) == len(values)
        assert not np.any(np.isnan(result))

    def test_alpha_validation(self):
        """Test alpha parameter validation."""
        groups = np.array([0, 0, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        # Alpha out of range
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ema_grouped(groups, 2, values, alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ema_grouped(groups, 2, values, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            ema_grouped(groups, 2, values, alpha=-0.5)

    def test_halflife_validation(self):
        """Test halflife parameter validation."""
        groups = np.array([0, 0, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        # Negative halflife
        with pytest.raises(ValueError, match="Halflife must be positive"):
            ema_grouped(groups, 2, values, halflife=-1)

        # Zero halflife
        with pytest.raises(ValueError, match="Halflife must be positive"):
            ema_grouped(groups, 2, values, halflife=0)

    def test_parameter_exclusivity(self):
        """Test that alpha and halflife are mutually exclusive."""
        groups = np.array([0, 0, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        # Both provided
        with pytest.raises(ValueError, match="only one of alpha or halflife"):
            ema_grouped(groups, 2, values, alpha=0.5, halflife=2)

        # Neither provided
        with pytest.raises(
            ValueError, match="one of alpha or halflife must be provided"
        ):
            ema_grouped(groups, 2, values)

    def test_length_mismatch(self):
        """Test error when group_key and values have different lengths."""
        groups = np.array([0, 0, 1])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(
            ValueError,
            match="group_key, values must have equal length. "
            "Got lengths: {'group_key': 3, 'values': 4}",
        ):
            ema_grouped(groups, 2, values, alpha=0.5)

    def test_times_length_mismatch(self):
        """Test error when times has different length than values."""
        groups = np.array([0, 0, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        times = pd.date_range("2024-01-01", periods=3, freq="1h")

        with pytest.raises(
            ValueError,
            match="group_key, values, times must have equal length. " \
            "Got lengths: {'group_key': 4, 'values': 4, 'times': 3}",
        ):
            ema_grouped(groups, 2, values, halflife="1h", times=times)

    def test_times_without_halflife(self):
        """Test error when times provided without halflife."""
        groups = np.array([0, 0, 1, 1])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        times = pd.date_range("2024-01-01", periods=4, freq="1h")

        with pytest.raises(
            ValueError, match="halflife must be provided when times are given"
        ):
            ema_grouped(groups, 2, values, alpha=0.5, times=times)

    def test_multidimensional_values_error(self):
        """Test error for multidimensional values."""
        groups = np.array([0, 0, 1, 1])
        values = np.arange(8).reshape(4, 2)  # 2D array

        with pytest.raises(ValueError, match="values must be one-dimensional"):
            ema_grouped(groups, 2, values, alpha=0.5)

    def test_non_integer_groups(self):
        """Test with non-integer group keys."""
        # String groups (will fail - need integers)
        groups = np.array(["a", "a", "b", "b"])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="group_key must be integer"):
            ema_grouped(groups, 2, values, alpha=0.5)

    def test_empty_input(self):
        """Test with empty input arrays."""
        groups = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)

        result = ema_grouped(groups, 0, values, alpha=0.5)
        assert len(result) == 0

    def test_single_value_per_group(self):
        """Test with single value per group."""
        groups = np.array([0, 1, 2, 3])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        result = ema_grouped(groups, 4, values, alpha=0.5)

        # Single values should equal themselves
        np.testing.assert_array_almost_equal(result, values)

    def test_large_number_of_groups(self):
        """Test with many groups."""
        n_groups = 100
        n_per_group = 50

        groups = np.repeat(np.arange(n_groups), n_per_group)
        values = np.random.randn(n_groups * n_per_group)

        result = ema_grouped(groups, 100, values, alpha=0.3)

        assert len(result) == len(values)
        assert not np.any(np.isnan(result))

    def test_consistency_with_single_ema(self):
        """Test that grouped EMA matches single EMA for a single group."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        groups = np.zeros(len(values), dtype=np.int64)

        result_grouped = ema_grouped(groups, 1, values, alpha=0.4)
        result_single = ema(values, alpha=0.4)

        np.testing.assert_array_almost_equal(result_grouped, result_single)

    def test_groups_not_starting_at_zero(self):
        """Test with group IDs that don't start at 0."""
        groups = np.array([5, 5, 5, 10, 10, 10])
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

        result = ema_grouped(groups, 11, values, alpha=0.5)

        # Should work - will allocate space for groups 0-10
        assert len(result) == len(values)
        assert not np.any(np.isnan(result))
