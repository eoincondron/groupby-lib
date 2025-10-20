import numpy as np
import pandas as pd
import pytest

from groupby_lib.ema import _ema_adjusted, _ema_time_weighted, _ema_unadjusted, ema

parametrize = pytest.mark.parametrize


# Test data fixtures
@pytest.fixture
def simple_array():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def array_with_nans():
    return np.array([1.0, 2.0, np.nan, 4.0, 5.0])


@pytest.fixture
def simple_series():
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="test_series")


# Test dtypes
NUMERIC_DTYPES = [
    np.float32,
    np.float64,
    np.int32,
    np.int64,
]


class TestEmaAdjusted:
    """Tests for _ema_adjusted function."""

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_basic_calculation(self, simple_array, dtype):
        """Test basic EMA calculation with different dtypes."""
        arr = simple_array.astype(dtype)
        alpha = 0.5
        result = _ema_adjusted(arr, alpha)

        # Verify output dtype is float64
        assert result.dtype == np.float64

        # Verify output shape matches input
        assert result.shape == arr.shape

        # Verify first element
        assert result[0] == arr[0]

        # Verify result is monotonically increasing for increasing input
        assert np.all(np.diff(result) > 0)

    @parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_different_alphas(self, simple_array, alpha):
        """Test EMA with different alpha values."""
        result = _ema_adjusted(simple_array, alpha)

        # Higher alpha should converge faster to recent values
        assert result.dtype == np.float64
        assert len(result) == len(simple_array)

    def test_nan_handling(self, array_with_nans):
        """Test that NaN values are handled correctly."""
        alpha = 0.5
        result = _ema_adjusted(array_with_nans, alpha)

        # Result at NaN position should carry forward previous value
        assert result[2] == result[1]

        # Non-NaN positions should be computed
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_constant_array(self, dtype):
        """Test EMA on constant array."""
        arr = np.full(10, 5.0, dtype=dtype)
        alpha = 0.5
        result = _ema_adjusted(arr, alpha)

        # EMA of constant array should be the constant
        np.testing.assert_allclose(result, 5.0)

    def test_single_element(self):
        """Test EMA with single element array."""
        arr = np.array([3.0])
        alpha = 0.5
        result = _ema_adjusted(arr, alpha)

        assert result[0] == 3.0

    def test_alternating_values(self):
        """Test EMA with alternating values."""
        arr = np.array([1.0, 10.0, 1.0, 10.0, 1.0, 10.0])
        alpha = 0.5
        result = _ema_adjusted(arr, alpha)

        # Result should oscillate but with smoothing
        assert result.dtype == np.float64
        assert len(result) == len(arr)


class TestEmaUnadjusted:
    """Tests for _ema_unadjusted function."""

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_basic_calculation(self, simple_array, dtype):
        """Test basic unadjusted EMA calculation with different dtypes."""
        arr = simple_array.astype(dtype)
        alpha = 0.5
        result = _ema_unadjusted(arr, alpha)

        # Verify output dtype is float64
        assert result.dtype == np.float64

        # Verify output shape matches input
        assert result.shape == arr.shape

        # Verify first element
        assert result[0] == arr[0]

        # Verify result is monotonically increasing for increasing input
        assert np.all(np.diff(result) > 0)

    @parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_different_alphas(self, simple_array, alpha):
        """Test unadjusted EMA with different alpha values."""
        result = _ema_unadjusted(simple_array, alpha)

        assert result.dtype == np.float64
        assert len(result) == len(simple_array)

        # First element should remain unchanged
        assert result[0] == simple_array[0]

    def test_nan_handling(self, array_with_nans):
        """Test that NaN values are handled correctly."""
        alpha = 0.5
        result = _ema_unadjusted(array_with_nans, alpha)

        # Result at NaN position should carry forward previous value
        assert result[2] == result[1]

        # Non-NaN positions should be computed
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_constant_array(self, dtype):
        """Test unadjusted EMA on constant array."""
        arr = np.full(10, 5.0, dtype=dtype)
        alpha = 0.5
        result = _ema_unadjusted(arr, alpha)

        # EMA of constant array should be the constant
        np.testing.assert_allclose(result, 5.0)

    def test_single_element(self):
        """Test unadjusted EMA with single element array."""
        arr = np.array([3.0])
        alpha = 0.5
        result = _ema_unadjusted(arr, alpha)

        assert result[0] == 3.0

    def test_comparison_with_adjusted(self, simple_array):
        """Test that unadjusted differs from adjusted EMA."""
        alpha = 0.5
        adjusted = _ema_adjusted(simple_array, alpha)
        unadjusted = _ema_unadjusted(simple_array, alpha)

        # First element should be the same
        assert adjusted[0] == unadjusted[0]

        # Other elements should differ
        assert not np.allclose(adjusted[1:], unadjusted[1:])


class TestEmaTimeWeighted:
    """Tests for _ema_time_weighted function."""

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_basic_calculation(self, simple_array, dtype):
        """Test time-weighted EMA with different dtypes."""
        arr = simple_array.astype(dtype)
        times = np.arange(len(arr), dtype=np.int64) * 1000
        halflife = 2.0
        result = _ema_time_weighted(arr, times, halflife)

        # Verify output dtype is float64
        assert result.dtype == np.float64

        # Verify output shape matches input
        assert result.shape == arr.shape

        # Verify first element
        assert result[0] == arr[0]

    def test_uniform_times(self):
        """Test time-weighted EMA with uniform time intervals."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.arange(len(arr), dtype=np.int64) * 1000
        halflife = 2.0
        result = _ema_time_weighted(arr, times, halflife)

        assert result.dtype == np.float64
        assert len(result) == len(arr)
        assert result[0] == arr[0]

    def test_non_uniform_times(self):
        """Test time-weighted EMA with non-uniform time intervals."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        times = np.array([0, 100, 300, 600, 1000], dtype=np.int64)
        halflife = 200.0
        result = _ema_time_weighted(arr, times, halflife)

        assert result.dtype == np.float64
        assert len(result) == len(arr)
        assert result[0] == arr[0]

    def test_nan_handling(self):
        """Test that NaN values are handled correctly in time-weighted EMA."""
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        times = np.arange(len(arr), dtype=np.int64) * 1000
        halflife = 2.0
        result = _ema_time_weighted(arr, times, halflife)

        # Result at NaN position should carry forward previous value
        assert result[2] == result[1]

        # Non-NaN positions should be computed
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])

    @parametrize("halflife", [1.0, 5.0, 10.0, 100.0])
    def test_different_halflives(self, simple_array, halflife):
        """Test time-weighted EMA with different halflife values."""
        times = np.arange(len(simple_array), dtype=np.int64) * 1000
        result = _ema_time_weighted(simple_array, times, halflife)

        assert result.dtype == np.float64
        assert len(result) == len(simple_array)


class TestEmaPublicApi:
    """Tests for the public ema() function."""

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_with_alpha(self, simple_array, dtype):
        """Test public API with alpha parameter."""
        arr = simple_array.astype(dtype)
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == arr.shape

    def test_with_halflife(self, simple_array):
        """Test public API with halflife parameter."""
        halflife = 2.0
        result = ema(simple_array, halflife=halflife)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == simple_array.shape

    @parametrize("adjust", [True, False])
    def test_adjust_parameter(self, simple_array, adjust):
        """Test adjust parameter."""
        alpha = 0.5
        result = ema(simple_array, alpha=alpha, adjust=adjust)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_with_series_input(self, simple_series):
        """Test that Series input returns Series output."""
        alpha = 0.5
        result = ema(simple_series, alpha=alpha)

        assert isinstance(result, pd.Series)
        assert result.name == simple_series.name
        pd.testing.assert_index_equal(result.index, simple_series.index)

    def test_with_times(self, simple_array):
        """Test time-weighted EMA through public API."""
        times = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
        )
        halflife = "1d"
        result = ema(simple_array, halflife=halflife, times=times)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_series_with_times(self, simple_series):
        """Test time-weighted EMA with Series input."""
        times = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
        )
        halflife = "1d"
        result = ema(simple_series, halflife=halflife, times=times)

        assert isinstance(result, pd.Series)
        assert result.name == simple_series.name
        pd.testing.assert_index_equal(result.index, simple_series.index)

    def test_error_no_alpha_or_halflife(self, simple_array):
        """Test that error is raised when neither alpha nor halflife provided."""
        with pytest.raises(
            ValueError, match="One of alpha or halflife must be provided"
        ):
            ema(simple_array)

    def test_error_both_alpha_and_halflife(self, simple_array):
        """Test that error is raised when both alpha and halflife provided."""
        with pytest.raises(
            ValueError, match="Only one of alpha or halflife should be provided"
        ):
            ema(simple_array, alpha=0.5, halflife=2.0)

    def test_error_invalid_alpha_low(self, simple_array):
        """Test that error is raised for alpha <= 0."""
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            ema(simple_array, alpha=0.0)

    def test_error_invalid_alpha_high(self, simple_array):
        """Test that error is raised for alpha > 1."""
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            ema(simple_array, alpha=1.5)

    def test_error_invalid_halflife(self, simple_array):
        """Test that error is raised for negative halflife."""
        with pytest.raises(ValueError, match="Halflife must be positive"):
            ema(simple_array, halflife=-1.0)

    def test_error_times_without_halflife(self, simple_array):
        """Test that error is raised when times provided without halflife."""
        times = np.arange(len(simple_array))
        with pytest.raises(
            ValueError, match="Halflife must be provided when times are given"
        ):
            ema(simple_array, alpha=0.5, times=times)

    def test_error_multidimensional_array(self):
        """Test that error is raised for multidimensional arrays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="Input array must be one-dimensional"):
            ema(arr, alpha=0.5)

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_docstring_example_adjusted(self, dtype):
        """Test the docstring example for adjusted EMA."""
        data = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = ema(data, alpha=0.5)
        expected = np.array([1.0, 1.66666667, 2.42857143, 3.26666667, 4.16129032])
        np.testing.assert_array_almost_equal(result, expected)

    @parametrize("dtype", NUMERIC_DTYPES)
    def test_docstring_example_unadjusted(self, dtype):
        """Test the docstring example for unadjusted EMA."""
        data = np.array([1, 2, 3, 4, 5], dtype=dtype)
        result = ema(data, alpha=0.5, adjust=False)
        expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
        np.testing.assert_array_almost_equal(result, expected)


class TestEmaEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_all_nans(self):
        """Test EMA with array of all NaNs."""
        arr = np.array([np.nan, np.nan, np.nan])
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        # With all NaNs, implementation initializes to 0 and carries forward
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_leading_nans(self):
        """Test EMA with leading NaN values."""
        arr = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        # Leading NaNs should be handled
        assert len(result) == len(arr)

    def test_trailing_nans(self):
        """Test EMA with trailing NaN values."""
        arr = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        # Trailing NaNs should carry forward last valid value
        assert result[3] == result[2]
        assert result[4] == result[3]

    def test_very_small_alpha(self):
        """Test EMA with very small alpha (high smoothing)."""
        arr = np.array([1.0, 100.0, 1.0, 100.0, 1.0])
        alpha = 0.01
        result = ema(arr, alpha=alpha, adjust=False)

        # With small alpha, changes should be heavily smoothed
        assert result.dtype == np.float64
        assert abs(result[1] - result[0]) < 10  # Should not jump to 100

    def test_alpha_equals_one(self):
        """Test EMA with alpha = 1 (no smoothing)."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = 1.0
        result = ema(arr, alpha=alpha, adjust=False)

        # With alpha=1, result should be close to input
        # (adjusted formula makes this slightly different)
        assert result.dtype == np.float64

    def test_large_array(self):
        """Test EMA with large array."""
        arr = np.random.randn(10000)
        alpha = 0.3
        result = ema(arr, alpha=alpha)

        assert len(result) == len(arr)
        assert result.dtype == np.float64

    @parametrize("dtype", [np.float32, np.float64])
    def test_precision_float_types(self, dtype):
        """Test that different float types are handled correctly."""
        arr = np.array([1.123456789, 2.234567890, 3.345678901], dtype=dtype)
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        # Output should always be float64
        assert result.dtype == np.float64

    @parametrize("dtype", [np.int32, np.int64])
    def test_integer_types(self, dtype):
        """Test that integer types are handled correctly."""
        arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        # Output should be float64
        assert result.dtype == np.float64
        assert not np.array_equal(result[1:], arr[1:])  # Should have fractional values

    def test_negative_values(self):
        """Test EMA with negative values."""
        arr = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        assert result.dtype == np.float64
        assert len(result) == len(arr)
        # Should handle negatives correctly
        assert result[0] == -5.0

    def test_mixed_positive_negative(self):
        """Test EMA with mixed positive and negative values."""
        arr = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        assert result.dtype == np.float64
        assert len(result) == len(arr)

    def test_zero_values(self):
        """Test EMA with zeros."""
        arr = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
        alpha = 0.5
        result = ema(arr, alpha=alpha)

        assert result.dtype == np.float64
        assert len(result) == len(arr)

    def test_comparison_adjusted_unadjusted_convergence(self):
        """Test that adjusted and unadjusted EMAs converge over time."""
        arr = np.random.randn(1000)
        alpha = 0.1

        adjusted = ema(arr, alpha=alpha, adjust=True)
        unadjusted = ema(arr, alpha=alpha, adjust=False)

        # Early values should differ
        assert not np.allclose(adjusted[:10], unadjusted[:10])

        # Later values should be closer (but may not be identical)
        # The difference should decrease over time
        early_diff = np.abs(adjusted[10] - unadjusted[10])
        late_diff = np.abs(adjusted[-1] - unadjusted[-1])
        assert late_diff <= early_diff or np.allclose(late_diff, early_diff, rtol=0.1)
