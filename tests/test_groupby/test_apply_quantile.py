import numpy as np
import pandas as pd
import pytest

from groupby_lib.groupby.core import GroupBy

from .conftest import assert_pd_equal


class TestApply:
    """Test suite for GroupBy.apply method."""

    def test_apply_basic_scalar_function(self):
        """Test apply with a function that returns scalars."""
        key = pd.Series([1, 1, 2, 2, 3, 3])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy(key)

        # Apply range function (max - min)
        result = gb.apply(values, lambda x: np.max(x) - np.min(x))
        expected = pd.Series([10, 10, 10], index=[1, 2, 3])

        assert_pd_equal(result, expected)

    def test_apply_with_mask(self):
        """Test apply with a boolean mask."""
        key = pd.Series([1, 1, 2, 2, 3, 3])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        mask = np.array([True, True, True, False, True, True])
        gb = GroupBy(key)

        # Apply sum, but only on masked values
        result = gb.apply(values, np.sum, mask=mask)
        expected = pd.Series([30, 30, 110], index=[1, 2, 3])

        assert_pd_equal(result, expected)

    def test_apply_with_numpy_func(self):
        """Test apply with standard numpy functions."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy(key)

        # Test with np.mean
        result = gb.apply(values, np.mean)
        expected = pd.Series([20.0, 50.0], index=[1, 2])
        assert_pd_equal(result, expected)

        # Test with np.std (note: np.std uses ddof=0 by default, unlike pandas which uses ddof=1)
        result = gb.apply(values, np.std)
        # Compute expected using np.std directly on each group
        expected = pd.Series([np.std([10, 20, 30]), np.std([40, 50, 60])], index=[1, 2])
        assert_pd_equal(result, expected)

    def test_apply_with_func_kwargs(self):
        """Test apply with function that takes keyword arguments."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy(key)

        # Use np.percentile with q parameter
        result = gb.apply(values, np.percentile, q=75)
        expected = pd.Series([25.0, 55.0], index=[1, 2])
        assert_pd_equal(result, expected)

    def test_apply_with_func_args_and_kwargs(self):
        """Test apply with function that takes both args and kwargs together."""
        key = pd.Series([1, 1, 2, 2])
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        gb = GroupBy(key)

        # Test with both keyword arguments working together
        result = gb.apply(values, np.percentile, q=50)
        expected = pd.Series([1.5, 3.5], index=[1, 2])
        assert_pd_equal(result, expected)

    def test_apply_with_multiple_values(self):
        """Test apply with multiple value columns."""
        key = pd.Series([1, 1, 2, 2])
        values1 = pd.Series([10, 20, 30, 40])
        values2 = pd.Series([1, 2, 3, 4])
        gb = GroupBy(key)

        result = gb.apply([values1, values2], np.mean)
        # Check shape and structure
        expected = pd.DataFrame(
            [[15.0, 1.5], [35.0, 3.5]], index=[1, 2], columns=["_arr_0", "_arr_1"]
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_with_dict_values(self):
        """Test apply with dictionary of values."""
        key = pd.Series([1, 1, 2, 2])
        values = {"a": pd.Series([10, 20, 30, 40]), "b": pd.Series([1, 2, 3, 4])}
        gb = GroupBy(key)

        result = gb.apply(values, np.sum)
        expected = pd.DataFrame([[30, 3], [70, 7]], index=[1, 2], columns=["a", "b"])
        assert_pd_equal(result, expected)

    def test_apply_array_returning_function(self):
        """Test apply with function that returns arrays."""
        key = pd.Series([1, 1, 2, 2])
        values = pd.Series([10, 20, 30, 40])
        gb = GroupBy(key)

        # Function that returns array with min and max
        result = gb.apply(values, lambda x: np.array([np.min(x), np.max(x)]))

        expected = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=[1, 2],
            columns=pd.MultiIndex.from_product([["_arr_0"], [0, 1]]),
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_categorical_keys(self):
        """Test apply with categorical keys."""
        key = pd.Categorical(["a", "a", "b", "b"], categories=["b", "a", "c"])
        values = pd.Series([10, 20, 30, 40])
        gb = GroupBy(key)

        result = gb.apply(values, np.mean)
        # Check values - the exact index type may vary
        expected = pd.Series([35., 15.], ["b", "a"])
        pd.testing.assert_series_equal(result, expected)

    def test_apply_with_string_keys(self):
        """Test apply with string keys."""
        key = pd.Series(["apple", "apple", "banana", "banana"])
        values = pd.Series([1, 2, 3, 4])
        gb = GroupBy(key)

        result = gb.apply(values, np.sum)
        expected = pd.Series([3, 7], index=["apple", "banana"])
        assert_pd_equal(result, expected)

    def test_apply_single_group(self):
        """Test apply when all values belong to single group."""
        key = pd.Series([1, 1, 1, 1])
        values = pd.Series([10, 20, 30, 40])
        gb = GroupBy(key)

        result = gb.apply(values, np.mean)
        expected = pd.Series([25.0], index=[1])
        assert_pd_equal(result, expected)

    def test_apply_empty_group_filtered_by_mask(self):
        """Test apply when mask filters out entire groups."""
        key = pd.Series([1, 1, 2, 2, 3, 3])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        mask = np.array([True, True, True, True, False, False])
        gb = GroupBy(key)

        result = gb.apply(values, np.sum, mask=mask)
        # Group 3 should not appear
        assert len(result) == 2
        assert 1 in result.index
        assert 2 in result.index
        assert 3 not in result.index

    def test_apply_multikey(self):
        """Test apply with multiple grouping keys."""
        key1 = pd.Series([1, 1, 2, 2])
        key2 = pd.Series(["a", "b", "a", "b"])
        values = pd.Series([10, 20, 30, 40])
        gb = GroupBy([key1, key2])

        result = gb.apply(values, np.sum)
        # Check that result is correct - index may be regular Index with tuples, not MultiIndex
        assert len(result) == 4
        # Access using tuple keys
        assert result.iloc[0] == 10
        assert result.iloc[1] == 20
        assert result.iloc[2] == 30
        assert result.iloc[3] == 40

    def test_apply_with_custom_function(self):
        """Test apply with custom user-defined function."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([1, 2, 3, 4, 5, 6])
        gb = GroupBy(key)

        # Custom function: coefficient of variation
        def cv(x):
            return np.std(x) / np.mean(x) if np.mean(x) != 0 else 0

        result = gb.apply(values, cv)
        assert len(result) == 2
        assert result.index.tolist() == [1, 2]


class TestQuantile:
    """Test suite for GroupBy.quantile method."""

    def test_quantile_basic_single(self):
        """Test quantile with a single quantile value."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.5])
        # With a single quantile and single value column, result is a Series
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert np.isclose(result.iloc[0], 20.0)
        assert np.isclose(result.iloc[1], 50.0)

    def test_quantile_basic_multiple(self):
        """Test quantile with multiple quantile values."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.25, 0.5, 0.75])
        # Check shape and index
        assert result.shape == (2, 3)
        assert result.index.tolist() == [1, 2]
        # Columns may have MultiIndex with value name and quantile
        assert result.columns.nlevels == 2
        # Check quantiles are in the second level
        assert 0.25 in result.columns.get_level_values(1)
        assert 0.5 in result.columns.get_level_values(1)
        assert 0.75 in result.columns.get_level_values(1)

        # Check approximate values (using iloc for robust access)
        assert np.isclose(result.iloc[0, 1], 20.0)  # Group 1, median
        assert np.isclose(result.iloc[1, 1], 50.0)  # Group 2, median

    def test_quantile_with_mask(self):
        """Test quantile with a boolean mask."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        mask = np.array([True, True, False, True, True, True])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.5], mask=mask)
        # Group 1 only has [10, 20], median = 15
        # Group 2 has all [40, 50, 60], median = 50
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert np.isclose(result.iloc[0], 15.0)
        assert np.isclose(result.iloc[1], 50.0)

    def test_quantile_extreme_values(self):
        """Test quantile at extreme values (0 and 1)."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.0, 1.0])
        # q=0 should be min, q=1 should be max
        # Result is a DataFrame with 2 quantile columns
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        assert result.iloc[0, 0] == 10  # Group 1, q=0.0
        assert result.iloc[0, 1] == 30  # Group 1, q=1.0
        assert result.iloc[1, 0] == 40  # Group 2, q=0.0
        assert result.iloc[1, 1] == 60  # Group 2, q=1.0

    def test_quantile_with_multiple_values(self):
        """Test quantile with multiple value columns."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values1 = pd.Series([10, 20, 30, 40, 50, 60])
        values2 = pd.Series([1, 2, 3, 4, 5, 6])
        gb = GroupBy(key)

        result = gb.quantile([values1, values2], q=[0.5])
        # Should have MultiIndex columns
        assert result.columns.nlevels == 2
        assert result.shape == (2, 2)

    def test_quantile_with_dict_values(self):
        """Test quantile with dictionary of values."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = {
            "a": pd.Series([10, 20, 30, 40, 50, 60]),
            "b": pd.Series([1, 2, 3, 4, 5, 6]),
        }
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.5])
        # Should have MultiIndex columns with value names and quantiles
        assert result.columns.nlevels == 2
        assert "a" in result.columns.get_level_values(0)
        assert "b" in result.columns.get_level_values(0)

    def test_quantile_categorical_keys(self):
        """Test quantile with categorical keys."""
        key = pd.Categorical(["a", "a", "a", "b", "b", "b"], categories=["a", "b", "c"])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.5])
        # Single quantile returns Series
        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert np.isclose(result.iloc[0], 20.0)
        assert np.isclose(result.iloc[1], 50.0)

    def test_quantile_quartiles(self):
        """Test quantile computing quartiles."""
        key = pd.Series([1] * 10)
        values = pd.Series(range(10))  # 0-9
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.25, 0.5, 0.75])
        # Check that we get reasonable quartile values
        assert result.shape == (1, 3)
        assert 0 <= result.iloc[0, 0] <= result.iloc[0, 1] <= result.iloc[0, 2] <= 9

    def test_quantile_single_value_per_group(self):
        """Test quantile when each group has only one value."""
        key = pd.Series([1, 2, 3])
        values = pd.Series([10, 20, 30])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.25, 0.5, 0.75])
        # All quantiles should equal the single value in each group
        assert result.shape == (3, 3)
        assert result.iloc[0, 1] == 10  # Group 1, median
        assert result.iloc[1, 1] == 20  # Group 2, median
        assert result.iloc[2, 1] == 30  # Group 3, median

    def test_quantile_empty_group_with_mask(self):
        """Test quantile when mask filters out entire groups."""
        key = pd.Series([1, 1, 2, 2, 3, 3])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        mask = np.array([True, True, True, True, False, False])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.5], mask=mask)
        # Group 3 should not appear (single quantile returns Series)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_quantile_multikey(self):
        """Test quantile with multiple grouping keys."""
        key1 = pd.Series([1, 1, 1, 2, 2, 2])
        key2 = pd.Series(["a", "a", "a", "b", "b", "b"])
        values = pd.Series([10, 20, 30, 40, 50, 60])
        gb = GroupBy([key1, key2])

        result = gb.quantile(values, q=[0.5])
        # Single quantile returns Series
        assert isinstance(result, pd.Series)
        # Index may be regular Index with tuples, not MultiIndex
        assert len(result) == 2
        assert np.isclose(result.iloc[0], 20.0)
        assert np.isclose(result.iloc[1], 50.0)

    def test_quantile_with_nan_handling(self):
        """Test quantile behavior with NaN values."""
        key = pd.Series([1, 1, 1, 2, 2, 2])
        values = pd.Series([10, 20, np.nan, 40, 50, 60])
        gb = GroupBy(key)

        result = gb.quantile(values, q=[0.5])
        # np.percentile returns nan if any value is nan (by default)
        assert isinstance(result, pd.Series)
        assert len(result) == 2
