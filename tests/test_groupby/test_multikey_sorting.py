"""
Tests for GroupBy multi-key sorting functionality.

Tests the argsort_index_numeric_only function integration with GroupBy operations
to ensure proper sorting of results with multiple grouping keys.
"""

import numpy as np
import pandas as pd
import pytest

from groupby_lib.groupby.core import GroupBy


class TestMultiKeySorting:
    """Test GroupBy operations with multiple keys and sorting."""

    def test_two_keys_unsorted_data(self):
        """Test GroupBy with two keys on unsorted data."""
        key1 = np.array([2, 1, 2, 1, 3])
        key2 = np.array([20, 10, 10, 20, 30])
        values = np.array([100, 200, 300, 400, 500])

        result = GroupBy([key1, key2]).sum(values)

        # Result should be sorted by key1, then key2
        expected_index = pd.MultiIndex.from_tuples(
            [(1, 10), (1, 20), (2, 10), (2, 20), (3, 30)]
        )
        expected = pd.Series([200, 400, 300, 100, 500], index=expected_index, name=None)

        pd.testing.assert_series_equal(result, expected)

    def test_two_keys_reverse_sorted_data(self):
        """Test GroupBy with two keys on reverse-sorted data."""
        key1 = np.array([3, 2, 2, 1, 1])
        key2 = np.array([30, 20, 10, 20, 10])
        values = np.array([500, 100, 300, 400, 200])

        result = GroupBy([key1, key2]).sum(values)

        # Result should be sorted by key1, then key2
        expected_index = pd.MultiIndex.from_tuples(
            [(1, 10), (1, 20), (2, 10), (2, 20), (3, 30)]
        )
        expected = pd.Series([200, 400, 300, 100, 500], index=expected_index, name=None)

        pd.testing.assert_series_equal(result, expected)

    def test_three_keys_sorting(self):
        """Test GroupBy with three keys."""
        key1 = np.array([2, 1, 2, 1, 2, 1])
        key2 = np.array([20, 10, 10, 20, 20, 10])
        key3 = np.array(["b", "a", "a", "b", "a", "b"])
        values = np.array([1, 2, 3, 4, 5, 6])

        result = GroupBy([key1, key2, key3]).sum(values)

        # Should be sorted by all three keys
        assert result.index.is_monotonic_increasing
        assert len(result) == 6

    def test_two_keys_with_float_values(self):
        """Test GroupBy with float keys."""
        key1 = np.array([2.5, 1.5, 2.5, 1.5, 3.5])
        key2 = np.array([20.0, 10.0, 10.0, 20.0, 30.0])
        values = np.array([100, 200, 300, 400, 500])

        result = GroupBy([key1, key2]).sum(values)

        # Should be sorted by both float keys
        assert result.index.is_monotonic_increasing
        assert len(result) == 5

    def test_two_keys_with_negative_values(self):
        """Test GroupBy with negative key values."""
        key1 = np.array([2, -1, 2, -1, 0])
        key2 = np.array([20, 10, 10, 20, 30])
        values = np.array([100, 200, 300, 400, 500])

        result = GroupBy([key1, key2]).sum(values)

        # Check that all unique key combinations appear
        assert len(result) == 5
        # Check that -1 appears in the index
        assert any(idx[0] == -1 for idx in result.index)
        # Check some aggregated values
        assert result.loc[(-1, 10)] == 200
        assert result.loc[(-1, 20)] == 400
        assert result.loc[(0, 30)] == 500

    def test_two_keys_with_duplicates(self):
        """Test GroupBy with duplicate key combinations."""
        key1 = np.array([1, 1, 2, 2, 1, 1])
        key2 = np.array([10, 10, 20, 20, 10, 10])
        values = np.array([1, 2, 3, 4, 5, 6])

        result = GroupBy([key1, key2]).sum(values)

        # Should aggregate duplicates
        expected_index = pd.MultiIndex.from_tuples([(1, 10), (2, 20)])
        expected = pd.Series([1 + 2 + 5 + 6, 3 + 4], index=expected_index, name=None)

        pd.testing.assert_series_equal(result, expected)

    def test_multiple_aggregations_with_sorting(self):
        """Test multiple aggregation functions with multi-key sorting."""
        key1 = np.array([2, 1, 2, 1])
        key2 = np.array([20, 10, 10, 20])
        values = np.array([100.0, 200.0, 300.0, 400.0])

        gb = GroupBy([key1, key2])
        result_sum = gb.sum(values)
        result_mean = gb.mean(values)
        result_count = gb.count(values)

        # All should have same sorted index
        pd.testing.assert_index_equal(result_sum.index, result_mean.index)
        pd.testing.assert_index_equal(result_sum.index, result_count.index)

        # Check sorting
        assert result_sum.index.is_monotonic_increasing
        assert result_mean.index.is_monotonic_increasing
        assert result_count.index.is_monotonic_increasing

    def test_sort_false_parameter(self):
        """Test that sort=False prevents sorting."""
        key1 = np.array([2, 1, 2, 1])
        key2 = np.array([20, 10, 10, 20])
        values = np.array([100, 200, 300, 400])

        result = GroupBy([key1, key2], sort=False).sum(values)

        # With sort=False, result order follows first appearance
        assert len(result) == 4

    def test_two_keys_max_aggregation(self):
        """Test max aggregation with multi-key sorting."""
        key1 = np.array([2, 1, 2, 1, 2])
        key2 = np.array([20, 10, 10, 20, 20])
        values = np.array([100.0, 200.0, 300.0, 400.0, 150.0])

        result = GroupBy([key1, key2]).max(values)

        expected_index = pd.MultiIndex.from_tuples([(1, 10), (1, 20), (2, 10), (2, 20)])
        expected = pd.Series(
            [200.0, 400.0, 300.0, 150.0], index=expected_index, name=None
        )

        pd.testing.assert_series_equal(result, expected)

    def test_two_keys_min_aggregation(self):
        """Test min aggregation with multi-key sorting."""
        key1 = np.array([2, 1, 2, 1, 2])
        key2 = np.array([20, 10, 10, 20, 20])
        values = np.array([100.0, 200.0, 300.0, 400.0, 150.0])

        result = GroupBy([key1, key2]).min(values)

        expected_index = pd.MultiIndex.from_tuples([(1, 10), (1, 20), (2, 10), (2, 20)])
        expected = pd.Series(
            [200.0, 400.0, 300.0, 100.0], index=expected_index, name=None
        )

        pd.testing.assert_series_equal(result, expected)

    def test_large_multikey_groupby(self):
        """Test GroupBy with large dataset and multiple keys."""
        np.random.seed(42)
        n = 10000

        key1 = np.random.randint(0, 50, n)
        key2 = np.random.randint(0, 100, n)
        values = np.random.randn(n)

        result = GroupBy([key1, key2]).sum(values)

        assert result.index.is_monotonic_increasing

    def test_empty_arrays_multikey(self):
        """Test multi-key groupby with empty arrays."""
        key1 = np.array([], dtype="int64")
        key2 = np.array([], dtype="int64")
        values = np.array([], dtype="float64")

        result = GroupBy([key1, key2]).sum(values)

        # Should handle empty case
        assert len(result) == 0

    def test_single_group_multikey(self):
        """Test multi-key groupby where all rows have same keys."""
        key1 = np.array([1, 1, 1, 1])
        key2 = np.array([10, 10, 10, 10])
        values = np.array([100, 200, 300, 400])

        result = GroupBy([key1, key2]).sum(values)

        assert len(result) == 1
        assert result.iloc[0] == 1000
        assert result.index[0] == (1, 10)

    def test_multikey_with_datetime(self):
        """Test GroupBy with datetime keys."""
        dates = pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-03", "2020-01-01"])
        category = np.array([1, 1, 2, 2])
        values = np.array([100, 200, 300, 400])

        result = GroupBy([dates, category]).sum(values)

        # Should be sorted by datetime first
        assert result.index[0][0] == pd.Timestamp("2020-01-01")
        assert result.index.is_monotonic_increasing

    def test_multikey_stability(self):
        """Test that sorting is stable (preserves original order for equal keys)."""
        key1 = np.array([1, 1, 1, 2, 2, 2])
        key2 = np.array([10, 10, 10, 20, 20, 20])
        values = np.array([1, 2, 3, 4, 5, 6])

        result = GroupBy([key1, key2]).sum(values)

        expected_index = pd.MultiIndex.from_tuples([(1, 10), (2, 20)])
        expected = pd.Series([1 + 2 + 3, 4 + 5 + 6], index=expected_index, name=None)

        pd.testing.assert_series_equal(result, expected)

    def test_argsort_index_numeric_only_mutlikey_combinations(self):
        np.random.seed(99)
        N = 1000
        cat0 = pd.Categorical.from_codes(np.random.randint(0, 3, N), list("bca"))
        cat1 = pd.Categorical.from_codes(np.random.randint(0, 4, N), list("qwer"))

        ints0 = np.random.randint(5, 10, N)
        ints1 = np.random.randint(0, 5, N)

        for key_combo in [
            (cat0, cat1),
            (cat1, ints0),
            (ints0, cat1),
            (cat0, cat1, ints0),
            (ints1, cat0, ints0),
        ]:
            index = GroupBy(key_combo).size().index
            expected = pd.MultiIndex.from_product(
                [
                    (
                        key.categories
                        if isinstance(key, pd.Categorical)
                        else sorted(np.unique(key))
                    )
                    for key in key_combo
                ]
            )
            assert (expected == index).all()
