"""
Unit tests for groupby-lib GroupBy API classes.

This module tests SeriesGroupBy and DataFrameGroupBy classes to ensure they
provide pandas-compatible behavior while leveraging the optimized core.
"""

import numpy as np
import pandas as pd
import pytest

from groupby_lib.groupby import DataFrameGroupBy, SeriesGroupBy, install_groupby_fast

# Install groupby_fast for all tests
install_groupby_fast()


class TestSeriesGroupBy:
    """Test SeriesGroupBy functionality."""

    def setup_method(self):
        """Setup test data."""
        self.data = pd.Series([1, 2, 3, 4, 5, 6], name="values")
        self.groups = pd.Series(["A", "B", "A", "B", "A", "B"], name="groups")

        # MultiIndex data for level testing
        self.multi_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2), ("A", 1), ("B", 2)],
            names=["letter", "number"],
        )
        self.multi_data = pd.Series(
            [10, 20, 30, 40, 50, 60], index=self.multi_index, name="values"
        )

    def test_basic_grouping(self):
        """Test basic grouping with by parameter."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        assert gb.ngroups == pandas_gb.ngroups
        assert list(gb.groups.keys()) == list(pandas_gb.groups.keys())

    def test_sum_aggregation(self):
        """Test sum aggregation matches pandas."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        result = gb.sum()
        expected = pandas_gb.sum()
        pd.testing.assert_series_equal(result, expected)

    def test_mean_aggregation(self):
        """Test mean aggregation matches pandas."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        result = gb.mean()
        expected = pandas_gb.mean()

        pd.testing.assert_series_equal(result, expected)

    def test_count_aggregation(self):
        """Test count aggregation matches pandas."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        result = gb.count()
        expected = pandas_gb.count()

        pd.testing.assert_series_equal(result, expected)

    def test_std_var_aggregation(self):
        """Test std and var aggregations with ddof parameter."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        # Test std
        result_std = gb.std(ddof=1)
        expected_std = pandas_gb.std(ddof=1)
        pd.testing.assert_series_equal(result_std, expected_std)

        # Test var
        result_var = gb.var(ddof=1)
        expected_var = pandas_gb.var(ddof=1)
        pd.testing.assert_series_equal(result_var, expected_var)

    def test_min_max_aggregation(self):
        """Test min and max aggregations."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        # Test min
        result_min = gb.min()
        expected_min = pandas_gb.min()
        pd.testing.assert_series_equal(result_min, expected_min)

        # Test max
        result_max = gb.max()
        expected_max = pandas_gb.max()
        pd.testing.assert_series_equal(result_max, expected_max)

    def test_cumulative_operations(self):
        """Test cumulative operations."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        # Test cumsum
        result_cumsum = gb.cumsum()
        expected_cumsum = pandas_gb.cumsum()
        pd.testing.assert_series_equal(result_cumsum, expected_cumsum)

        # Test cummax
        result_cummax = gb.cummax()
        expected_cummax = pandas_gb.cummax()
        pd.testing.assert_series_equal(result_cummax, expected_cummax)

        # Test cummin
        result_cummin = gb.cummin()
        expected_cummin = pandas_gb.cummin()
        pd.testing.assert_series_equal(result_cummin, expected_cummin)

        # Test cumcount
        result_cumcount = gb.cumcount()
        expected_cumcount = pandas_gb.cumcount()
        pd.testing.assert_series_equal(result_cumcount, expected_cumcount)

    def test_level_grouping_by_number(self):
        """Test grouping by level number."""
        gb = self.multi_data.groupby_fast(level=0)
        pandas_gb = self.multi_data.groupby(level=0)

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_level_grouping_by_name(self):
        """Test grouping by level name."""
        gb = self.multi_data.groupby_fast(level="letter")
        pandas_gb = self.multi_data.groupby(level="letter")

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_multiple_level_grouping(self):
        """Test grouping by multiple levels."""
        gb = self.multi_data.groupby_fast(level=[0, 1])
        pandas_gb = self.multi_data.groupby(level=[0, 1])

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_combined_by_and_level_grouping(self):
        """Test combining by and level parameters."""
        extra_grouper = pd.Series(
            ["X", "Y", "X", "Y", "X", "Y"], index=self.multi_data.index, name="extra"
        )

        gb = self.multi_data.groupby_fast(by=extra_grouper, level=0)
        pandas_gb = self.multi_data.groupby(
            [extra_grouper, self.multi_data.index.get_level_values(0)]
        )

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_mask_parameter(self):
        """Test mask parameter functionality."""
        mask = np.array([True, False, True, True, False, True])

        gb = self.data.groupby_fast(by=self.groups)
        result = gb.sum(mask=mask)

        # Compare to manual calculation
        masked_data = self.data[mask]
        masked_groups = self.groups[mask]
        expected = masked_data.groupby(masked_groups).sum()

        pd.testing.assert_series_equal(result, expected)

    def test_first_last_operations(self):
        """Test first and last operations."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        # Test first
        result_first = gb.first()
        expected_first = pandas_gb.first()
        pd.testing.assert_series_equal(result_first, expected_first)

        # Test last
        result_last = gb.last()
        expected_last = pandas_gb.last()
        pd.testing.assert_series_equal(result_last, expected_last)

    def test_size_operation(self):
        """Test size operation."""
        gb = self.data.groupby_fast(by=self.groups)
        pandas_gb = self.data.groupby(self.groups)

        result = gb.size()
        expected = pandas_gb.size()

        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_margins_functionality(self):
        """Test margins parameter in aggregation methods."""
        gb = self.data.groupby_fast(by=self.groups)

        # Test sum with margins
        result_with_margins = gb.sum(margins=True)
        assert "All" in result_with_margins.index

        # The 'All' value should be the total sum
        regular_sum = gb.sum()
        total_sum = regular_sum.sum()
        assert result_with_margins.loc["All"] == total_sum

        # Test other methods with margins
        methods = ["mean", "min", "max", "std", "var"]
        for method in methods:
            result = getattr(gb, method)(margins=True)
            assert "All" in result.index

    def test_rolling_functionality(self):
        """Test rolling window operations."""
        gb = self.data.groupby_fast(by=self.groups)
        rolling_gb = gb.rolling(window=2)

        # Test rolling sum
        rolling_result = rolling_gb.sum()
        assert isinstance(rolling_result, pd.Series)
        assert len(rolling_result) == len(self.data)

        # Test rolling mean
        rolling_mean = rolling_gb.mean()
        assert isinstance(rolling_mean, pd.Series)

        # Test other rolling methods
        rolling_methods = ["min", "max"]
        for method in rolling_methods:
            result = getattr(rolling_gb, method)()
            assert isinstance(result, pd.Series)
            assert len(result) == len(self.data)

    def test_null_value_handling(self):
        """Test handling of null values."""
        data_with_nulls = pd.Series([10, np.nan, 30, 40, np.nan, 60], name="values")
        groups_with_nulls = pd.Series(["A", "B", "A", "B", "A", "B"], name="groups")

        gb = data_with_nulls.groupby_fast(by=groups_with_nulls)
        pandas_gb = data_with_nulls.groupby(groups_with_nulls)

        # Test count (excludes nulls)
        result_count = gb.count()
        expected_count = pandas_gb.count()
        pd.testing.assert_series_equal(result_count, expected_count)

        # Test sum (handles nulls)
        result_sum = gb.sum()
        expected_sum = pandas_gb.sum()
        pd.testing.assert_series_equal(result_sum, expected_sum)

    def test_constructor_validation(self):
        """Test constructor parameter validation."""
        # Should raise error if no by or level provided
        with pytest.raises(
            ValueError, match="Must provide either 'by' or 'level'"
        ):
            SeriesGroupBy._from_by_keys(self.data)

        # Should raise error if obj is not Series
        with pytest.raises(TypeError, match="obj must be a pandas Series"):
            SeriesGroupBy._from_by_keys(pd.DataFrame({"A": [1, 2, 3]}), by=self.groups[:3])

    def test_repr(self):
        """Test string representation."""
        gb = self.data.groupby_fast(by=self.groups)
        repr_str = repr(gb)
        assert "SeriesGroupBy" in repr_str
        assert "ngroups=2" in repr_str


class TestDataFrameGroupBy:
    """Test DataFrameGroupBy functionality."""

    def setup_method(self):
        """Setup test data."""
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6],
                "B": [10, 20, 30, 40, 50, 60],
                "C": [100, 200, 300, 400, 500, 600],
            }
        )
        self.groups = pd.Series(["X", "Y", "X", "Y", "X", "Y"], name="groups")

    def test_basic_dataframe_grouping(self):
        """Test basic DataFrame grouping."""
        gb = self.df.groupby_fast(by=self.groups)
        pandas_gb = self.df.groupby(self.groups)

        assert gb.ngroups == pandas_gb.ngroups
        assert list(gb.groups.keys()) == list(pandas_gb.groups.keys())

    def test_dataframe_sum_aggregation(self):
        """Test DataFrame sum aggregation."""
        gb = self.df.groupby_fast(by=self.groups)
        pandas_gb = self.df.groupby(self.groups)

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_dataframe_aggregations(self):
        """Test various DataFrame aggregations."""
        gb = self.df.groupby_fast(by=self.groups)
        pandas_gb = self.df.groupby(self.groups)

        methods = ["mean", "min", "max", "count", "std", "var"]

        for method in methods:
            result = getattr(gb, method)()
            expected = getattr(pandas_gb, method)()
            pd.testing.assert_frame_equal(result, expected)

    def test_getitem_single_column(self):
        """Test __getitem__ with single column selection."""
        gb = self.df.groupby_fast(by=self.groups)

        # Single column should return SeriesGroupBy
        single_col_gb = gb["A"]
        assert isinstance(single_col_gb, SeriesGroupBy)

        # Should produce same result as pandas
        pandas_gb = self.df.groupby(self.groups)
        pandas_single = pandas_gb["A"]

        result = single_col_gb.sum()
        expected = pandas_single.sum()
        pd.testing.assert_series_equal(result, expected)

    def test_getitem_multiple_columns(self):
        """Test __getitem__ with multiple column selection."""
        gb = self.df.groupby_fast(by=self.groups)

        # Multiple columns should return DataFrameGroupBy
        multi_col_gb = gb[["A", "B"]]
        assert isinstance(multi_col_gb, DataFrameGroupBy)

        # Should produce same result as pandas
        pandas_gb = self.df.groupby(self.groups)
        pandas_multi = pandas_gb[["A", "B"]]

        result = multi_col_gb.sum()
        expected = pandas_multi.sum()
        pd.testing.assert_frame_equal(result, expected)

    def test_group_column_excluded(self):
        """Test __getitem__ with multiple column selection."""
        gb = self.df.groupby_fast("A")

        # Should produce same result as pandas
        pandas_gb = self.df.groupby("A")

        result = gb.sum()
        expected = pandas_gb.sum()
        pd.testing.assert_frame_equal(result, expected)

    def test_dataframe_size_operation(self):
        """Test DataFrame size operation returns Series."""
        gb = self.df.groupby_fast(by=self.groups)
        pandas_gb = self.df.groupby(self.groups)

        result = gb.size()
        expected = pandas_gb.size()

        # Size should return a Series even for DataFrameGroupBy
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_dataframe_mask_parameter(self):
        """Test mask parameter with DataFrame."""
        mask = np.array([True, False, True, True, False, True])

        gb = self.df.groupby_fast(by=self.groups)
        result = gb.sum(mask=mask)

        # Compare to manual calculation
        masked_df = self.df[mask]
        masked_groups = self.groups[mask]
        expected = masked_df.groupby(masked_groups).sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_dataframe_constructor_validation(self):
        """Test DataFrame constructor validation."""
        # Should raise error if no by or level provided
        with pytest.raises(
            ValueError, match="Must provide either 'by' or 'level'"
        ):
            DataFrameGroupBy._from_by_keys(self.df)

        # Should raise error if obj is not DataFrame
        with pytest.raises(TypeError, match="obj must be a pandas DataFrame"):
            DataFrameGroupBy._from_by_keys(pd.Series([1, 2, 3]), by=self.groups[:3])

    def test_dataframe_repr(self):
        """Test DataFrame string representation."""
        gb = self.df.groupby_fast(by=self.groups)
        repr_str = repr(gb)
        assert "DataFrameGroupBy" in repr_str
        assert "ngroups=2" in repr_str

    def test_dataframe_margins_functionality(self):
        """Test margins parameter with DataFrame."""
        gb = self.df.groupby_fast(by=self.groups)

        # Test sum with margins
        result_with_margins = gb.sum(margins=True)
        assert "All" in result_with_margins.index

        # Test other methods with margins
        methods = ["mean", "min", "max", "std", "var"]
        for method in methods:
            result = getattr(gb, method)(margins=True)
            assert "All" in result.index
            assert isinstance(result, pd.DataFrame)

    def test_dataframe_rolling_functionality(self):
        """Test DataFrame rolling window operations."""
        gb = self.df.groupby_fast(by=self.groups)
        rolling_gb = gb.rolling(window=2)

        # Test rolling sum
        rolling_result = rolling_gb.sum()
        assert isinstance(rolling_result, pd.DataFrame)
        assert rolling_result.shape[0] == self.df.shape[0]
        assert rolling_result.shape[1] == self.df.shape[1]

        # Test rolling mean
        rolling_mean = rolling_gb.mean()
        assert isinstance(rolling_mean, pd.DataFrame)

        # Test other rolling methods
        rolling_methods = ["min", "max"]
        for method in rolling_methods:
            result = getattr(rolling_gb, method)()
            assert isinstance(result, pd.DataFrame)
            assert result.shape == self.df.shape


class TestGroupByIntegration:
    """Test integration between SeriesGroupBy and DataFrameGroupBy."""

    def setup_method(self):
        """Setup test data."""
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6],
                "B": [10, 20, 30, 40, 50, 60],
            }
        )
        self.groups = pd.Series(["X", "Y", "X", "Y", "X", "Y"])

    def test_dataframe_to_series_consistency(self):
        """Test that DataFrameGroupBy column selection matches direct SeriesGroupBy."""
        df_gb = self.df.groupby_fast(by=self.groups)
        series_gb = self.df["A"].groupby_fast(by=self.groups)

        # Results should be identical
        df_result = df_gb["A"].sum()
        series_result = series_gb.sum()

        pd.testing.assert_series_equal(df_result, series_result)

    def test_grouper_sharing(self):
        """Test that grouper objects can be shared between instances."""
        df_gb = self.df.groupby_fast(by=self.groups)

        # Column selection should share the same grouper
        col_gb = df_gb["A"]

        # Both should have same number of groups
        assert df_gb.ngroups == col_gb.ngroups
        assert list(df_gb.groups.keys()) == list(col_gb.groups.keys())

    def test_iter(self):
        df_gb = self.df.groupby_fast(by=self.groups)
        group_dict = dict(df_gb)
        assert list(group_dict) == ["X", "Y"]
        for k, sub in df_gb:
            pd.testing.assert_frame_equal(sub, self.df.loc[self.groups == k])


class TestEMA:
    """Test EMA functionality."""

    def test_series_ema_with_alpha(self):
        """Test basic Series EMA with alpha parameter."""
        data = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        groups = pd.Series([1, 1, 1, 2, 2, 2])

        gb = data.groupby_fast(by=groups)
        result = gb.ema(alpha=0.5)

        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert not result.isna().any()

        # Check that groups are independent
        # Group 1 values should differ from Group 2 values
        group1_result = result.iloc[:3]
        group2_result = result.iloc[3:]
        assert group1_result.iloc[0] == 1.0  # First value unchanged
        assert group2_result.iloc[0] == 10.0  # First value unchanged

    def test_series_ema_with_halflife(self):
        """Test Series EMA with halflife parameter."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0])
        groups = pd.Series(["A", "A", "B", "B"])

        gb = data.groupby_fast(by=groups)
        result = gb.ema(halflife=2.0)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert not result.isna().any()

    def test_series_ema_with_time_weights(self):
        """Test Series EMA with time-weighted calculation."""
        data = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        groups = pd.Series([1, 1, 1, 2, 2, 2])
        times = pd.date_range("2024-01-01", periods=6, freq="1h")

        gb = data.groupby_fast(by=groups)
        result = gb.ema(halflife="2h", times=times)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert not result.isna().any()

    def test_series_ema_with_mask(self):
        """Test Series EMA with mask parameter."""
        data = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        groups = pd.Series([1, 1, 1, 2, 2, 2])
        mask = np.array([True, True, False, True, True, True])

        gb = data.groupby_fast(by=groups)
        result = gb.ema(alpha=0.5, mask=mask)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_dataframe_ema(self):
        """Test DataFrame EMA functionality."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, 40.0]})
        groups = pd.Series([1, 1, 2, 2])

        gb = df.groupby_fast(by=groups)
        result = gb.ema(alpha=0.5)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert not result.isna().any().any()

    def test_dataframe_single_column_ema(self):
        """Test EMA on single column selection from DataFrame."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, 40.0]})
        groups = pd.Series([1, 1, 2, 2])

        gb = df.groupby_fast(by=groups)
        result = gb["A"].ema(alpha=0.5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(df)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data(self):
        """Test handling of empty data."""
        empty_series = pd.Series([], dtype="float64", name="empty")
        empty_groups = pd.Series([], dtype="object", name="groups")

        gb = empty_series.groupby_fast(by=empty_groups)
        assert gb.ngroups == 0

        # Aggregations on empty data should return empty results
        result = gb.sum()
        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_single_group(self):
        """Test handling of single group."""
        data = pd.Series([1, 2, 3, 4])
        groups = pd.Series(["A", "A", "A", "A"])

        gb = data.groupby_fast(by=groups)
        pandas_gb = data.groupby(groups)

        assert gb.ngroups == 1

        result = gb.sum()
        expected = pandas_gb.sum()
        pd.testing.assert_series_equal(result, expected)

    def test_large_number_of_groups(self):
        """Test handling of large number of groups."""
        n = 1000
        data = pd.Series(range(n))
        groups = pd.Series(
            [f"group_{i}" for i in range(n)]
        )  # Each value in its own group

        gb = data.groupby_fast(by=groups)
        pandas_gb = data.groupby(groups)

        assert gb.ngroups == n

        result = gb.sum()
        expected = pandas_gb.sum()
        pd.testing.assert_series_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
