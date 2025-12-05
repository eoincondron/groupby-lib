import numpy as np
import pandas as pd
import pytest

from groupby_lib.groupby import core
from groupby_lib.groupby.core import GroupBy, expand_index_to_new_level


class TestBuildGroupIndexers:
    """Test suite for build_group_indexers method with mask support."""

    def test_basic_without_mask(self):
        """Test basic functionality without mask."""
        key = pd.Series([1, 2, 1, 3, 2, 1])
        gb = GroupBy(key)
        groups = gb.groups

        assert len(groups) == 3
        assert np.array_equal(groups[1], np.array([0, 2, 5]))
        assert np.array_equal(groups[2], np.array([1, 4]))
        assert np.array_equal(groups[3], np.array([3]))

    @pytest.mark.parametrize(
        "key_type",
        [
            "int64",
            "int32",
            "float64",
            "string",
            "category",
        ],
    )
    def test_different_key_types(self, key_type):
        """Test with different key dtypes."""
        if key_type == "string":
            key = pd.Series(["a", "b", "a", "c", "b", "a"], dtype="string")
            expected_keys = ["a", "b", "c"]
        elif key_type == "category":
            key = pd.Series(["a", "b", "a", "c", "b", "a"], dtype="category")
            expected_keys = ["a", "b", "c"]
        else:
            key = pd.Series([1, 2, 1, 3, 2, 1], dtype=key_type)
            expected_keys = [1, 2, 3]

        gb = GroupBy(key)
        assert len(gb.groups) == 3
        for k in expected_keys:
            assert k in gb.groups

    def test_categorical_with_unused_categories(self):
        """Test categorical keys with unused categories."""
        key = pd.Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c", "d"], ordered=True
        )
        gb = GroupBy(key)
        groups = gb.groups

        # Only used categories should appear
        assert len(groups) == 2
        assert "a" in groups
        assert "b" in groups
        assert "c" not in groups
        assert "d" not in groups

    def test_chunked_factorization(self, monkeypatch):
        """Test with chunked factorization by lowering threshold."""
        # Monkeypatch the threshold to force chunked factorization
        monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        # Create data larger than threshold
        key = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        gb = GroupBy(key)
        groups = gb.groups

        assert len(groups) == 3
        assert np.array_equal(groups[1], np.array([0, 3, 6, 9]))
        assert np.array_equal(groups[2], np.array([1, 4, 7]))
        assert np.array_equal(groups[3], np.array([2, 5, 8]))

    def test_chunked_factorization_categorical(self, monkeypatch):
        """Test chunked factorization with categorical keys."""
        # Monkeypatch the threshold to force chunked factorization
        monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        # Create categorical data larger than threshold
        key = pd.Categorical(["a", "b", "c", "a", "b", "c", "a", "b"])
        gb = GroupBy(key)
        groups = gb.groups

        assert len(groups) == 3
        assert np.array_equal(groups["a"], np.array([0, 3, 6]))
        assert np.array_equal(groups["b"], np.array([1, 4, 7]))
        assert np.array_equal(groups["c"], np.array([2, 5]))

    def test_all_same_group(self):
        """Test when all rows belong to same group."""
        key = pd.Series([1, 1, 1, 1, 1])
        gb = GroupBy(key)
        groups = gb.groups

        assert len(groups) == 1
        assert np.array_equal(groups[1], np.array([0, 1, 2, 3, 4]))

    @pytest.mark.parametrize("chunked", [True, False])
    def test_multikey(self, monkeypatch, chunked):
        """Test with multiple grouping keys, chunked factorization, and mask."""
        # Monkeypatch the threshold to force chunked factorization
        if chunked:
            monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        key1 = pd.Series([1, 2, 1, 2, 1, 2, 1, 2])
        key2 = pd.Series(["a", "a", "b", "b", "a", "a", "b", "b"])
        gb = GroupBy([key1, key2])
        groups = gb.groups
        expected = key1.groupby([key1, key2]).groups
        assert len(groups) == 4
        assert all(
            np.array_equal(groups[k], expected[k]) for k in expected.keys()
        )


class TestExpandIndexToNewLevel:
    """Test class for expand_index_to_new_level function."""

    def test_expand_simple_index(self):
        """Test expanding a simple single-level index."""
        index = pd.Index([1, 2, 3], name="original")
        new_level = pd.Index(["a", "b"], name="new")

        result = expand_index_to_new_level(index, new_level)

        # Expected: each element of original index repeated twice,
        # combined with [a, b, a, b, a, b]
        expected = pd.MultiIndex.from_tuples(
            [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a"), (3, "b")],
            names=["original", "new"],
        )

        assert result.equals(expected)

    def test_expand_multiindex(self):
        """Test expanding an existing MultiIndex."""
        index = pd.MultiIndex.from_arrays(
            [[1, 2], [10, 20]],
            names=["level_0", "level_1"]
        )
        new_level = pd.Index(["x", "y", "z"], name="level_2")

        result = expand_index_to_new_level(index, new_level)

        # Each group [1, 10] and [2, 20] should be repeated 3 times
        # Combined with [x, y, z] tiled
        expected = pd.MultiIndex.from_tuples(
            [
                (1, 10, "x"), (1, 10, "y"), (1, 10, "z"),
                (2, 20, "x"), (2, 20, "y"), (2, 20, "z"),
            ],
            names=["level_0", "level_1", "level_2"],
        )

        assert result.equals(expected)

    def test_expand_with_unnamed_index(self):
        """Test expanding an index without a name."""
        index = pd.Index([1, 2])
        new_level = pd.Index(["a", "b"])

        result = expand_index_to_new_level(index, new_level)

        expected = pd.MultiIndex.from_tuples(
            [(1, "a"), (1, "b"), (2, "a"), (2, "b")],
            names=[None, None],
        )

        assert result.equals(expected)

    def test_expand_with_single_element_new_level(self):
        """Test expanding with a new level containing only one element."""
        index = pd.Index([1, 2, 3], name="nums")
        new_level = pd.Index(["only"], name="single")

        result = expand_index_to_new_level(index, new_level)

        expected = pd.MultiIndex.from_tuples(
            [(1, "only"), (2, "only"), (3, "only")],
            names=["nums", "single"],
        )

        assert result.equals(expected)

    def test_expand_with_single_element_original_index(self):
        """Test expanding an index with only one element."""
        index = pd.Index([100], name="single")
        new_level = pd.Index(["a", "b", "c"], name="letters")

        result = expand_index_to_new_level(index, new_level)

        expected = pd.MultiIndex.from_tuples(
            [(100, "a"), (100, "b"), (100, "c")],
            names=["single", "letters"],
        )

        assert result.equals(expected)

    def test_expand_with_range_index(self):
        """Test expanding with RangeIndex."""
        index = pd.Index([1, 2])
        new_level = pd.RangeIndex(3, name="range")

        result = expand_index_to_new_level(index, new_level)

        expected = pd.MultiIndex.from_tuples(
            [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            names=[None, "range"],
        )

        assert result.equals(expected)

    def test_expand_preserves_level_names(self):
        """Test that level names are preserved correctly."""
        index = pd.MultiIndex.from_arrays(
            [[1, 2], ["a", "b"]],
            names=["numbers", "letters"]
        )
        new_level = pd.Index([10, 20, 30], name="tens")

        result = expand_index_to_new_level(index, new_level)

        assert result.names == ["numbers", "letters", "tens"]

    def test_expand_with_string_index(self):
        """Test expanding with string-based index."""
        index = pd.Index(["cat", "dog"], name="animals")
        new_level = pd.Index(["small", "medium", "large"], name="size")

        result = expand_index_to_new_level(index, new_level)

        expected = pd.MultiIndex.from_tuples(
            [
                ("cat", "small"), ("cat", "medium"), ("cat", "large"),
                ("dog", "small"), ("dog", "medium"), ("dog", "large"),
            ],
            names=["animals", "size"],
        )

        assert result.equals(expected)


class TestGroupSortIndexer:
    """Test class for GroupBy._group_sort_indexer property."""

    def test_basic_sort_indexer(self):
        """Test basic group sort indexer with simple data."""
        key = pd.Series(["a", "b", "a", "c", "b"])
        gb = GroupBy(key)

        # _group_sort_indexer should give indices that sort data by groups
        indexer = gb._group_sort_indexer

        # Should group all 'a's together, then 'b's, then 'c's
        sorted_keys = key.values[indexer]

        # Check that all 'a's come first, then 'b's, then 'c's
        a_positions = np.where(sorted_keys == "a")[0]
        b_positions = np.where(sorted_keys == "b")[0]
        c_positions = np.where(sorted_keys == "c")[0]

        assert a_positions.max() < b_positions.min()
        assert b_positions.max() < c_positions.min()

        # Check we have the right counts
        assert len(a_positions) == 2
        assert len(b_positions) == 2
        assert len(c_positions) == 1

    def test_sort_indexer_preserves_order_within_groups(self):
        """Test that the indexer preserves original order within each group."""
        key = pd.Series(["a", "b", "a", "c", "b"])
        gb = GroupBy(key)

        indexer = gb._group_sort_indexer

        # The indices for 'a' should be [0, 2] in that order
        # The indices for 'b' should be [1, 4] in that order
        # The indices for 'c' should be [3]

        # Since groups are sorted (a, b, c), we expect indexer to be [0, 2, 1, 4, 3]
        expected = np.array([0, 2, 1, 4, 3])
        np.testing.assert_array_equal(indexer, expected)

    def test_sort_indexer_numeric_keys(self):
        """Test sort indexer with numeric keys."""
        key = np.array([3, 1, 2, 1, 3, 2])
        gb = GroupBy(key)

        indexer = gb._group_sort_indexer
        sorted_keys = key[indexer]

        # Keys should be sorted: [1, 1, 2, 2, 3, 3]
        expected_keys = np.array([1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(sorted_keys, expected_keys)

        # Check the actual indices
        expected_indexer = np.array([1, 3, 2, 5, 0, 4])
        np.testing.assert_array_equal(indexer, expected_indexer)

    def test_sort_indexer_single_group(self):
        """Test sort indexer when all data is in a single group."""
        key = np.array([1, 1, 1, 1])
        gb = GroupBy(key)

        indexer = gb._group_sort_indexer

        # All elements are in the same group, so order should be preserved
        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(indexer, expected)

    def test_sort_indexer_all_unique_groups(self):
        """Test sort indexer when each element is its own group."""
        key = np.array([5, 2, 8, 1, 3])
        gb = GroupBy(key)

        indexer = gb._group_sort_indexer
        sorted_keys = key[indexer]

        # Keys should be sorted in ascending order
        expected_keys = np.array([1, 2, 3, 5, 8])
        np.testing.assert_array_equal(sorted_keys, expected_keys)

        # Check indices
        expected_indexer = np.array([3, 1, 4, 0, 2])
        np.testing.assert_array_equal(indexer, expected_indexer)

    def test_sort_indexer_with_multikey(self):
        """Test sort indexer with multiple grouping keys."""
        key1 = np.array([1, 2, 1, 2, 1])
        key2 = np.array(["a", "b", "b", "a", "a"])
        gb = GroupBy([key1, key2])

        indexer = gb._group_sort_indexer

        # Groups should be: (1,a), (1,b), (2,a), (2,b)
        # Original positions: (1,a): [0, 4], (1,b): [2], (2,a): [3], (2,b): [1]
        sorted_key1 = key1[indexer]
        sorted_key2 = key2[indexer]

        # Check that groups are properly sorted
        for i in range(len(indexer) - 1):
            # Compare (key1[i], key2[i]) with (key1[i+1], key2[i+1])
            if sorted_key1[i] < sorted_key1[i + 1]:
                continue
            elif sorted_key1[i] == sorted_key1[i + 1]:
                assert sorted_key2[i] <= sorted_key2[i + 1]

    def test_sort_indexer_consistent_with_groups(self):
        """Test that _group_sort_indexer is consistent with .groups property."""
        key = pd.Series(["x", "y", "x", "z", "y", "x"])
        gb = GroupBy(key)

        indexer = gb._group_sort_indexer
        groups = gb.groups

        # Concatenate all group indices in sorted order
        sorted_group_keys = sorted(groups.keys())
        expected_indexer = np.concatenate([groups[k] for k in sorted_group_keys])

        np.testing.assert_array_equal(indexer, expected_indexer)

    def test_sort_indexer_large_data(self):
        """Test sort indexer with larger dataset."""
        np.random.seed(42)
        n = 10000
        key = np.random.randint(0, 100, n)
        gb = GroupBy(key)

        indexer = gb._group_sort_indexer
        sorted_keys = key[indexer]

        # Verify that the result is properly sorted by groups
        # All elements with the same key should be consecutive
        assert len(indexer) == n

        # Check that keys are in non-decreasing order
        assert np.all(sorted_keys[:-1] <= sorted_keys[1:])

    def test_sort_indexer_with_unsorted_keys(self):
        """Test sort indexer explicitly with unsorted keys."""
        key = np.array([5, 1, 3, 1, 5, 3])
        gb = GroupBy(key, sort=True)

        indexer = gb._group_sort_indexer
        sorted_keys = key[indexer]

        # Should be sorted: [1, 1, 3, 3, 5, 5]
        expected_keys = np.array([1, 1, 3, 3, 5, 5])
        np.testing.assert_array_equal(sorted_keys, expected_keys)

    def test_sort_indexer_no_sort(self):
        """Test sort indexer when sort=False is specified."""
        key = np.array([3, 1, 2, 1, 3])
        gb = GroupBy(key, sort=False)

        indexer = gb._group_sort_indexer
        sorted_keys = key[indexer]

        # With sort=False, groups appear in order of first occurrence
        # Groups: 3 (first at index 0), 1 (first at index 1), 2 (first at index 2)
        # So we expect: [3, 3, 1, 1, 2]
        expected_keys = np.array([3, 3, 1, 1, 2])
        np.testing.assert_array_equal(sorted_keys, expected_keys)

        # Check actual indices
        expected_indexer = np.array([0, 4, 1, 3, 2])
        np.testing.assert_array_equal(indexer, expected_indexer)

    def test_sort_indexer_cached_property(self):
        """Test that _group_sort_indexer is properly cached."""
        key = np.array([1, 2, 1, 2])
        gb = GroupBy(key)

        # Access the property twice
        indexer1 = gb._group_sort_indexer
        indexer2 = gb._group_sort_indexer

        # Should be the exact same object (cached)
        assert indexer1 is indexer2

    def test_sort_indexer_with_string_categories(self):
        """Test sort indexer with categorical string data."""
        key = pd.Categorical(["dog", "cat", "dog", "bird", "cat"])
        gb = GroupBy(key)

        indexer = gb._group_sort_indexer

        # With categorical, order follows category order not alphabetical
        # Default categorical order is: bird, cat, dog
        sorted_keys = np.array(key)[indexer]

        # Check that all elements with same category are grouped together
        unique_sorted = []
        current = None
        for k in sorted_keys:
            if k != current:
                unique_sorted.append(k)
                current = k

        # There should be exactly 3 unique consecutive groups
        assert len(unique_sorted) == 3


class TestBuildGroupSortedIndex:
    """Test class for GroupBy._build_group_sorted_index method."""

    def test_basic_build_group_sorted_index(self):
        """Test basic functionality of _build_group_sorted_index."""
        key = pd.Series(["a", "b", "a", "c", "b"])
        gb = GroupBy(key)

        result = gb._build_group_sorted_index()

        # Should create a MultiIndex with group keys as outer level
        # and original positions as inner level
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 2

        # Check that the outer level contains the group keys in sorted order
        outer_level_values = result.get_level_values(0)
        # Groups should be sorted: a, a, b, b, c
        expected_outer = pd.Index(["a", "a", "b", "b", "c"])
        assert outer_level_values.equals(expected_outer)

        # Check that inner level preserves original order within groups
        # Group 'a': indices [0, 2]
        # Group 'b': indices [1, 4]
        # Group 'c': index [3]
        inner_level_values = result.get_level_values(1).to_numpy()
        expected_inner = np.array([0, 2, 1, 4, 3])
        np.testing.assert_array_equal(inner_level_values, expected_inner)

    def test_build_group_sorted_index_with_custom_inner_index(self):
        """Test _build_group_sorted_index with a custom inner index."""
        key = pd.Series(["x", "y", "x"])
        inner_index = pd.Index(["row_a", "row_b", "row_c"], name="rows")

        gb = GroupBy(key)
        result = gb._build_group_sorted_index(inner_index=inner_index)

        # Should create a MultiIndex with group keys and custom inner index
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 2
        assert result.names[1] == "rows"

        # Check the inner level values
        inner_values = result.get_level_values(1)
        # Group 'x': [row_a, row_c], Group 'y': [row_b]
        expected_inner = pd.Index(["row_a", "row_c", "row_b"])
        assert inner_values.equals(expected_inner)

    def test_build_group_sorted_index_with_multiindex_inner(self):
        """Test _build_group_sorted_index with a MultiIndex as inner_index."""
        key = pd.Series(["a", "b", "a", "b"])
        inner_index = pd.MultiIndex.from_arrays(
            [[1, 2, 3, 4], ["p", "q", "r", "s"]],
            names=["num", "letter"]
        )

        gb = GroupBy(key)
        result = gb._build_group_sorted_index(inner_index=inner_index)

        # Should create a MultiIndex with 3 levels:
        # group key, inner level 0, inner level 1
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 3
        assert result.names == [None, "num", "letter"]

        # Check the structure
        # Group 'a': indices [0, 2] -> (1, 'p'), (3, 'r')
        # Group 'b': indices [1, 3] -> (2, 'q'), (4, 's')
        expected_level_1 = np.array([1, 3, 2, 4])
        expected_level_2 = ["p", "r", "q", "s"]
        np.testing.assert_array_equal(
            result.get_level_values(1).to_numpy(), expected_level_1
        )
        assert list(result.get_level_values(2)) == expected_level_2

    def test_build_group_sorted_index_with_multikey(self):
        """Test _build_group_sorted_index with multiple grouping keys."""
        key1 = pd.Series([1, 2, 1, 2, 1])
        key2 = pd.Series(["a", "b", "b", "a", "a"])

        gb = GroupBy([key1, key2])
        result = gb._build_group_sorted_index()

        # Should create a MultiIndex with group keys as outer levels
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 3  # 2 for group keys + 1 for original positions

        # Check group key levels
        assert result.names[:2] == [None, None]

        # Groups should be: (1,a), (1,b), (2,a), (2,b)
        # Original positions: (1,a): [0, 4], (1,b): [2], (2,a): [3], (2,b): [1]
        expected_first_level = [1, 1, 1, 2, 2]
        expected_second_level = ["a", "a", "b", "a", "b"]
        assert list(result.get_level_values(0)) == expected_first_level
        assert list(result.get_level_values(1)) == expected_second_level

    def test_build_group_sorted_index_with_multikey_and_multiindex_inner(self):
        """Test with both multi-key grouping and MultiIndex inner_index."""
        key1 = pd.Series([1, 1, 2, 2])
        key2 = pd.Series(["a", "b", "a", "b"])
        inner_index = pd.MultiIndex.from_arrays(
            [[10, 20, 30, 40], ["p", "q", "r", "s"]],
            names=["val", "char"]
        )

        gb = GroupBy([key1, key2])
        result = gb._build_group_sorted_index(inner_index=inner_index)

        # Should have 4 levels:
        # 2 from group keys + 2 from inner MultiIndex
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 4
        assert result.names == [None, None, "val", "char"]

        # Groups: (1,a): [0], (1,b): [1], (2,a): [2], (2,b): [3]
        # So order should be: 0, 1, 2, 3
        expected_val_level = [10, 20, 30, 40]
        expected_char_level = ["p", "q", "r", "s"]
        assert list(result.get_level_values(2)) == expected_val_level
        assert list(result.get_level_values(3)) == expected_char_level

    def test_build_group_sorted_index_with_unsorted_groups(self):
        """Test _build_group_sorted_index with unsorted group keys."""
        key = pd.Series([3, 1, 2, 1, 3])
        gb = GroupBy(key, sort=True)

        result = gb._build_group_sorted_index()

        # Groups should be sorted: 1, 1, 2, 3, 3
        outer_values = result.get_level_values(0).to_numpy()
        expected_outer = np.array([1, 1, 2, 3, 3])
        np.testing.assert_array_equal(outer_values, expected_outer)

        # Check inner indices
        # Group 1: [1, 3], Group 2: [2], Group 3: [0, 4]
        inner_values = result.get_level_values(1).to_numpy()
        expected_inner = np.array([1, 3, 2, 0, 4])
        np.testing.assert_array_equal(inner_values, expected_inner)

    def test_build_group_sorted_index_no_sort(self):
        """Test _build_group_sorted_index when sort=False."""
        key = pd.Series([3, 1, 2, 1, 3])
        gb = GroupBy(key, sort=False)

        result = gb._build_group_sorted_index()

        # With sort=False, groups appear in order of first occurrence
        # Groups: 3, 1, 2
        outer_values = result.get_level_values(0).to_numpy()
        # Expected order: 3, 3, 1, 1, 2
        expected_outer = np.array([3, 3, 1, 1, 2])
        np.testing.assert_array_equal(outer_values, expected_outer)

    def test_build_group_sorted_index_single_group(self):
        """Test _build_group_sorted_index with a single group."""
        key = pd.Series([1, 1, 1, 1])
        gb = GroupBy(key)

        result = gb._build_group_sorted_index()

        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 2

        # All should be in same group
        outer_values = result.get_level_values(0).to_numpy()
        expected = np.array([1, 1, 1, 1])
        np.testing.assert_array_equal(outer_values, expected)

        # Inner indices should be [0, 1, 2, 3]
        inner_values = result.get_level_values(1).to_numpy()
        expected_inner = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(inner_values, expected_inner)

    def test_build_group_sorted_index_with_named_inner_index(self):
        """Test that inner index names are preserved."""
        key = pd.Series(["a", "b", "a"])
        inner_index = pd.Index([100, 200, 300], name="ids")

        gb = GroupBy(key)
        result = gb._build_group_sorted_index(inner_index=inner_index)

        # Check that names are preserved
        assert result.names[1] == "ids"

        # Check values
        inner_values = result.get_level_values(1).to_numpy()
        # Group 'a': [100, 300], Group 'b': [200]
        expected = np.array([100, 300, 200])
        np.testing.assert_array_equal(inner_values, expected)
