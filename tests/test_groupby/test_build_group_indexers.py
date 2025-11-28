import numpy as np
import pandas as pd
import pytest

from groupby_lib.groupby import core
from groupby_lib.groupby.core import GroupBy


class TestBuildGroupIndexers:
    """Test suite for build_group_indexers method with mask support."""

    def test_basic_without_mask(self):
        """Test basic functionality without mask."""
        key = pd.Series([1, 2, 1, 3, 2, 1])
        gb = GroupBy(key)
        groups = gb._build_group_indexers()

        assert len(groups) == 3
        assert np.array_equal(groups[1], np.array([0, 2, 5]))
        assert np.array_equal(groups[2], np.array([1, 4]))
        assert np.array_equal(groups[3], np.array([3]))

    def test_basic_with_mask(self):
        """Test basic functionality with mask."""
        key = pd.Series([1, 2, 1, 3, 2, 1])
        mask = np.array([True, True, False, True, True, False])
        gb = GroupBy(key)
        groups = gb._build_group_indexers(mask=mask)

        # Only indices 0, 1, 3, 4 should be included
        assert len(groups) == 3
        assert np.array_equal(groups[1], np.array([0]))
        assert np.array_equal(groups[2], np.array([1, 4]))
        assert np.array_equal(groups[3], np.array([3]))

    def test_mask_filters_entire_group(self):
        """Test that groups completely filtered by mask are excluded."""
        key = pd.Series([1, 1, 2, 2, 3, 3])
        mask = np.array([True, True, True, True, False, False])
        gb = GroupBy(key)
        groups = gb._build_group_indexers(mask=mask)

        # Group 3 should not appear
        assert len(groups) == 2
        assert 1 in groups
        assert 2 in groups
        assert 3 not in groups

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

        mask = np.array([True, True, False, True, True, False])
        gb = GroupBy(key)
        groups = gb._build_group_indexers(mask=mask)

        assert len(groups) == 3
        for k in expected_keys:
            assert k in groups

    def test_categorical_with_unused_categories(self):
        """Test categorical keys with unused categories."""
        key = pd.Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c", "d"], ordered=True
        )
        gb = GroupBy(key)
        groups = gb._build_group_indexers()

        # Only used categories should appear
        assert len(groups) == 2
        assert "a" in groups
        assert "b" in groups
        assert "c" not in groups
        assert "d" not in groups

    def test_categorical_with_mask(self):
        """Test categorical keys with mask."""
        key = pd.Categorical(
            ["a", "b", "a", "b", "c"], categories=["a", "b", "c"], ordered=True
        )
        mask = np.array([True, True, False, True, True])
        gb = GroupBy(key)
        groups = gb._build_group_indexers(mask=mask)

        assert len(groups) == 3
        assert np.array_equal(groups["a"], np.array([0]))
        assert np.array_equal(groups["b"], np.array([1, 3]))
        assert np.array_equal(groups["c"], np.array([4]))

    def test_chunked_factorization(self, monkeypatch):
        """Test with chunked factorization by lowering threshold."""
        # Monkeypatch the threshold to force chunked factorization
        monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        # Create data larger than threshold
        key = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        gb = GroupBy(key)
        groups = gb._build_group_indexers()

        assert len(groups) == 3
        assert np.array_equal(groups[1], np.array([0, 3, 6, 9]))
        assert np.array_equal(groups[2], np.array([1, 4, 7]))
        assert np.array_equal(groups[3], np.array([2, 5, 8]))

    def test_chunked_factorization_with_mask(self, monkeypatch):
        """Test chunked factorization with mask."""
        # Monkeypatch the threshold to force chunked factorization
        monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        # Create data larger than threshold
        key = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        mask = np.array([True, True, False, True, False, True, True, True, False, True])
        gb = GroupBy(key)
        groups = gb._build_group_indexers(mask=mask)

        # Only indices where mask is True: 0, 1, 3, 5, 6, 7, 9
        assert len(groups) == 3
        assert np.array_equal(groups[1], np.array([0, 3, 6, 9]))
        assert np.array_equal(groups[2], np.array([1, 7]))
        assert np.array_equal(groups[3], np.array([5]))

    def test_chunked_factorization_categorical(self, monkeypatch):
        """Test chunked factorization with categorical keys."""
        # Monkeypatch the threshold to force chunked factorization
        monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        # Create categorical data larger than threshold
        key = pd.Categorical(["a", "b", "c", "a", "b", "c", "a", "b"])
        gb = GroupBy(key)
        groups = gb._build_group_indexers()

        assert len(groups) == 3
        assert np.array_equal(groups["a"], np.array([0, 3, 6]))
        assert np.array_equal(groups["b"], np.array([1, 4, 7]))
        assert np.array_equal(groups["c"], np.array([2, 5]))

    def test_chunked_factorization_categorical_with_mask(self, monkeypatch):
        """Test chunked factorization with categorical keys and mask."""
        # Monkeypatch the threshold to force chunked factorization
        monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        # Create categorical data larger than threshold
        key = np.array([1, 2, 3, 1, 2, 3, 1, 2])
        mask = np.array([True, False, True, True, True, False, True, True])
        gb = GroupBy(key)
        assert hasattr(gb.group_ikey, "chunks")
        groups = gb._build_group_indexers(mask=mask)

        # Only indices where mask is True: 0, 2, 3, 4, 6, 7
        assert len(groups) == 3
        assert np.array_equal(groups[1], np.array([0, 3, 6]))
        assert np.array_equal(groups[2], np.array([4, 7]))
        assert np.array_equal(groups[3], np.array([2]))

    def test_empty_result_with_mask(self):
        """Test when mask filters out all rows."""
        key = pd.Series([1, 2, 3])
        mask = np.array([False, False, False])
        gb = GroupBy(key)
        groups = gb._build_group_indexers(mask=mask)

        assert len(groups) == 0

    def test_all_same_group(self):
        """Test when all rows belong to same group."""
        key = pd.Series([1, 1, 1, 1, 1])
        gb = GroupBy(key)
        groups = gb._build_group_indexers()

        assert len(groups) == 1
        assert np.array_equal(groups[1], np.array([0, 1, 2, 3, 4]))

    def test_all_same_group_with_mask(self):
        """Test when all rows belong to same group with mask."""
        key = pd.Series([1, 1, 1, 1, 1])
        mask = np.array([True, False, True, False, True])
        gb = GroupBy(key)
        groups = gb._build_group_indexers(mask=mask)

        assert len(groups) == 1
        assert np.array_equal(groups[1], np.array([0, 2, 4]))

    def test_multikey_without_mask(self):
        """Test with multiple grouping keys without mask."""
        key1 = pd.Series([1, 1, 2, 2, 1])
        key2 = pd.Series(["a", "b", "a", "b", "a"])
        gb = GroupBy([key1, key2])
        groups = gb._build_group_indexers()

        assert len(groups) == 4
        assert np.array_equal(groups[(1, "a")], np.array([0, 4]))
        assert np.array_equal(groups[(1, "b")], np.array([1]))
        assert np.array_equal(groups[(2, "a")], np.array([2]))
        assert np.array_equal(groups[(2, "b")], np.array([3]))

    def test_multikey_with_mask(self):
        """Test with multiple grouping keys with mask."""
        key1 = pd.Series([1, 1, 2, 2, 1])
        key2 = pd.Series(["a", "b", "a", "b", "a"])
        # mask includes indices: 0, 1, 3, 4
        # key1[0,1,3,4] = [1, 1, 2, 1], key2[0,1,3,4] = ['a', 'b', 'b', 'a']
        # So groups should be: (1,'a'):[0,4], (1,'b'):[1], (2,'b'):[3]
        mask = np.array([True, True, False, True, True])
        gb = GroupBy([key1, key2])
        groups = gb._build_group_indexers(mask=mask)

        # Only indices 0, 1, 3, 4 are included by mask
        assert len(groups) == 3
        expected = key1[mask].groupby([key1, key2], observed=True).groups
        for k, arr in expected.items():
            np.testing.assert_array_equal(groups[k], arr)

    def test_multikey_chunked_with_mask(self, monkeypatch):
        """Test with multiple grouping keys, chunked factorization, and mask."""
        # Monkeypatch the threshold to force chunked factorization
        monkeypatch.setattr(core, "THRESHOLD_FOR_CHUNKED_FACTORIZE", 5)

        key1 = pd.Series([1, 2, 1, 2, 1, 2, 1, 2])
        key2 = pd.Series(["a", "a", "b", "b", "a", "a", "b", "b"])
        mask = np.array([True, True, False, True, True, False, True, True])
        gb = GroupBy([key1, key2])
        groups = gb._build_group_indexers(mask=mask)

        # Only indices where mask is True: 0, 1, 3, 4, 6, 7
        assert len(groups) == 4
        assert np.array_equal(groups[(1, "a")], np.array([0, 4]))
        assert np.array_equal(groups[(2, "a")], np.array([1]))
        assert np.array_equal(groups[(1, "b")], np.array([6]))
        assert np.array_equal(groups[(2, "b")], np.array([3, 7]))
