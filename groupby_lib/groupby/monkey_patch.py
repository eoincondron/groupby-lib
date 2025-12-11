"""
Monkey-patch functionality to add groupby_fast methods to pandas Series and DataFrame.

This module provides runtime patching to add our optimized GroupBy implementations
as methods on pandas objects, allowing seamless usage like df.groupby_fast('column').sum().
"""

from typing import Any, List, Optional, Union

import pandas as pd

try:
    import polars as pl  # noqa

    polars_installed = True
except ImportError:
    polars_installed = False

from .api import DataFrameGroupBy, SeriesGroupBy


def groupby_fast_series(
    self: pd.Series, by=None, level=None, **kwargs
) -> SeriesGroupBy:
    """
    Group Series using optimized groupby-lib GroupBy implementation.

    This method provides the same interface as pandas Series.groupby() but uses
    the groupby-lib optimized GroupBy engine for better performance.

    Parameters
    ----------
    by : array-like, optional
        Grouping key(s). Can be any type acceptable to groupby-lib GroupBy constructor.
        If None, must specify level.
    level : int, str, or sequence, optional
        If the Series has a MultiIndex, group by specific level(s) of the index.
        Can be level number(s) or name(s). If None, must specify by.
    **kwargs
        Additional keyword arguments (for pandas compatibility, mostly ignored)

    Returns
    -------
    SeriesGroupBy
        groupby-lib SeriesGroupBy object with optimized performance

    Examples
    --------
    >>> import pandas as pd
    >>> from groupby_lib.groupby import install_groupby_fast
    >>> install_groupby_fast()  # Add groupby_fast methods
    >>>
    >>> s = pd.Series([1, 2, 3, 4, 5, 6])
    >>> groups = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'])
    >>> result = s.groupby_fast(groups).sum()
    >>> print(result)
    A    9
    B   12
    dtype: int64
    """
    return SeriesGroupBy._from_by_keys(self, by=by, level=level)


def groupby_fast_dataframe(
    self: pd.DataFrame, by=None, level=None, **kwargs
) -> DataFrameGroupBy:
    """
    Group DataFrame using optimized groupby-lib GroupBy implementation.

    This method provides the same interface as pandas DataFrame.groupby() but uses
    the groupby-lib optimized GroupBy engine for better performance.

    Parameters
    ----------
    by : array-like, optional
        Grouping key(s). Can be any type acceptable to groupby-lib GroupBy constructor.
        If None, must specify level.
    level : int, str, or sequence, optional
        If the DataFrame has a MultiIndex, group by specific level(s) of the index.
        Can be level number(s) or name(s). If None, must specify by.
    **kwargs
        Additional keyword arguments (for pandas compatibility, mostly ignored)

    Returns
    -------
    DataFrameGroupBy
        groupby-lib DataFrameGroupBy object with optimized performance

    Examples
    --------
    >>> import pandas as pd
    >>> from groupby_lib.groupby import install_groupby_fast
    >>> install_groupby_fast()  # Add groupby_fast methods
    >>>
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
    >>> groups = ['X', 'Y', 'X', 'Y']
    >>> result = df.groupby_fast(groups).sum()
    >>> print(result)
       A   B
    X  4  40
    Y  6  60
    """
    return DataFrameGroupBy._from_by_keys(self, by=by, level=level)


def install_groupby_fast(patch_name: str = "groupby_fast"):
    """
    Install groupby_fast methods on pandas Series and DataFrame classes.

    This function monkey-patches pandas to add optimized groupby_fast methods
    that use the groupby-lib GroupBy implementation. After calling this function,
    all Series and DataFrame objects will have a .groupby_fast() method available.

    Examples
    --------
    >>> import pandas as pd
    >>> from groupby_lib.groupby import install_groupby_fast
    >>>
    >>> # Install the fast groupby methods
    >>> install_groupby_fast()
    >>>
    >>> # Now all pandas objects have groupby_fast
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> gb = df.groupby_fast(['X', 'Y', 'X'])  # Uses groupby-lib implementation
    >>> result = gb.sum()
    """
    # Add methods to the classes
    setattr(pd.Series, patch_name, groupby_fast_series)
    setattr(pd.DataFrame, patch_name, groupby_fast_dataframe)
    if polars_installed:
        setattr(pl.Series, patch_name, groupby_fast_series)
        setattr(pl.DataFrame, patch_name, groupby_fast_dataframe)

    print("✅ groupby-lib patches installed methods installed!")
    print(
        f"   Use df.{patch_name}() and series.{patch_name}() for optimized performance"
    )


def uninstall_groupby_fast(patch_name: str = "groupby_fast"):
    """
    Remove groupby_fast methods from pandas Series and DataFrame classes.

    This function removes the monkey-patched methods, restoring pandas to its
    original state. Useful for cleanup or testing.
    """
    if polars_installed:
        try:
            delattr(pl.Series, patch_name)
            delattr(pl.DataFrame, patch_name)
        except AttributeError:
            print(f"⚠️ groupby-lib {patch_name} methods were not found in polars")

        try:
            delattr(pd.Series, patch_name)
            delattr(pd.DataFrame, patch_name)
            print(f"✅ groupby-lib {patch_name} methods removed")
        except AttributeError:
            print(f"⚠️ groupby-lib {patch_name} methods were not found")


# Optional: Auto-install when module is imported
# Uncomment the next line if you want automatic installation
# install_groupby_fast()
