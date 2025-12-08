"""
Pandas-compatible API classes for groupby-lib GroupBy operations.

This module provides familiar pandas-like interfaces that utilize the optimized
groupby-lib GroupBy engine for better performance while maintaining full compatibility.
"""

from abc import ABC, abstractmethod
from functools import wraps, cached_property
from typing import Hashable, Optional, Tuple, Union, List, Callable, Dict

import numpy as np
import pandas as pd
import polars as pl

from .core import ArrayType1D, GroupBy


def groupby_aggregation(
    description: str,
    extra_params: str = "",
    include_numeric_only: bool = True,
    include_margins: bool = True,
    **docstring_params,
):
    """
    Decorator for SeriesGroupBy/DataFrameGroupBy aggregation methods.

    This decorator:
    1. Eliminates boilerplate return value processing
    2. Auto-generates consistent docstrings
    3. Handles mask parameter consistently

    Parameters
    ----------
    description : str
        Brief description of what the method does (e.g., "Compute sum of group values")
    extra_params : str, optional
        Additional parameter documentation to include
    **docstring_params : dict
        Additional parameters for docstring template
    """

    def decorator(func):
        method_name = func.__name__

        # Generate docstring
        param_docs = """        mask : ArrayType1D, optional
            Boolean mask to apply before aggregation"""

        if include_margins:
            param_docs += """
        margins : bool or list of int, default False
            Add margins (subtotals) to result. If list of integers,
            include margin rows for the specified levels only."""

        if include_numeric_only:
            param_docs = (
                """        numeric_only : bool, default True
            Include only numeric columns
"""
                + param_docs
            )

        if extra_params:
            param_docs = extra_params + "\n" + param_docs

        func.__doc__ = f"""
        {description}.

        Parameters
        ----------{param_docs}

        Returns
        -------
        pd.Series
            Series with group {method_name}s
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Call the core grouper method directly - it already returns
            # proper pandas objects
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def groupby_cumulative(description: str):
    """Decorator for cumulative operations."""

    def decorator(func):
        method_name = func.__name__

        func.__doc__ = f"""
        {description} for each group.

        Returns
        -------
        pd.Series
            Series with {method_name} values
        """

        @wraps(func)
        def wrapper(self):
            # Call the core grouper method directly
            return func(self)

        return wrapper

    return decorator


class BaseGroupBy(ABC):
    """
    Abstract base class for groupby-lib GroupBy API classes.

    This class contains common functionality shared between SeriesGroupBy
    and DataFrameGroupBy classes.
    """

    _grouper: GroupBy
    _obj: Union[pd.Series, pd.DataFrame]
    _values_to_group: Union[pd.Series, pd.DataFrame]

    @property
    def grouper(self) -> GroupBy:
        """Access to the underlying GroupBy engine."""
        return self._grouper

    @property
    def groups(self):
        """Dict mapping group names to row labels."""
        return self._grouper.groups

    @property
    def ngroups(self) -> int:
        """Number of groups."""
        return self._grouper.ngroups

    @property
    @abstractmethod
    def _values_to_group(self) -> Union[pd.Series, Dict[Hashable, pd.Series]]:
        """
        Extract values to be grouped from the object.

        This method should be implemented by subclasses to return the
        appropriate values (Series or dict of Series) to be grouped.

        Returns
        -------
        Union[pd.Series, Dict[Hashable, pd.Series]]
            Values to group
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ngroups={self.ngroups})"

    def __iter__(self) -> Tuple[Hashable, Union[pd.Series, pd.DataFrame]]:
        for key, indexer in self.groups.items():
            yield key, self._obj.loc[indexer]

    @groupby_aggregation("Compute sum of group values")
    def sum(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.sum(self._values_to_group, mask=mask, margins=margins)

    @groupby_aggregation("Compute mean of group values")
    def mean(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.mean(self._values_to_group, mask=mask, margins=margins)

    @groupby_aggregation(
        "Compute standard deviation of group values",
        extra_params="        ddof : int, default 1\n            Degrees of freedom",
    )
    def std(
        self, ddof: int = 1, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.std(
            self._values_to_group, ddof=ddof, mask=mask, margins=margins
        )

    @groupby_aggregation(
        "Compute variance of group values",
        extra_params="        ddof : int, default 1\n            Degrees of freedom",
    )
    def var(
        self, ddof: int = 1, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.var(
            self._values_to_group, ddof=ddof, mask=mask, margins=margins
        )

    @groupby_aggregation("Compute minimum of group values")
    def min(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.min(self._values_to_group, mask=mask, margins=margins)

    @groupby_aggregation("Compute maximum of group values")
    def max(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.max(self._values_to_group, mask=mask, margins=margins)

    @groupby_aggregation("Compute median of group values")
    def median(self, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.median(self._values_to_group, mask=mask)

    @groupby_aggregation("Compute quantiles of group values")
    def quantile(
        self,
        q: List[float] | np.ndarray,
        mask: Optional[ArrayType1D] = None,
    ) -> pd.Series:
        return self._grouper.quantile(self._values_to_group, q=q, mask=mask)

    @groupby_aggregation(
        "Compute count of non-null group values",
        include_numeric_only=False,
        include_margins=False,
    )
    def count(self, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.count(self._values_to_group, mask=mask)

    @groupby_aggregation(
        "Compute group sizes (including null values)",
        include_numeric_only=False,
        include_margins=False,
    )
    def size(self, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.size(mask=mask)

    @groupby_aggregation(
        "Get first non-null value in each group",
        extra_params=(
            "        numeric_only : bool, default False\n"
            "            Include only numeric columns"
        ),
        include_margins=False,
    )
    def first(
        self, numeric_only: bool = False, mask: Optional[ArrayType1D] = None
    ) -> pd.Series:
        return self._grouper.first(self._values_to_group, mask=mask)

    @groupby_aggregation(
        "Get last non-null value in each group",
        extra_params=(
            "        numeric_only : bool, default False\n"
            "            Include only numeric columns"
        ),
        include_margins=False,
    )
    def last(
        self, numeric_only: bool = False, mask: Optional[ArrayType1D] = None
    ) -> pd.Series:
        return self._grouper.last(self._values_to_group, mask=mask)

    def nth(self, n: int) -> pd.Series:
        """
        Take nth value from each group.

        Parameters
        ----------
        n : int
            Position to take (0-indexed)

        Returns
        -------
        pd.Series
            Series with nth values
        """
        result = self._grouper.nth(self._values_to_group, n)
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._obj.name)
        )

    def head(self, n: int = 5) -> pd.Series:
        """
        Return first n rows of each group.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        pd.Series
            Series with first n values from each group
        """
        result = self._grouper.head(self._obj, n)
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._obj.name)
        )

    def tail(self, n: int = 5) -> pd.Series:
        """
        Return last n rows of each group.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        pd.Series
            Series with last n values from each group
        """
        result = self._grouper.tail(self._obj, n)
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._obj.name)
        )

    def agg(self, func, mask: Optional[ArrayType1D] = None) -> pd.Series:
        """
        Apply aggregation function to each group.

        Parameters
        ----------
        func : str or callable
            Aggregation function name or callable

        Returns
        -------
        pd.Series
            Series with aggregated values
        """
        if isinstance(func, str):
            if hasattr(self, func):
                return getattr(self, func)()
            else:
                result = self._grouper.agg(self._obj, func)
        else:
            result = self._grouper.apply(self._obj, func)

        return result

    aggregate = agg  # Alias

    def apply(
        self,
        func,
        mask: Optional[np.ndarray] = None,
        *func_args,
        **func_kwargs,
    ) -> pd.Series:
        """
        Apply a custom function to each group.

        This method applies a numpy-compatible function to each group of values.
        The function is called with each group's values and any additional arguments
        provided. When mask is None, it uses the .groups attribute which contains a mapping
        from group label to fancy indexer, whihc is built once and cached.
        When mask is not None we must build the same mapping for the masked data.
        Results are computed in parallel for better performance.

        Parameters
        ----------
        values : ArrayCollection
            Values to apply the function to. Can be a single array/Series or a
            collection (list, dict) of arrays/Series.
        np_func : Callable
            Function to apply to each group. Should accept numpy arrays as input
            and work with the signature: np_func(array, *func_args, **func_kwargs).
        mask : np.ndarray, optional
            Boolean mask array indicating which rows to include. If provided,
            only rows where mask is True will be included in the calculation.
            Default is None (include all rows).
        *func_args
            Additional positional arguments to pass to np_func.
        **func_kwargs
            Additional keyword arguments to pass to npfunc.
        """
        return self._grouper.apply(self._obj, func, mask, *func_args, **func_kwargs)

    @groupby_cumulative("Cumulative sum")
    def cumsum(self) -> pd.Series:
        return self._grouper.cumsum(self._obj)

    @groupby_cumulative("Cumulative maximum")
    def cummax(self) -> pd.Series:
        return self._grouper.cummax(self._obj)

    @groupby_cumulative("Cumulative minimum")
    def cummin(self) -> pd.Series:
        return self._grouper.cummin(self._obj)

    @groupby_cumulative(
        "Number each item in each group from 0 to the length of that group - 1"
    )
    def cumcount(self) -> pd.Series:
        return self._grouper.cumcount(self._obj)

    def ema(
        self,
        alpha: Optional[float] = None,
        halflife: Optional[float] = None,
        times: Optional[ArrayType1D] = None,
        mask: Optional[ArrayType1D] = None,
        index_by_groups: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate exponentially-weighted moving average (EWMA) for each group.

        Computes an exponential moving average for each group independently.
        Each group maintains its own state and the EMA is calculated within
        each group separately.

        Parameters
        ----------
        alpha : float, optional
            Smoothing factor, between 0 and 1. Higher values give more weight
            to recent data. Either alpha or halflife must be provided (not both).
        halflife : float, optional
            Halflife for the exponential decay. Either alpha or halflife must
            be provided (not both). When used with times parameter, halflife
            should be a string (e.g., '1h', '30min') compatible with pd.Timedelta.
        times : array-like, optional
            Array of timestamps corresponding to the input data. If provided,
            the EWMA will be time-weighted based on the halflife parameter.
            Must be the same length as values.
        mask : np.ndarray, optional
            Boolean mask array indicating which rows to include. If provided,
            only rows where mask is True will be included in the calculation.
            Default is None (include all rows).
        index_by_groups : bool, default False
            If True, sort groups before calculating EMA for better performance.

        Returns
        -------
        pd.Series or pd.DataFrame
            The exponentially-weighted moving average for each group.
            Returns the same shape as input values (transform-style output).

        Examples
        --------
        >>> import pandas as pd
        >>> s = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        >>> groups = pd.Series([1, 1, 1, 2, 2, 2])
        >>> gb = s.groupby_fast(groups)
        >>> gb.ema(alpha=0.5)
        0     1.000000
        1     1.666667
        2     2.428571
        3    10.000000
        4    16.666667
        5    24.285714
        dtype: float64
        """
        return self._grouper.ema(
            self._obj,
            alpha=alpha,
            halflife=halflife,
            times=times,
            mask=mask,
            index_by_groups=index_by_groups,
        )


class SeriesGroupBy(BaseGroupBy):
    """
    A pandas-like SeriesGroupBy class that uses groupby-lib GroupBy as the engine.

    This class provides a familiar pandas interface while leveraging the optimized
    GroupBy implementation for better performance.

    Parameters
    ----------
    obj : pd.Series
        The pandas Series to group
    grouper : GroupBy
        The GroupBy engine instance

    Examples
    --------
    Basic grouping:
    >>> import pandas as pd
    >>> from groupby_lib.groupby import SeriesGroupBy
    >>> s = pd.Series([1, 2, 3, 4, 5, 6])
    >>> groups = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'])
    >>> gb = SeriesGroupBy._from_by_keys(s, by=groups)
    >>> gb.sum()
    A    9
    B   12
    dtype: int64

    Level-based grouping:
    >>> idx = pd.MultiIndex.from_tuples(\n    ...     [('A', 1), ('A', 2), ('B', 1)],\n    ...     names=['letter', 'num'])
    >>> s = pd.Series([10, 20, 30], index=idx)
    >>> gb = SeriesGroupBy._from_by_keys(s, level='letter')
    >>> gb.sum()
    A    30
    B    30
    dtype: int64
    """

    def __init__(self, obj: pd.Series, grouper: GroupBy):
        if not isinstance(obj, (pd.Series, pl.Series)):
            raise TypeError("obj must be a pandas Series")
        self._obj = obj
        self._grouper = grouper

    @classmethod
    def _from_by_keys(cls, obj: pd.Series, by=None, level=None) -> "SeriesGroupBy":
        """
        Create a SeriesGroupBy instance from by and level arguments.

        This is the constructor that should be used when creating a SeriesGroupBy
        from by/level arguments, as is done in the monkeypatch methods.

        Parameters
        ----------
        obj : pd.Series
            The pandas Series to group
        by : array-like, optional
            Grouping key(s), can be any type acceptable to core.GroupBy constructor.
            If None, must specify level.
        level : int, str, or sequence, optional
            If the Series has a MultiIndex, group by specific level(s) of the index.
            Can be level number(s) or name(s). If None, must specify by.

        Returns
        -------
        SeriesGroupBy
            A SeriesGroupBy instance with the constructed grouper
        """
        if by is None and level is None:
            raise ValueError("Must provide either 'by' or 'level' for grouping")

        grouping_keys = []

        # Process by argument first (to match pandas order)
        if by is not None:
            if isinstance(by, (list, tuple)):
                grouping_keys.extend(by)
            else:
                grouping_keys.append(by)

        # Process level argument
        if level is not None:
            # Resolve index levels inline
            if not isinstance(level, (list, tuple)):
                levels = [level]
            else:
                levels = level

            for lv in levels:
                grouping_keys.append(obj.index.get_level_values(lv))

        # Create the grouper
        grouper = GroupBy(grouping_keys)

        # Create the proper instance
        return cls(obj, grouper=grouper)

    @cached_property
    def _values_to_group(self) -> pd.Series:
        return self._obj

    def rolling(self, window: int, min_periods: Optional[int] = None):
        """
        Provide rolling window calculations within groups.

        Parameters
        ----------
        window : int
            Size of the moving window
        min_periods : int, optional
            Minimum number of observations required to have a value

        Returns
        -------
        SeriesGroupByRolling
            Rolling window object
        """
        return SeriesGroupByRolling(self, window, min_periods)


class BaseGroupByRolling:
    """
    Base class for rolling window operations on GroupBy objects.

    This class provides shared functionality for rolling window calculations
    within each group, reducing code duplication between Series and DataFrame
    rolling operations.

    Parameters
    ----------
    groupby_obj : SeriesGroupBy or DataFrameGroupBy
        The groupby object to apply rolling operations to
    window : int
        Size of the rolling window
    min_periods : int, optional
        Minimum number of observations required to have a value.
        Defaults to window size.
    """

    def __init__(
        self,
        groupby_obj: Union["SeriesGroupBy", "DataFrameGroupBy"],
        window: int,
        min_periods: Optional[int] = None,
    ):
        self._groupby_obj = groupby_obj
        self._window = window
        self._min_periods = min_periods if min_periods is not None else window

    def agg(
        self,
        method_name: str,
        mask: Optional[ArrayType1D] = None,
        index_by_groups: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply a rolling method and return the result.

        Parameters
        ----------
        method_name : str
            Name of the rolling method (e.g., 'sum', 'mean', 'min', 'max')
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation
        index_by_groups : bool, default False
            If True, the result has the sorted group keys as the outer index level
            similarly to pandas behavior. This is considerably slower than the default behavior.

        Returns
        -------
        pd.Series or pd.DataFrame
            Result of the rolling operation
        """
        method = getattr(self._groupby_obj._grouper, f"rolling_{method_name}")
        return self._format_result(
            method(
                self._groupby_obj._obj,
                window=self._window,
                min_periods=self._min_periods,
                mask=mask,
                index_by_groups=index_by_groups,
            )
        )

    def sum(
        self,
        mask: Optional[ArrayType1D] = None,
        index_by_groups: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling sum within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling sum values with same shape as input
        """
        return self.agg("sum", mask=mask, index_by_groups=index_by_groups)

    def mean(
        self,
        mask: Optional[ArrayType1D] = None,
        index_by_groups: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling mean within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling mean values with same shape as input
        """
        return self.agg("mean", mask=mask, index_by_groups=index_by_groups)

    def min(
        self,
        mask: Optional[ArrayType1D] = None,
        index_by_groups: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling minimum within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling minimum values with same shape as input
        """
        return self.agg("min", mask=mask, index_by_groups=index_by_groups)

    def max(
        self,
        mask: Optional[ArrayType1D] = None,
        index_by_groups: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling maximum within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling maximum values with same shape as input
        """
        return self.agg("max", mask=mask, index_by_groups=index_by_groups)

    def _format_result(self, result) -> Union[pd.Series, pd.DataFrame]:
        """Format the result according to the specific GroupBy type."""
        raise NotImplementedError("Subclasses must implement _format_result")


class SeriesGroupByRolling(BaseGroupByRolling):
    """
    Rolling window operations for SeriesGroupBy objects.

    This class provides rolling window calculations within each group,
    similar to pandas SeriesGroupBy.rolling().
    """

    def _format_result(self, result) -> pd.Series:
        """Format result as a pandas Series."""
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._groupby_obj._obj.name)
        )


class DataFrameGroupBy(BaseGroupBy):
    """
    A pandas-like DataFrameGroupBy class that uses groupby-lib GroupBy as the engine.

    This class provides a familiar pandas interface for DataFrame grouping operations
    while leveraging the optimized GroupBy implementation for better performance.

    Parameters
    ----------
    obj : pd.DataFrame
        The pandas DataFrame to group
    grouper : GroupBy
        The GroupBy engine instance
    columns_used_as_keys : set, optional
        Set of column names that were used as grouping keys

    Examples
    --------
    Basic grouping:
    >>> import pandas as pd
    >>> from groupby_lib.groupby import DataFrameGroupBy
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
    >>> groups = pd.Series(['X', 'Y', 'X', 'Y'])
    >>> gb = DataFrameGroupBy._from_by_keys(df, by=groups)
    >>> gb.sum()
        A   B
    X   4  40
    Y   6  60
    """

    def __init__(
        self,
        obj: pd.DataFrame,
        grouper: GroupBy,
        value_columns: Optional[List] = None,
    ):
        if not isinstance(obj, (pd.DataFrame, pl.DataFrame)):
            raise TypeError("obj must be a pandas DataFrame")
        self._obj = obj
        self._grouper = grouper
        if value_columns is not None:
            for col in value_columns:
                if col not in obj.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")
            self.value_columns = value_columns
        else:
            self.value_columns = obj.columns.tolist()

    @property
    def _values_to_group(self) -> Union[pd.Series, Dict[Hashable, pd.Series]]:
        """
        Extract values to be grouped from the object.

        This method should be implemented by subclasses to return the
        appropriate values (Series or dict of Series) to be grouped.

        Returns
        -------
        Union[pd.Series, Dict[Hashable, pd.Series]]
            Values to group
        """
        return pd.DataFrame(
            {col: self._obj[col] for col in self.value_columns},
            copy=False,
        )

    @classmethod
    def _from_by_keys(
        cls, obj: Union[pd.DataFrame, pl.DataFrame], by=None, level=None
    ) -> "DataFrameGroupBy":
        """
        Create a DataFrameGroupBy instance from by and level arguments.

        This is the constructor that should be used when creating a DataFrameGroupBy
        from by/level arguments, as is done in the monkeypatch methods.

        Parameters
        ----------
        obj : pd.DataFrame, pl.DataFrame
            The DataFrame to group
        by : array-like, optional
            Grouping key(s), can be any type acceptable to core.GroupBy constructor.
            If None, must specify level.
        level : int, str, or sequence, optional
            If the DataFrame has a MultiIndex, group by specific level(s) of the index.
            Can be level number(s) or name(s). If None, must specify by.

        Returns
        -------
        DataFrameGroupBy
            A DataFrameGroupBy instance with the constructed grouper
        """
        if not isinstance(obj, (pd.DataFrame, pl.DataFrame)):
            raise TypeError("obj must be a pandas DataFrame")
        
        if by is None and level is None:
            raise ValueError("Must provide either 'by' or 'level' for grouping")

        grouping_keys = []
        columns_used_as_keys = set()

        # Process by argument first (to match pandas order)
        if by is not None:
            # Resolve by keys inline
            # Special case: if by is a tuple and it's a valid column name,
            # treat as single key
            if isinstance(by, tuple) and by in obj.columns:
                by = [by]
            elif not isinstance(by, (list, tuple)):
                by = [by]

            for key in by:
                # Check for array-like objects first (before checking columns,
                # since arrays aren't hashable)
                if hasattr(key, "__iter__") and not isinstance(
                    key, (str, bytes, tuple)
                ):
                    # Array-like object (not string or tuple) - use directly
                    if hasattr(key, "__len__") and len(key) != len(obj):
                        raise ValueError(
                            f"Length of grouper ({len(key)}) != "
                            f"length of DataFrame ({len(obj)})"
                        )
                    grouping_keys.append(key)

                elif callable(key):
                    # Callable - apply to index
                    grouping_keys.append(obj.index.map(key))

                else:
                    # Try to use as column name (including tuple column names)
                    try:
                        if key in obj.columns:
                            grouping_keys.append(obj[key])
                            columns_used_as_keys.add(key)
                        elif hasattr(obj.index, "names") and key in obj.index.names:
                            # It's an index level name
                            if isinstance(obj.index, pd.MultiIndex):
                                level_idx = obj.index.names.index(key)
                                grouping_keys.append(
                                    obj.index.get_level_values(level_idx)
                                )
                            else:
                                # Single level index
                                grouping_keys.append(obj.index)
                        else:
                            raise KeyError(f"Column or index level '{key}' not found")
                    except TypeError:
                        # Unhashable type - treat as array-like if it has proper length
                        if hasattr(key, "__len__") and len(key) == len(obj):
                            grouping_keys.append(key)
                        else:
                            raise KeyError(f"Invalid grouping key: {key}")

        # Process level argument
        if level is not None:
            # Resolve index levels inline
            if not isinstance(level, (list, tuple)):
                levels = [level]
            else:
                levels = level

            for lv in levels:
                grouping_keys.append(obj.index.get_level_values(lv))

        # Create the grouper
        grouper = GroupBy(grouping_keys)

        # Create the proper instance
        value_columns = [col for col in obj.columns if col not in columns_used_as_keys]
        return cls(obj, grouper=grouper, value_columns=value_columns)

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            return super().__getattribute__(name)

    def __getitem__(self, key):
        """
        Select column(s) from the grouped DataFrame.

        Parameters
        ----------
        key : str or list
            Column name(s) to select

        Returns
        -------
        SeriesGroupBy or DataFrameGroupBy
            SeriesGroupBy if single column, DataFrameGroupBy if multiple columns
        """
        if isinstance(key, Hashable):
            subset = self._obj[key]
            return SeriesGroupBy(subset, grouper=self._grouper)
        else:
            # Multiple columns - return DataFrameGroupBy with subset
            return DataFrameGroupBy(
                self._obj,
                grouper=self._grouper,
                value_columns=key,
            )

    def rolling(self, window: int, min_periods: Optional[int] = None):
        """
        Provide rolling window calculations within groups.

        Parameters
        ----------
        window : int
            Size of the moving window
        min_periods : int, optional
            Minimum number of observations required to have a value

        Returns
        -------
        DataFrameGroupByRolling
            Rolling window object
        """
        return DataFrameGroupByRolling(self, window, min_periods)


class DataFrameGroupByRolling(BaseGroupByRolling):
    """
    Rolling window operations for DataFrameGroupBy objects.

    This class provides rolling window calculations within each group,
    similar to pandas DataFrameGroupBy.rolling(). It inherits from
    BaseGroupByRolling to reduce code duplication.
    """

    def _format_result(self, result) -> pd.DataFrame:
        """Format result as a pandas DataFrame."""
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
