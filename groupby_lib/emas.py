from typing import Optional

import numba as nb
import numpy as np
import pandas as pd

from .util import check_data_inputs_aligned, _convert_timestamp_to_tz_unaware


_COMMON_VALUE_TYPES = [
    nb.types.Array(dtype, 1, "A", readonly=readonly)
    for dtype in [nb.types.float32, nb.types.int32, nb.types.float64, nb.types.int64]
    for readonly in (False, True)
]


_EMA_SIGNATURES = [
    nb.types.float64[:](arr_type, nb.types.float64) for arr_type in _COMMON_VALUE_TYPES
]


@nb.njit(nogil=True, cache=True)
def _ema_adjusted(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate exponentially-weighted moving average using the adjusted formula.

    The adjusted formula accounts for the imbalance in weights at the beginning
    of the series by maintaining residual sums and weights. This provides more
    accurate results at the start of the series compared to the unadjusted formula.

    Parameters
    ----------
    arr : np.ndarray
        Input array of values (float32, float64, int32, or int64).
    alpha : float
        Smoothing factor between 0 and 1. Higher values give more weight to recent data.

    Returns
    -------
    np.ndarray
        Exponentially-weighted moving average as float64 array.

    Notes
    -----
    The adjusted formula maintains running residuals and weights:
    - residual = sum of past values weighted by (1-alpha)^t
    - residual_weights = sum of (1-alpha)^t
    - out[i] = (x[i] + residual) / (1 + residual_weights)

    NaN values propagate the last valid EMA value forward.
    """
    out = np.zeros_like(arr, dtype="float64")
    beta = 1 - alpha
    residual = 0
    residual_weights = 0
    for i, x in enumerate(arr):
        if np.isnan(x):
            out[i] = out[i - 1]
        else:
            out[i] = (x + residual) / (1 + residual_weights)
            residual_weights += 1
            residual += x

        residual *= beta
        residual_weights *= beta

    return out


@nb.njit(nogil=True, cache=True)
def _ema_unadjusted(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate exponentially-weighted moving average using the unadjusted formula.

    The unadjusted formula applies a simple recursive exponential weighting without
    accounting for the initial bias. This is computationally simpler but may be less
    accurate at the beginning of the series.

    Parameters
    ----------
    arr : np.ndarray
        Input array of values (float32, float64, int32, or int64).
    alpha : float
        Smoothing factor between 0 and 1. Higher values give more weight to recent data.

    Returns
    -------
    np.ndarray
        Exponentially-weighted moving average as float64 array.

    Notes
    -----
    The unadjusted formula uses simple recursion:
    - out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    - First value is used as-is: out[0] = arr[0]

    NaN values propagate the last valid EMA value forward.
    """
    out = arr.astype("float64")
    beta = 1 - alpha
    for i, x in enumerate(arr[1:], 1):
        if np.isnan(x):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * x + beta * out[i - 1] if i > 0 else x

    return out


@nb.njit(nogil=True, cache=True)
def _ema_time_weighted(arr: np.ndarray, times: np.ndarray, halflife: int) -> np.ndarray:
    """
    Calculate time-weighted exponentially-weighted moving average.

    This function computes an EMA where the decay factor is adjusted based on
    the actual time elapsed between observations. This is useful for irregularly
    spaced time series data.

    Parameters
    ----------
    arr : np.ndarray
        Input array of values (float32, float64, int32, or int64).
    times : np.ndarray
        Array of integer timestamps (typically nanoseconds since epoch).
        Must be the same length as arr and monotonically increasing.
    halflife : int
        Halflife in nanoseconds.
        The decay factor is calculated such that the weight is reduced by half after this time interval.

    Returns
    -------
    np.ndarray
        Time-weighted exponentially-weighted moving average as float64 array.

    Notes
    -----
    The time-weighted formula adjusts the decay factor based on elapsed time:
    - For each step, beta = exp(-log(2) / (halflife / time_delta))
    - residual and residual_weights are multiplied by beta
    - out[i] = (x[i] + residual) / (1 + residual_weights)

    The first value is used as-is: out[0] = arr[0].
    NaN values propagate the last valid EMA value forward.
    """
    out = np.zeros_like(arr, dtype="float64")
    residual = out[0] = arr[0]
    residual_weights = 1
    for i, x in enumerate(arr[1:], 1):
        hl = (times[i] - times[i - 1]) / halflife
        beta = np.exp(-np.log(2) * hl)
        residual *= beta
        residual_weights *= beta

        if np.isnan(x):
            out[i] = out[i - 1]
        else:
            out[i] = (x + residual) / (1 + residual_weights)
            residual_weights += 1
            residual += x

    return out


def _halflife_to_int(halflife):
    halflife = pd.Timedelta(halflife).value
    if halflife <= 0:
        raise ValueError("Halflife must be positive.")
    return halflife


def _times_to_int_array(times):
    times, _ = _convert_timestamp_to_tz_unaware(times)
    return times.view(np.int64)


@check_data_inputs_aligned("values, times")
def ema(
    values: np.ndarray | pd.Series,
    alpha: Optional[float] = None,
    halflife: Optional[str | pd.Timedelta] = None,
    times: Optional[np.ndarray | pd.DatetimeIndex] = None,
    adjust: bool = True,
) -> np.ndarray | pd.Series:
    """Exponentially-weighted moving average (EWMA).

    Parameters
    ----------
    arr : array-like
        Input array.
    alpha : float, default 0.5
        Smoothing factor, between 0 and 1. Higher values give more weight to recent data.
    halflife: str | pd.Timedelta, e.g. "1s"
        Define the decay rate as halflife using a pd.Timedelta or a string compatible with same.
    times : array-like, optional
        Array of timestamps corresponding to the input data. If provided, the EWMA will be time
        weighted based on the halflife parameter.
    adjust : bool, default True
        If True, use the adjusted formula which accounts for the imbalance in weights at the beginning of the series.

    Returns
    -------
    np.ndarray
        The exponentially-weighted moving average of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from groupby_lib.ema import ema
    >>> data = np.array([1, 2, 3, 4, 5], dtype=float)
    >>> ema(data, alpha=0.5)
    array([1.        , 1.66666667, 2.42857143, 3.26666667, 4.16129032])
    >>> ema(data, alpha=0.5, adjust=False)
    array([1.    , 1.5   , 2.25  , 3.125 , 4.0625])

    Notes
    -----
    The EWMA is calculated using the formula:

        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]

    where y[t] is the EWMA at time t, x[t] is the input value at time t,
    and alpha is the smoothing factor.

    When `adjust` is True, the formula accounts for the imbalance in weights at the beginning of the series.
    """
    arr = np.asarray(values)

    def _maybe_to_series(result):
        if isinstance(values, pd.Series):
            return pd.Series(result, index=values.index, name=values.name)
        return result

    if times is not None:
        if halflife is None:
            raise ValueError("Halflife must be provided when times are given.")
        halflife = _halflife_to_int(halflife)

        times = _times_to_int_array(times)
        ema = _ema_time_weighted(arr, times, halflife)
        return _maybe_to_series(ema)

    if halflife is not None:
        if alpha is not None:
            raise ValueError("Only one of alpha or halflife should be provided.")

        if halflife <= 0:
            raise ValueError("Halflife must be positive.")

        alpha = 1 - np.exp(-np.log(2) / halflife)

    elif alpha is None:
        raise ValueError("One of alpha or halflife must be provided.")
    else:
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")

    if values.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")
    if not (0 < alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    if adjust:
        ema = _ema_adjusted(arr, alpha)
    else:
        ema = _ema_unadjusted(arr, alpha)

    return _maybe_to_series(ema)


_KEY_TYPES = [
    nb.types.Array(int_type, 1, "A", readonly=readonly)
    for int_type in [nb.types.int8, nb.types.int16, nb.types.int32, nb.types.int64]
    for readonly in (False, True)
]

_EMA_SIGNATURES_GROUPED = [
    nb.types.float64[:](
        key_type,
        arr_type,
        nb.types.float64,
        nb.types.int64,
        nb.types.optional(nb.types.bool[:]),
    )
    for key_type in _KEY_TYPES
    for arr_type in _COMMON_VALUE_TYPES
]


@nb.njit(nogil=True, cache=True)
def _ema_grouped(
    group_key: np.ndarray,
    values: np.ndarray,
    alpha: float,
    ngroups: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate exponentially-weighted moving average by group.

    This is the core numba-compiled function for computing grouped EMA. Each group
    maintains its own state (residuals, weights, last seen value) and the EMA is
    calculated independently within each group.

    Parameters
    ----------
    group_key : np.ndarray
        Integer array of group identifiers (int64). Values must be in range [0, ngroups).
    values : np.ndarray
        Array of values to compute EMA for (float32, float64, int32, or int64).
        Must be the same length as group_key.
    alpha : float
        Smoothing factor between 0 and 1. Higher values give more weight to recent data.
    ngroups : int
        Total number of groups (max(group_key) + 1).

    Returns
    -------
    np.ndarray
        Exponentially-weighted moving average as float64 array, same shape as values.

    Notes
    -----
    State is maintained per group using arrays indexed by group_key:
    - residuals[k]: weighted sum of past values for group k
    - residual_weights[k]: sum of weights for group k
    - last_seen[k]: last computed EMA value for group k (for NaN handling)

    The adjusted formula is used (similar to pandas adjust=True):
    - out[i] = (x[i] + residuals[k]) / (1 + residual_weights[k])

    NaN values propagate the last valid EMA value for that group.
    Groups are processed in the order they appear in the data.
    """
    out = np.zeros_like(values, dtype="float64")
    beta = 1 - alpha
    residuals = np.zeros(ngroups, dtype="float64")
    residual_weights = np.zeros(ngroups, dtype="float64")
    last_seen = np.full(ngroups, np.nan, dtype="float64")

    masked = mask is not None

    for i, (k, x) in enumerate(zip(group_key, values)):
        if np.isnan(x) or (masked and not mask[i]):
            out[i] = last_seen[k]
        else:
            out[i] = (x + residuals[k]) / (1 + residual_weights[k])
            residual_weights[k] += 1
            residuals[k] += x

        residuals[k] *= beta
        residual_weights[k] *= beta

        last_seen[k] = out[i]

    return out


_EMA_SIGNATURES_GROUPED_TIMED = [
    nb.types.float64[:](
        key_type,
        arr_type,
        nb.types.int64[:],
        nb.types.int64,
        nb.types.int64,
        nb.types.optional(nb.types.bool[:]),
    )
    for key_type in _KEY_TYPES
    for arr_type in _COMMON_VALUE_TYPES
]


@nb.njit(nogil=True, cache=True)
def _ema_grouped_timed(
    group_key: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    halflife: int,
    ngroups: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate time-weighted exponentially-weighted moving average by group.

    This is the core numba-compiled function for computing grouped time-weighted EMA.
    Each group maintains its own state (residuals, weights, last seen time, last seen value)
    and the EMA decay factor is adjusted based on the actual time elapsed within each group.

    Parameters
    ----------
    group_key : np.ndarray
        Integer array of group identifiers (int64). Values must be in range [0, ngroups).
    values : np.ndarray
        Array of values to compute EMA for (float32, float64, int32, or int64).
        Must be the same length as group_key.
    alpha : float, default 0.5
        Smoothing factor, between 0 and 1. Higher values give more weight to recent data.
    halflife: str | pd.Timedelta, e.g. "1s"
        Define the decay rate as halflife using a pd.Timedelta or a string compatible with same.
    times : array-like, optional
        Array of timestamps corresponding to the input data. If provided, the EWMA will be time
        weighted based on the halflife parameter.
    adjust : bool, default True
        If True, use the adjusted formula which accounts for the imbalance in weights at the beginning of the series.
    ngroups : int
        Total number of groups (max(group_key) + 1).

    Returns
    -------
    np.ndarray
        Time-weighted exponentially-weighted moving average as float64 array,
        same shape as values.

    Notes
    -----
    State is maintained per group using arrays indexed by group_key:
    - residuals[k]: weighted sum of past values for group k
    - residual_weights[k]: sum of weights for group k
    - last_seen_times[k]: timestamp of last observation in group k
    - last_seen[k]: last computed EMA value for group k (for NaN handling)

    The time-weighted formula adjusts decay based on elapsed time:
    - beta = exp(-log(2) / (halflife / time_delta))
    - out[i] = (x[i] + residuals[k]) / (1 + residual_weights[k])

    For the first observation in each group, no decay is applied (last_seen_times[k] == 0).
    NaN values propagate the last valid EMA value for that group.
    """
    out = np.zeros_like(values, dtype="float64")
    residuals = np.zeros(ngroups, dtype="float64")
    residual_weights = np.zeros(ngroups, dtype="float64")
    last_seen_times = np.zeros(ngroups, dtype="int64")
    last_seen = np.full(ngroups, np.nan, dtype="float64")

    masked = mask is not None

    for i, (k, x) in enumerate(zip(group_key, values)):
        if last_seen_times[k] > 0:
            hl = (times[i] - last_seen_times[k]) / halflife
            beta = np.exp(-np.log(2) * hl)
            residuals[k] *= beta
            residual_weights[k] *= beta

        if np.isnan(x) or (masked and not mask[i]):
            out[i] = last_seen[k]
        else:
            out[i] = (x + residuals[k]) / (1 + residual_weights[k])
            residual_weights[k] += 1
            residuals[k] += x

        last_seen_times[k] = times[i]
        last_seen[k] = out[i]

    return out


@check_data_inputs_aligned("group_key", "values", "times", "mask")
def ema_grouped(
    group_key: np.ndarray | pd.Series,
    ngroups: int,
    values: np.ndarray | pd.Series,
    alpha: Optional[float] = None,
    halflife: Optional[str | pd.Timedelta] = None,
    times: Optional[np.ndarray | pd.DatetimeIndex] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray | pd.Series:
    """Exponentially-weighted moving average (EWMA) by group.

    Computes an exponential moving average for each group independently.
    Each group maintains its own state and the EMA is calculated
    within each group separately.

    Parameters
    ----------
    group_key : array-like
        Group identifiers. Must be integers or convertible to integers.
    values : array-like
        Input values to compute EMA for.
    alpha : float, optional
        Smoothing factor, between 0 and 1. Higher values give more weight
        to recent data. Either alpha or halflife must be provided (not both).
    halflife : float, optional
        Halflife for the exponential decay. Either alpha or halflife must
        be provided (not both).
    times : array-like, optional
        Array of timestamps corresponding to the input data. If provided,
        the EWMA will be time-weighted based on the halflife parameter.
        Must be the same length as values.

    Returns
    -------
    np.ndarray or pd.Series
        The exponentially-weighted moving average for each group.
        Returns the same type as the input values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from groupby_lib.ema import ema_grouped
    >>>
    >>> # Simple grouped EMA
    >>> groups = np.array([0, 0, 0, 1, 1, 1])
    >>> values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    >>> result = ema_grouped(groups, values, alpha=0.5)
    >>>
    >>> # With pandas Series
    >>> groups = pd.Series([0, 0, 0, 1, 1, 1])
    >>> values = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    >>> result = ema_grouped(groups, values, halflife=2)
    >>>
    >>> # Time-weighted EMA
    >>> times = pd.date_range('2024-01-01', periods=6, freq='1h')
    >>> result = ema_grouped(groups, values, halflife='2h', times=times)

    Notes
    -----
    - Groups are processed in the order they appear in the data
    - Each group's EMA starts fresh (no information carries between groups)
    - NaN values in the input will propagate the last valid EMA value
    - The adjusted formula is always used (similar to adjust=True in ema)

    See Also
    --------
    ema : Exponential moving average for a single series
    """
    # Convert inputs to arrays
    group_key_arr = np.asarray(group_key)
    values_arr = np.asarray(values)

    if values_arr.dtype.kind not in ("f", "i"):
        raise TypeError("values must be numeric")

    # Validate dimensions first
    if values_arr.ndim != 1:
        raise ValueError("values must be one-dimensional")

    # Helper to convert back to Series if needed
    def _maybe_to_series(result):
        if isinstance(values, pd.Series):
            return pd.Series(result, index=values.index, name=values.name)
        return result

    # Handle empty input
    if len(group_key_arr) == 0:
        return _maybe_to_series(np.array([], dtype=np.float64))

    # Ensure group_key is integer type
    if group_key_arr.dtype.kind not in ("i", "u"):
        # Try to convert to integer
        try:
            group_key_arr = group_key_arr.astype(np.int64)
        except (ValueError, TypeError):
            raise ValueError("group_key must be integer or convertible to integer")

    if halflife is not None:
        if alpha is not None:
            raise ValueError("only one of alpha or halflife should be provided")

        halflife = _halflife_to_int(halflife)
        alpha = 1 - np.exp(-np.log(2) / halflife)

    nb_kwargs = dict(
        group_key=group_key_arr,
        values=values_arr,
        ngroups=ngroups,
    )
    if mask is not None:
        mask = np.asarray(mask)
    nb_kwargs["mask"] = mask

    # Handle time-weighted case
    if times is not None:
        if halflife is None:
            raise ValueError("halflife must be provided when times are given")

        nb_kwargs["halflife"] = halflife
        nb_kwargs["times"] = _times_to_int_array(times)
        result = _ema_grouped_timed(**nb_kwargs)
        return _maybe_to_series(result)

    if alpha is None:
        raise ValueError("one of alpha or halflife must be provided")
    else:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")

        # Compute grouped EMA
        result = _ema_grouped(**nb_kwargs, alpha=alpha)
        return _maybe_to_series(result)
