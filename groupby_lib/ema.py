from typing import Optional

import numba as nb
import numpy as np
import pandas as pd

from .util import check_data_inputs_aligned


_EMA_SIGNATURES = [
    nb.types.float64[:](arr_type, nb.types.float64)
    for arr_type in (
        nb.types.float32[:],
        nb.types.float64[:],
        nb.types.int64[:],
        nb.types.int32[:],
    )
]


@nb.njit(_EMA_SIGNATURES, nogil=True, cache=True)
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


@nb.njit(_EMA_SIGNATURES, nogil=True, cache=True)
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


@nb.njit(
    [
        nb.types.float64[:](arr_type, nb.types.int64[:], nb.types.float64)
        for arr_type in (
            nb.types.float32[:],
            nb.types.float64[:],
            nb.types.int64[:],
            nb.types.int32[:],
        )
    ],
    nogil=True,
    cache=True,
)
def _ema_time_weighted(
    arr: np.ndarray, times: np.ndarray, halflife: int
) -> np.ndarray:
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


_ema_adjusted._can_cache = True
_ema_unadjusted._can_cache = True
_ema_time_weighted._can_cache = True


def _halflife_to_int(halflife):
    halflife = pd.Timedelta(halflife).value
    if halflife <= 0:
        raise ValueError("Halflife must be positive.")
    return halflife


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

        times = np.asarray(times).view(np.int64)
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


_EMA_SIGNATURES_GROUPED = [
    nb.types.float64[:](nb.types.int64[:], arr_type, nb.types.float64, nb.types.int64)
    for arr_type in (
        nb.types.float32[:],
        nb.types.float64[:],
        nb.types.int64[:],
        nb.types.int32[:],
    )
]


@nb.njit(_EMA_SIGNATURES_GROUPED, nogil=True, cache=True)
def _ema_grouped(
    group_key: np.ndarray, values: np.ndarray, alpha: float, ngroups: int
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

    for i, (k, x) in enumerate(zip(group_key, values)):
        if np.isnan(x):
            out[i] = last_seen[k]
        else:
            out[i] = (x + residuals[k]) / (1 + residual_weights[k])
            residual_weights[k] += 1
            residuals[k] += x

        residuals[k] *= beta
        residual_weights[k] *= beta

        last_seen[k] = out[i]

    return out


_ema_grouped._can_cache = True


_EMA_SIGNATURES_GROUPED_TIMED = [
    nb.types.float64[:](
        nb.types.int64[:], arr_type, nb.types.int64[:], nb.types.float64, nb.types.int64
    )
    for arr_type in (
        nb.types.float32[:],
        nb.types.float64[:],
        nb.types.int64[:],
        nb.types.int32[:],
    )
]


@nb.njit(_EMA_SIGNATURES_GROUPED_TIMED, nogil=True, cache=True)
def _ema_grouped_timed(
    group_key: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    halflife: int,
    ngroups: int,
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

    for i, (k, x) in enumerate(zip(group_key, values)):
        if last_seen_times[k] > 0:
            hl = halflife / (times[i] - last_seen_times[k])
            beta = np.exp(-np.log(2) / hl)
            residuals[k] *= beta
            residual_weights[k] *= beta

        if np.isnan(x):
            out[i] = last_seen[k]
        else:
            out[i] = (x + residuals[k]) / (1 + residual_weights[k])
            residual_weights[k] += 1
            residuals[k] += x

        last_seen_times[k] = times[i]
        last_seen[k] = out[i]

    return out
