import numba as nb
import numpy as np
import pandas as pd

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
def _ema_adjusted(arr, alpha):
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
def _ema_unadjusted(arr, alpha):
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
def _ema_time_weighted(arr, times, halflife):
    out = np.zeros_like(arr, dtype="float64")
    residual = out[0] = arr[0]
    residual_weights = 1
    for i, x in enumerate(arr[1:], 1):
        hl = halflife / (times[i] - times[i - 1])
        beta = np.exp(-np.log(2) / hl)
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


def ema(
    values: np.ndarray | pd.Series,
    alpha: float = None,
    halflife: float = None,
    times=None,
    adjust=True,
):
    """Exponentially-weighted moving average (EWMA).

    Parameters
    ----------
    arr : array-like
        Input array.
    alpha : float, default 0.5
        Smoothing factor, between 0 and 1. Higher values give more weight to recent data.
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
        halflife = pd.Timedelta(halflife).total_seconds()
        if halflife <= 0:
            raise ValueError("Halflife must be positive.")
        if len(times) != len(values):
            raise ValueError("Times and values must have the same length.")

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
def _ema_grouped(group_key, values, alpha, ngroups):
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
def _ema_grouped_timed(group_key, values, times, halflife, ngroups):
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
