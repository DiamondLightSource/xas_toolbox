from functools import partial

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_deglitch(y: np.ndarray, window: int | float, threshold: float) -> np.ndarray:
    """
    Apply a median filter on y data. Values with values above standard
    deviation of `data - median filtered * threshold` will be replaced with the
    median filtered value.

    Arguments:
        y (np.ndarray): y data to filter.
        window (int|float): Window size for rolling median filter.
        threshold (float): Tolerance for replacement with filtered value.

    Returns:
        yf (np.ndarray): Filtered array.
    """
    # it would be nice to have auto window + threshold finding (would require a \
    # bit of prior knowledge though)
    if window % 2 == 0:
        window += 1
    pad_width = window // 2

    if y.ndim == 1:
        yf = np.pad(y, pad_width, mode="edge")
        yf = np.mean(sliding_window_view(np.array(yf), window), axis=1)
        tol = np.std(yf - y)
        yf = np.where(np.abs(yf - y) > tol * threshold, y, yf)
    else:
        yf = np.apply_along_axis(
            partial(np.pad, pad_width=pad_width, mode="edge"), axis=1, arr=y
        )
        yf = np.mean(sliding_window_view(yf, window, axis=1), axis=-1)
        tol = np.std(yf - y, axis=1)

        # could be changed..?
        for i in range(y.shape[0]):
            yf[i, :] = np.where(
                np.abs(yf[i, :] - y[i, :]) > tol[i] * threshold, y[i, :], yf[i, :]
            )

    return yf
