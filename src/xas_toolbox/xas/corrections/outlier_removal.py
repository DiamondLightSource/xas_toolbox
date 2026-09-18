# "outlier_correction" / "remove_outliers_b18" from XAS_tools.

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal.windows import tukey

from xas_toolbox.utils.fitting import median_poly_fit
from xas_toolbox.utils.maths.start_stop import _get_start_stop


def fit_weighted_regions(
    y: np.ndarray, startstop: list[tuple[int, int]], yf: np.ndarray, pad: int
) -> np.ndarray:
    """
    All points in start-stop ranges replaced with a weighted fit of
    yf to y.

    Arguments:
        y (np.ndarray): Data to correct.
        startstop (list[tuple[int, int]]): List of (start, stop) \
        ranges in y for fitting.
        yf (np.ndarray): Data to fit to y in (start, stop) regions.
        pad (int: Padding applied to window in weighting function.

    Returns:
        out (np.ndarray): y with corrections in the startstop regions.
    """
    out = y.copy()
    # this assumes y.ndim = 1!

    for region in startstop:
        abs_range = int(np.abs(region[0] - region[1]) + pad)
        center = int(abs_range / 2 + region[0])
        win_size = int(3 * abs_range)
        fit_size = int(5 * abs_range)
        fit_range = [center - int(fit_size / 2), center - int(fit_size / 2) + fit_size]

        window = tukey(win_size)
        weight = np.zeros(fit_size)
        weight[abs_range : abs_range + win_size] = window
        weight = -weight + 1

        if fit_range[0] < 0:
            weight = weight[-fit_range[0] :]
            fit_range[0] = 0
        if fit_range[1] > y.size:
            weight = weight[: -(fit_range[1] - y.size)]
            fit_range[1] = y.size

        yfit = median_poly_fit(
            y[fit_range[0] : fit_range[-1]], yf[fit_range[0] : fit_range[-1]], weight
        ) * (-weight + 1)

        out[fit_range[0] : fit_range[1]] = (
            y[fit_range[0] : fit_range[-1]] * weight + yfit
        )

    return out


# having a method to straight-up remove outlier points and interpolate could be useful.


def correct_outliers(
    y: np.ndarray,
    outliers: list[int] | list[np.ndarray],
    window: int = 21,
    min_spacing: int = 20,
    pad: int = 10,
) -> np.ndarray:
    """
    Replace regions in y which are flagged as containing outlier points with
    a weighted fit to median-filtered y.

    Arguments:
        y (np.ndarray): Data to correct.
        outliers (list[int] | list[np.ndarray]): List/nested list of outlier points.
        window (int, Optional): Window size for median filter.
        min_spacing (int, Optional): Minimum spacing between "Outlier Regions".
        pad (int, Optional): Padding for window function giving fit weighting.

    Returns:
        y_corrected (np.ndarray): Corrected data.
    """
    if y.ndim == 1:
        yf = median_filter(y, size=window)
        startstop = _get_start_stop(outliers, np.diff(outliers), min_spacing)
        y_corrected = fit_weighted_regions(y, startstop, yf, pad)

    else:
        yf = median_filter(y, size=window, axes=0)
        y_corrected = np.zeros_like(y)
        for i in range(y.shape[0]):
            startstop = _get_start_stop(outliers[i], np.diff(outliers[i]), min_spacing)
            _y_corrected = fit_weighted_regions(y[i, :], startstop, yf[i, :], pad)
            y_corrected[i, :] = _y_corrected

    return y_corrected
