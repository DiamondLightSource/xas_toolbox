from functools import partial
from typing import Literal

import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.stats import t

from xas_toolbox.xas.edges import calc_e0


def _calculate_lambda(alpha: float, n: int, i: int) -> float:
    """
    Calculate test statistic (lambda), this is significance
    level that a given number of points (`n`) are outliers.
    """
    p = 1 - (alpha / (2 * (n - i + 1)))
    tppf = t.ppf(p, n - i - 1)
    lambda_i = np.divide(((n - i) * tppf), np.sqrt((n - i - 1 + tppf**2) * (n - i + 1)))

    return lambda_i


def gesd_outlier_detection(
    y: np.ndarray, filter_size: int, p: float = 0.05, no_outs: int = 50
) -> np.ndarray[int] | list[np.ndarray[int]]:
    """
    Using the generalized extreme student deviate test, identify potential outlier \
        points in an array.

    Arguments:
        y (np.ndarray): Data to peform outlier test on.
        filter_size (int): Size of median filter used on data to compute \
                            stats relative to.
        p (float, Optional): p-value corresponding to significance level used\
                            in t-distrbution in `calculate_lambda()` \
                            (p=0.05 = 95% significance).
        no_outs (int, Optional): Maximum number of points in data (axis=1 if `y` is nd)\
                                 to flag as outliers.

    Returns:
        outlier_points (np.ndarray[int] | list[np.ndarray[int]]):\
                             Array/list of arrays of indexes of outlier points.

    Refs:
        <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm>
    """
    if y.ndim == 1:
        ydiff = y - median_filter(y, filter_size)

        where_out = []

        for i in range(1, no_outs + 1):
            _mean, _std = np.nanmean(ydiff), np.nanstd(ydiff)
            pos = np.nanargmax(np.abs(ydiff - _mean))
            ri = np.nanmax(np.abs(ydiff - _mean)) / _std
            li = _calculate_lambda(p, ydiff.size - i, 1)
            if ri < li:
                break
            ydiff[pos] = np.nan
            where_out = np.sort(np.arange(0, ydiff.size)[np.isnan(ydiff)])
        outlier_points = np.array(where_out, dtype=int).flatten()
    else:
        ydiff = y - np.apply_along_axis(
            partial(median_filter, size=filter_size), axis=1, arr=y
        )
        where_out = []
        outlier_points = []
        for k in range(ydiff.shape[0]):
            where_out_tmp = []
            diff = ydiff[k, :]
            for i in range(1, no_outs + 1):
                _mean, _std = np.nanmean(diff), np.nanstd(diff)
                pos = np.nanargmax(np.abs(diff - _mean))
                ri = np.nanmax(np.abs(diff - _mean)) / _std
                ln = _calculate_lambda(p, diff.size - i, 1)
                if ri < ln:
                    break
                diff[pos] = np.nan
                where_out_tmp = np.sort(np.arange(0, diff.size)[np.isnan(diff)])
                outlier_idx = np.array(where_out_tmp).flatten()
                outlier_points.append(outlier_idx)

    return outlier_points


def normal_outlier_detection(
    y: np.ndarray,
    filter_size: int,
    order: int,
    tol: float,
    method: Literal["stdev", "MAD"] = "stdev",
) -> np.ndarray[int] | list[np.ndarray[int]]:
    """
    Outlier points identified based on standard deviation or MAD values
      (reliance on data being normally distributed hence the name..).
    """

    outliers = []

    yf = savgol_filter(y, filter_size, order)

    if yf.ndim > 1:
        if method == "stdev":
            lim = tol * np.std(y - yf, axis=1) + np.median(y - yf, axis=1)
        else:
            lim = 2 * tol * np.median(
                np.abs(y - yf - np.median(y - yf, axis=0)), axis=1
            ) + np.median(y - yf, axis=1)
        for i in range(y.shape[0]):
            outs_tmp = np.where(np.abs(y[i, :] - yf[i, :]) > lim[i])[0]
            outliers.append(outs_tmp)
    else:
        if method == "stdev":
            lim = tol * np.std(y - yf) + np.median(y - yf)
        else:
            lim = 2 * tol * np.median(np.abs(y - yf - np.median(y - yf))) + np.median(
                y - yf
            )
        outliers = np.where(np.abs(y - yf) > lim)[0]

    return outliers


def remove_edge_jump(
    x: np.ndarray,
    y: np.ndarray,
    outliers: np.ndarray | list[np.ndarray],
    pre: float = 30,
    post: float = 50,
) -> np.ndarray[int] | list[np.ndarray[int]]:
    """
    Remove outlier coordinates if they are in a region around the absorption
    edge.

    Arguments:
        x (np.ndarray): Energy array (must be 1d).
        y (np.ndarray): Absorption array.
        outliers (np.ndarray | list[np.ndarray]): Outlier mask/coordinates.
        pre (float, Optional): Energy before an edge to start the region (default = 30).
        post (float, Optional): Energy after an edge to end the region (default = 50).

    Returns:
        outliers (np.ndarray | list[np.ndarray]): Outlier mask/coordinates with\
                outliers flagged in the edge-region removed.
    """
    e0, edge_coords, edge_no = calc_e0(x, y)

    if y.ndim == 1:
        for e in edge_coords:
            lower = np.where(x <= x[e] - pre)[0][-1]
            upper = np.where(x <= x[e] + post)[0][-1]

            outliers = np.array([o for o in outliers if lower > o or upper < o])
    else:
        for i in range(len(edge_coords)):
            ecs = edge_coords[i]
            for e in ecs:
                lower = np.where(x <= x[e] - pre)[0][-1]
                upper = np.where(x <= x[e] + post)[0][-1]

                outliers[i] = np.array(
                    [o for o in outliers[i] if lower > o or upper < o], dtype=int
                )

    return outliers


def detect_outlier_scans(y: np.ndarray, scaling: float = 3) -> np.ndarray | None:
    """
    Routine previously from b18 repetitions (autoprocessing) for
    identifying if a scan in a series of repetition scans is an outlier.

    Arguments:
        y (np.ndarray): Stacked absorption data.
        scaling (float, optional): Tolerance factor for outlier scans (larger = higher \
            tolerance).

    Returns:
        out_pos (np.ndarray | None): Scan numbers for outliers (`None` if no outliers\
              found).
    """
    y_n = (y.T - y.mean(axis=1)) / y.std(axis=1).T
    y_med = np.median(y_n, axis=0)

    fit_data = np.ones([4, y_med.shape[0]])
    fit_data[0, :] = y_med
    fit_data[2, :] = np.arange(y_med.shape[0])
    fit_data[3, :] = np.arange(y_med.shape[0]) ** 2

    fitted = lstsq(fit_data.T, y_n.T)[0]
    med_baseline = fit_data.T @ fitted
    med_diff = y_n - med_baseline.T

    med_ssq = (med_diff**2).sum(axis=1)
    mad = np.median(np.abs(med_ssq - np.median(med_ssq)))
    vals = np.abs(med_ssq - np.mean(med_ssq)) - mad

    print(f"MAD is {mad}")
    print(f"Max of test values is {vals.max()}")

    test = vals > mad * scaling
    if np.any(test):
        out_pos = np.nonzero(test)[0]
        print(f"Outlier indices {out_pos}")
        return out_pos
    return None
