"""
Different methods from find_e0 for edge-finding. <br>
These are mainly for low-noise f2 data.
"""

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def find_edges(y: np.ndarray, filter: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Find coordinates for potential edge-jumps in array of absorption
    values.

    Args:
        y (np.ndarray): (1d) Array of absorption values.
        filter (bool, Optional): Whether to apply a Gaussian filter to the y-data
              (default=`False`).
    Returns:
        tuple (tuple): tuple containing:
            where_edge (np.ndarray): Indices of y at the maximum derivative of a\
                  detected edge.
            where_pre (np.ndarray): Indices of y at the start of an edge region.
    """
    # would also like to test this versus. larch/xas-toolbox calc_e0.
    if y.ndim > 1:
        if filter is False:
            yf = np.round(np.gradient(y, axis=1), 1)
        else:
            yf = np.round(np.gradient(gaussian_filter(y, sigma=5), axis=1), 1)
    else:
        if filter is False:
            yf = np.round(np.gradient(y), 1)
        else:
            yf = np.round(np.gradient(gaussian_filter(y, sigma=5)), 1)

    yf = np.diff(np.where(yf <= 0, 0, 1), prepend=0)

    where_edge = np.where(yf < 0)[-1]
    where_pre = np.where(yf > 0)[0]

    return where_edge, where_pre


def get_edge_bounds(
    x: np.ndarray, y: np.ndarray, edge_energy: float
) -> tuple[Any, list[Any], Any]:
    """
    Find the closest edge and bounds around it given a target energy.

    Arguments:
        x (np.ndarray): (1d) x-array (energy).
        y (np.ndarray): (1d) y-array (usually f2/lni0it).
        edge_energy (float): Target absorption edge energy.

    Returns:
        tuple (tuple): tuple containing:
            edge (int): Closest index to where the edge is.
            bounds (tuple[int, int]): Pre- and post-edge bounds around the edge.
            pre (int): Upper bound on pre-edge.
    """

    edge_idx, pre_idx = find_edges(y)
    edge_points = np.concat(([0], edge_idx, [len(y) - 1]))
    edge_energies = np.array([x[p] for p in edge_points])

    win_view = sliding_window_view(edge_energies, window_shape=2)
    rmean = win_view.mean(axis=1)
    midpoints = [np.where(x <= e)[0][-1] for e in rmean]

    # closest pre-edge point to the absorption edge:
    diffs_pre = [(c, np.abs(x[c] - edge_energy)) for c in pre_idx]
    diffs_pre = sorted(diffs_pre, key=lambda c: c[1])
    pre = diffs_pre[0][0]

    midpoint_arr = np.concat(([0], midpoints, [len(x) - 1]))
    # edge closest to pre-edge indexed:
    diffs_edge = [(c, x[c] - x[pre]) for c in edge_idx]
    # throw away negative differences:
    diffs_edge = [d for d in diffs_edge if d[-1] > 0]
    diffs_edge = sorted(diffs_edge, key=lambda c: c[-1])
    edge = diffs_edge[0][0]

    # two midpoints closest to the edge:
    edge_no = np.where(edge_idx >= edge)[0][0]
    bounds = [midpoint_arr[edge_no + 1], midpoint_arr[edge_no + 2]]
    if bounds[0] < 0:
        bounds[0] = 0
    if bounds[-1] > len(x) - 1:
        bounds[-1] = len(x) - 1

    if np.abs(bounds[0] - pre) <= 3:
        # this is a catch to make the lower bound fall further
        # away from the edge.. should be altered to look at gradient
        # instead of using hard coded number.
        bounds[0] -= 5
    return edge, bounds, pre


def fit_quad_bkg(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[int, int],
    pre_idx: int,
    edge_idx: int,
    include: bool = True,
) -> tuple[float, float, float]:
    """
    Determine jump height based on interpolating data onto quadratic.

    Arguments:
        x (np.ndarray): (1d) x-array (energy).
        y (np.ndarray): (1d) y-array (f2 or lni0it).
        bounds (tuple[int, int]): Span for pre- and post-edge region around edge.
        pre_idx (int): Where to start pre-edge region.
        edge_idx (int): Edge energy indexed value.
        include (bool): Whether to calculate jump or not.

    Returns:
        tuple (tuple): tuple containing:
            yjump (float): Calculated jump height.
            ymax (float): y value at maximum about the edge.
            ypre (float): y value at lower bound about the edge.
    """
    lower = bounds[0]
    upper = bounds[-1]
    # get pre- and post-edge sections for y data:
    pre_y = y[lower:pre_idx]
    post_y = y[edge_idx:upper]
    pre_x = x[lower:pre_idx]

    # where is a maximum reached in y?
    max_idx = np.where(post_y == np.max(post_y))[0][0] + edge_idx
    ymax = y[max_idx]
    ypre = y[pre_idx]

    if not include:
        yjump = 0
    else:
        # where is a minimum reached in y?
        lo_idx = np.where(pre_y == np.min(pre_y))[0][0] + lower
        if lo_idx != lower:
            pre_y = y[lower:lo_idx]
            pre_x = x[lower:lo_idx]
        bkg = interp1d(pre_x, pre_y, kind="quadratic", fill_value="extrapolate")(x)
        yjump = y[max_idx] - bkg[max_idx]

    return yjump, ymax, ypre


def _check_edges(x: np.ndarray, y: np.ndarray, abs_idx: int) -> bool:
    """
    Check if any other elements in a compound will have an absorption
    edge within +/- 50 eV of the absorbing atom.

    Arguments:
        x (np.ndarray): 1d x array (energy).
        y (np.ndarray): y (f2/lni0it). Must have same spacing as `x`.
        abs_idx (int): Index where the energy = edge energy.

    Returns:
        include (bool): Whether to include the element in further edge-step
                         calculations.
    """
    where_edge, where_pre = find_edges(y)
    abs_energy = x[abs_idx]
    diffs = [np.abs(x[e] - abs_energy) for e in where_edge]
    if np.min(diffs) <= 50:
        include = True
    else:
        include = False
    return include
