import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from xas_toolbox.xas.edges.find_e0 import calc_e0
from xas_toolbox.xas.edges.find_edges import get_edge_bounds

logger = logging.getLogger(__name__)


def edge_crop(
    x: np.ndarray, y: np.ndarray
) -> None | tuple[list | np.ndarray, list | np.ndarray]:
    """
    Given a scan with multiple edges detected, split up into separate ones
    for later pre-processing. If only one edge is detected the `x` and `y`
    data is returned.

    Arguments:
        x (np.ndarray): Energy axis.
        y (np.ndarray): Absorption data (nd- or 1d).

    Returns:
        tuple (tuple): Tuple containing:
            x_out (list[np.ndarray]): List of split energies.
            y_out (list[np.ndarray]): List of split absorption data.
    """
    e0s, e_idxs, e_nums = calc_e0(x, y)

    x_out, y_out = [], []
    if y.ndim == 1:
        if e_nums == 1:
            logger.warning("Only one edge detected")
            return x, y

        for energy in sorted(e0s):
            idx, bounds, pre = get_edge_bounds(x, gaussian_filter(y, sigma=5), energy)
            x_out.append(x[bounds[0] : bounds[-1]])
            y_out.append(y[bounds[0] : bounds[-1]])
    else:
        if list(set(e_nums)) == [1]:
            logger.warning("Only one edge detected")
            return x, y

        e0s, e_idxs, e_nums = calc_e0(x, np.median(y, axis=0))
        for energy in sorted(e0s):
            idx, bounds, pre = get_edge_bounds(
                x, gaussian_filter(np.median(y, axis=0), sigma=5), energy
            )
            x_out.append(x[bounds[0] : bounds[-1]])
            y_out.append(y[:, bounds[0] : bounds[-1]])

    return x_out, y_out
