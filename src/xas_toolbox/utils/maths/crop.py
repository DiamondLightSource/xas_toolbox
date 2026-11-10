from typing import Literal

import numpy as np

from xas_toolbox.utils.typing import _Array1d, _ArrayNd


def crop_xy(
    x: _Array1d,
    y: _Array1d | _ArrayNd,
    bounds: tuple[int | float, int | float],
    mode: Literal["index", "value"] = "index",
):
    """
    Crop x and y data via bounds either by index of bounds or
    when they reach value in x.

    Arguments:
        x: x data (1d array)
        y: y data(1d/nd array)
        bounds: bounds to crop between (lower, upper)
        mode: whether the bounds specify indices or values in
          x to crop the data by.

    Returns:
        tuple (tuple): tuple containing:
            xcrop: Cropped x-data
            ycrop: Cropped y-data
    """

    if mode == "index":
        lower = int(bounds[0])
        upper = int(bounds[-1])
    else:
        upper = int(np.abs(x - bounds[-1]).argmin())
        lower = int(np.abs(x - bounds[0]).argmin())

    xcrop = x[lower:upper]
    if y.ndim == 1:
        ycrop = y[lower:upper]
    else:
        ycrop = y[:, lower:upper]
    return xcrop, ycrop
