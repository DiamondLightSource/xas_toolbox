import numpy as np
from numpy.linalg import lstsq


def _straight_line_coeffs(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Find straight line coefficients fitting the line spanned by `x` and `y` via:
    <i>y = a*x + c</i>.

    Arguments:
        x (np.ndarray): x-axis.
        y (np.ndarray): y-axis.

    Returns:
        tuple (tuple): tuple containing:
            a (float): Gradient term.
            b (float): y-intercept term.
    """
    if y.ndim == 1:
        a = (y[-1] - y[0]) / (x[-1] - x[0])
        b = -(y[-1] * x[0] - y[0] * x[-1]) / (x[-1] - x[0])
    else:
        a = (y[:, -1] - y[:, 0]) / (x[-1] - x[0])
        b = -(y[:, -1] * x[0] - y[:, 0] * x[-1]) / (x[-1] - x[0])
    return a, b


def median_poly_fit(
    y: np.ndarray, ymed: np.ndarray, weight: float | np.ndarray
) -> np.ndarray:
    """
    Fit the difference between data and median-filtered data via least-squares
    to third order.

    Arguments:
        y (np.ndarray): Data to fit.
        ymed (np.ndarray): Data to subtract from y to fit (typically the median of\
              the data).

    Returns:
        fit (np.ndarray): fitted data.
    """
    tofit = y - ymed
    fitted = poly_fit(tofit, weight)
    return ymed + fitted


def poly_fit(y, weight: np.ndarray | float | None = None) -> np.ndarray:
    """
    Fit data via least-squares to third order and return fit result.
    If weight not provided it's not applied.

    Arguments:
        y (np.ndarray): Data to fit.
        weight (np.ndarray|float, optional): Weighting to apply to the data,
                        could be a window or scalar factor.

    Returns:
        out (np.ndarray): Fit to the data.
    """
    # add in an "order" option!!?
    fit_data = np.ones([3, y.shape[0]], dtype=y.dtype)
    fit_data[1, :] = np.arange(y.shape[0])
    fit_data[2, :] = np.arange(y.shape[0]) ** 2

    if weight is not None:
        fit_weights = fit_data * np.sqrt(weight)
        to_fit = y * np.sqrt(weight)
    else:
        fit_weights = fit_data
        to_fit = y

    fitted = lstsq(fit_weights.T, to_fit.T)[0]
    return fit_weights.T @ fitted
