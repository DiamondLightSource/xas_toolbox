import numpy as np

from xas_toolbox.xas.edges import calc_e0

_rng = np.random.default_rng(13)


def _get_bounds(x: np.ndarray, e0: float | list) -> list[tuple[int, int]]:
    """
    Get approximate upper and lower bounds for adding noise.
    """
    if isinstance(e0, float):
        e0 = [e0]

    bounds = []
    lo = 0
    bounds = []

    for e0_tmp in e0:
        lower = np.where(x > e0_tmp - 10)[0][0]
        upper = np.where(x > e0_tmp + 10)[0][0]
        bounds_tmp = (lo, lower)
        lo = upper
        bounds.append(bounds_tmp)
    bounds_tmp = (lo, len(x))
    bounds.append(bounds_tmp)

    return bounds


def add_gaussian_noise(
    x: np.ndarray,
    y: np.ndarray,
    stdev: float,
    mean: float = 0,
    e0: float | list | np.ndarray = None,
) -> np.ndarray:
    """
    Add random Gaussian noise to xas data. <br>
    `x` needs to be provided along with `y` so that noise is not added
    to edge-jump regions.

    Arguments:
        x (np.ndarray): Energy array for data (1d).
        y (np.ndarray): Absorption data (can be N-d or 1d).
        stdev (float): Width of noise distribution.
        mean (float, Optional): Mean of noise distribution, default is 0.

    Returns:
        out (np.ndarray): Copy of `y` with normally distributed noise added to it.
    """
    if e0 is None:
        if y.ndim > 1:
            # if a "stack" is used, assume all edge positions are close to first scan.
            e0, e_coords, e_num = calc_e0(x, y[0])
            # could also take average e0 values!
        else:
            e0, e_coords, e_num = calc_e0(x, y)

    out = y.copy()
    bounds = _get_bounds(x, e0)

    for _range in bounds:
        if y.ndim == 1:
            out[_range[0] : _range[-1]] += _rng.normal(
                mean, stdev, _range[-1] - _range[0]
            )
        else:
            out[:, _range[0] : _range[-1]] += _rng.normal(
                mean, stdev, (y.shape[0], _range[-1] - _range[0])
            )
    return out


def add_poisson_noise(y: np.ndarray, freq: float, amp: float) -> np.ndarray:
    """
    Add Poisson noise to data.

    Arguments:
        y (np.ndarray): Array to add noise to.
        freq (float): Probability of a point having noise.
        amp (float): Mean amplitude of a noise event.

    Returns:
        out (np.ndarray): Copy of `y` with Poisson noise added.
    """
    out = y.copy()
    if y.ndim == 1:
        out += _rng.normal(0, amp, len(y)) * _rng.poisson(freq, len(y))
    else:
        out += np.multiply(_rng.normal(0, amp, y.shape), _rng.poisson(freq, y.shape))
    return out
