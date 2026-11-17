import logging

import numpy as np
from larch.math import remove_nans2
from scipy.ndimage import gaussian_filter
from xraydb import guess_edge
from xraylib import EdgeEnergy, SymbolToAtomicNumber

from xas_toolbox.utils.maths.start_stop import _mean_start_stop
from xas_toolbox.utils.typehints import _Array1dFloat, _ArrayNdFloat
from xas_toolbox.utils.xray import _edges

logger = logging.getLogger(__name__)


def calc_e0(x: np.ndarray, y: np.ndarray, sigma: float = 5):
    """
    Find all values of energy in which a scan has an edge jump.

    Arguments:
        x (np.ndarray): (1d) Array of energy values.
        y (np.ndarray): (1d or nd) Array of absorption values.
        sigma (float, Optional): Smoothing parameter for gaussian filter on data.
                                 It is not recommended to set it lower than 3.
    Returns:
        tuple (tuple): tuple containing:
            edge_energy (float | list[float]): List of x values where edges occur.
            edge_coords (int | np.ndarray[int]): List of indices in x where edges occur.
            edge_number (int | list[int]): Detected number of edges for scan(s).
    """

    x, y = remove_nans2(x, y)
    post = 0.05 * (x[-1] - x[0])

    if y.ndim == 1:
        yf = np.gradient(gaussian_filter(y, sigma))
        yf /= np.max(yf)
    else:
        yf = np.gradient(gaussian_filter(y, sigma), axis=1)
        yf /= np.max(yf, axis=1)[:, None]

    yf = np.round(yf, 0)
    yf = np.diff(yf, prepend=0)
    if y.ndim == 1:
        edge_coords = np.where(yf == 1)[0]
        if len(edge_coords) > 1:
            edge_coords = _filter_edge_coords(edge_coords, x, post)
        edge_energy = [x[e] for e in edge_coords]
        edge_number: int | list[int] = len(edge_coords)
    else:
        edge_coords = []
        for i in range(y.shape[0]):
            tmp = np.where(yf[i, :] == 1)[0]
            ec = _filter_edge_coords(coords=tmp, energy=x, min_spacing=post)
            edge_coords.append(ec)

        edge_energy = []
        edge_number = []

        for i in range(len(edge_coords)):
            edge_number.append(len(edge_coords[i]))
            if len(edge_coords[i]) > 1:
                for e in edge_coords[i]:
                    edge_energy.append(x[e])
            else:
                edge_energy.append(x[edge_coords[i]])
        edge_energy = list(np.array(edge_energy).flatten())

    return edge_energy, edge_coords, edge_number


def get_edge_symbol(
    x: _ArrayNdFloat | _Array1dFloat | list[_Array1dFloat],
    y: _Array1dFloat | _ArrayNdFloat,
    symbol: str | None = None,
):
    """
    Get absorption edge and atom labels for x, y data with optional absorbing atom
    input.

    Arguments:
        x: Energy array/list
        y: Absorption array/list
        symbol (Optional): Absorbing atom.

    Returns:
        out: List/single value for predicted absorption edge + absorbing atom
        (if symbol=`None`).
    """
    if symbol is not None:
        out = edge_from_symbol(x, y, symbol)
    else:
        if isinstance(x, np.ndarray) and x.ndim == 1:
            e0, _, _ = calc_e0(x, y)
            if isinstance(e0, list):
                out = guess_edge(e0[0])
            else:
                out = guess_edge(e0)
        else:
            out = []
            if isinstance(x, np.ndarray):
                for i in range(x.shape[0]):
                    e0, idx, no = calc_e0(x[i, :], y[i, :])
                    if isinstance(e0, list):
                        out.append(guess_edge(e0[0]))
                    else:
                        out.append(guess_edge(e0))
            if isinstance(x, list):
                for i in range(len(x)):
                    e0, idx, no = calc_e0(x[i], y[i])
                    if isinstance(e0, list):
                        out.append(guess_edge(e0[0]))
                    else:
                        out.append(guess_edge(e0))
    return out


def edge_from_symbol(
    x: _Array1dFloat | _ArrayNdFloat | list[_Array1dFloat],
    y: _Array1dFloat | _ArrayNdFloat | list[_Array1dFloat],
    symbol: str,
) -> str | list[str]:
    if isinstance(x, np.ndarray):
        if x.ndim > 1:
            edge = []
            for i in range(x.shape[0]):
                tmp = edge_from_symbol_1d(x[i, :], y[i, :], symbol=symbol)  # type:ignore
                edge.append(tmp)
        else:
            edge = edge_from_symbol_1d(x, np.array(y), symbol)
    else:
        edge = []
        for i in range(len(x)):
            tmp = edge_from_symbol_1d(x[i], y[i], symbol=symbol)
            edge.append(tmp)

    edge = list(set(edge))
    if len(edge) == 1:
        edge = edge[0]
    return edge


def edge_from_symbol_1d(
    x: _Array1dFloat, y: _Array1dFloat | _ArrayNdFloat, symbol: str
) -> str:
    """
    Get absorption edge label for a sample with unkown edge but
    known absorbing atom.

    Args:
        x (np.ndarray): (1d) Array of energy values.
        y (np.ndarray): (1d) Array of absorption values.
        symbol (str): Absorbing atom label.

    Returns:
        edge (str): Absorption edge label.
    """
    Z = SymbolToAtomicNumber(symbol)  # noqa: N806
    e0, ec, eno = calc_e0(x, y)
    e0s = np.array([EdgeEnergy(Z, i) for i in range(5)])
    if isinstance(e0, float):
        diffs = np.abs(e0s - e0)
    else:
        diffs = np.abs(e0s - e0[0])

    edge = _edges[np.where(diffs == diffs.min())[0][0]]
    return edge


def compare_e0s(
    x: _Array1dFloat,
    y: _Array1dFloat | _ArrayNdFloat,
    symbol: str | None = None,
    edge: str | None = None,
):
    """
    Get a tabulated e0 value from either sample metadata for absorbing atom + edge
    or guess these values.

    Arguments:
        x (np.ndarray): (1d) Array of energy values.
        y (np.ndarray): (1d) Array of absorption values.
        symbol (str | None, Optional): Absorbing atom (set to `None` if not known).
        edge (str | None, Optional): Absorbing edge (set to `None` if not known).

    Returns:
        tuple (tuple): Tuple containing:
            e0_pred (float): Tabulated e0 from given/prediced `symbol` and `edge`.
            e0_calc (float): Calcuated e0 from `x` and `y`.
    """
    e0_calc, idx, no = calc_e0(x, y)
    if isinstance(e0_calc, list):
        e0_calc = e0_calc[0]
    if edge is None:
        if symbol is None:
            symbol, edge = guess_edge(e0_calc)
        else:
            edge = edge_from_symbol(x, y, symbol)
    if isinstance(edge, list):
        edge = edge[0]
    # compare with xraydb value:
    # e0_pred = xray_edge(symbol, edge, energy_only=True)
    e0_pred = EdgeEnergy(SymbolToAtomicNumber(symbol), _edges.index(edge.upper())) * 1e3
    if e0_calc is not None:
        return e0_pred, e0_calc
    else:
        raise ValueError("Not able to find e0 for dataset.")


def _filter_edge_coords(
    coords: np.ndarray | list, energy: np.ndarray, min_spacing: int
):
    """
    Filter a list of points by a minimum spacing in energy and return average
    value between each new cluster.

    Returns:
        edge_coords (np.ndarray[int]): Array of filtered edge coordinates.
    """
    if len(coords) == 1:
        return coords
    energies = [energy[i] for i in coords]
    averages = _mean_start_stop(coords, np.diff(energies), min_spacing)
    return np.array(averages, dtype=int)
