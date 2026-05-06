import numpy as np
from larch.math import remove_nans2
from scipy.ndimage import gaussian_filter
import logging
from xraylib import EdgeEnergy, SymbolToAtomicNumber
from xraydb import guess_edge

from xas_toolbox.utils.maths.start_stop import _mean_start_stop
from xas_toolbox.utils.xray import _edges

logger = logging.getLogger(__name__)
    
def calc_e0(x:np.ndarray, y:np.ndarray, sigma:float=5)\
                ->tuple[list|float,int|float,int|list]:
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
            edge_coords (int | list[int]): List of indices in x where edges occur.
            edge_number (int | list[int]): Detected number of edges for scan(s).
    """
    
    x, y = remove_nans2(x, y)
    post = 0.05*(x[-1]-x[0])
    y = gaussian_filter(y,sigma)
    if y.ndim == 1:
        dy = np.gradient(y)
    else: dy = np.gradient(y, axis=1)
    dy = dy/np.max(dy)
    yf = np.round(dy, 0)

    if y.ndim == 1:
        dny = np.concat(([0], np.diff(yf)))
        edge_coords = np.where(dny == 1)[0]

        if len(edge_coords) > 1:
            edge_coords = _filter_edge_coords(edge_coords, x, post)
        edge_energy = [x[e] for e in edge_coords]
        edge_number = len(edge_coords)
    else:
        dny = np.diff(yf, axis=1, prepend=0)
        edge_coords = [np.where(dny[i,:] == 1)[0] for i in range(y.shape[0])]
        for i in range(y.shape[0]):
            if len(edge_coords[i]) > 1:
                edge_coords[i] = _filter_edge_coords(edge_coords[i], x, post)
        edge_energy = [x[e] for i in range(y.shape[0]) for e in edge_coords[i]]
        edge_number = [len(e) for e in edge_coords]
    
    return edge_energy, edge_coords, edge_number
    

def edge_from_symbol(x:np.ndarray, y:np.ndarray, symbol:str)->str:
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
    Z = SymbolToAtomicNumber(symbol)
    e0, ec, eno = calc_e0(x, y)
    e0s = np.array([
        EdgeEnergy(Z, e.upper()) for e in _edges
        ])
    diffs = np.abs(e0s - e0)
    edge = _edges[np.where(diffs == diffs.min())[0][0]]
    return edge

    
def compare_e0s(x:np.ndarray, y:np.ndarray, symbol:str|None=None, edge:str|None=None)->tuple[float, float]:
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
    if edge is None:
        if symbol is None: symbol, edge = guess_edge(e0_calc)
        else: edge = edge_from_symbol(x, y, symbol)
    e0_calc = e0_calc[0]
    # compare with xraydb value:
    # e0_pred = xray_edge(symbol, edge, energy_only=True)
    e0_pred = EdgeEnergy(SymbolToAtomicNumber(symbol), _edges.index(edge.upper()))*1e3
    if e0_calc is not None:
        return e0_pred, e0_calc
    else:
        raise ValueError("Not able to find e0 for dataset.")
        
def _filter_edge_coords(coords:np.ndarray|list, energy:np.ndarray, min_spacing:int)->np.ndarray:
    """
    Filter a list of points by a minimum spacing in energy and return average 
    value between each new cluster.
    """
    energies = [energy[i] for i in coords]
    averages = _mean_start_stop(coords, np.diff(energies), min_spacing)
    return averages

