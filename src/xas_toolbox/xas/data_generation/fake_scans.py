import logging
from operator import itemgetter

import numpy as np
from xraydb import f2_chantler
from xraydb.xray import xray_edge
from xraylib import AtomicNumberToSymbol, CompoundParser

from xas_toolbox.xas.edges import find_edges

from .xafs_signal import make_signal

logger = logging.getLogger(__name__)


def xray_edge_data(elements: str | list[str], edge: str | list[str]) -> dict:
    """
    From a list or single elements and one or more edges return a dictionary
    of information on e0 value, jump-ratio and fluorescence-yield ordered by e0 value.
    <br>
    <i> If a list of elements is used and different edges are wanted, the length
    of the edge list must be equal to the length of the elements list. </i>

    Arguments:
        elements (str|list[str]): Single element (e.g. "Fe") or list of elements (e.g. ["Co", "Mn"]).
        edge (str|list[str]): Single edge (e.g. "k") or list of edges (e.g. ["L1","l2"]).

    Returns:
        xray_data (dict): Dictionary indexed by elements (or <i>element_edge</i> for single-atom
                          multi-edge data) with keys <i>"e0", "jump-ratio", "fyield", "edge"</i>.

    """  # noqa: E501

    xray_data = {}

    # multi-element data:
    if isinstance(elements, list):
        for i in range(len(elements)):
            if isinstance(edge, list):
                edge_tmp = edge[i]
            else:
                edge_tmp = edge
            e0, fyield, jr = xray_edge(elements[i], edge_tmp)
            xray_data[elements[i]] = [e0, jr, fyield, edge]

    # single-element data:
    else:
        if isinstance(edge, list):
            for edge_tmp in edge:
                e0, fyield, jr = xray_edge(elements, edge_tmp)
                xray_data[f"{elements}_{edge_tmp}"] = [e0, jr, fyield, edge]
        else:
            e0, fyield, jr = xray_edge(elements, edge)
            xray_data[elements] = [e0, jr, fyield, edge]

    sorted_xray = {}
    keys = ["e0", "jump-ratio", "fyield", "edge"]
    sorted_tmp = sorted(xray_data.items(), key=itemgetter(1))
    for k, v in sorted_tmp:
        sorted_xray[k] = {keys[i]: v[i] for i in range(len(v))}

    return sorted_xray


#### energy:


def get_full_energy_range(
    xray_data: dict, npoints: int = 1000, pre: float = 200, post: float = 800
) -> np.ndarray:
    """
    Get a uniform energy range spanning over all edges provided in xray_data dictionary. <br>
    The axis will be padded by -pre and +post to give extra spacing at the start and end of the fake-scan.

    Arguments:
        xray_data (dict): Dictionary of absorbing atom data.
        npoints (int, optional): Number of points for the scan.
        pre (float, optional): Value in <i>eV</i> for the scan to start, relative to the first edge.
        post (float, optional): Value in <i>eV</i> for the scan to end, relative to the last edge position.

    Returns:
        energy_axis (np.ndarray): Uniform energy array.
    """  # noqa: E501
    energies = [xray_data[k]["e0"] for k in xray_data.keys()]
    start = energies[0] - pre
    stop = energies[-1] + post
    return _get_padded_x(start=start, stop=stop, npoints=npoints)


def _get_padded_x(start: float, stop: float, npoints: int = 1000) -> np.ndarray:
    """
    Get a uniform array starting at <i>start</i> ending at <i>stop</i> with
    <i>npoints</i> data-points.

    Arguments:
        npoints (int): Number of data-points.
        start (float): Value to start array.
        stop (float): Value to stop array.

    Returns:
        out (np.ndarray): Array spanning <i>start, stop</i> with <i>npoints</i> data-points.
    """  # noqa: E501
    return np.linspace(start, stop, npoints)


#### f2:


def get_f2(formula: str | list[str], energy: np.ndarray) -> np.ndarray:
    """
    Get the imaginary part of xray atomic form factor given
    a formula and energy range.

    Arguments:
        formula (str|list[str]): Chemical formula/list of species.
        energy (np.ndarray): Energy array.

    Returns:
        f2_abs (np.ndarray): Imaginary part of atomic form factor.
    """
    f2_abs = np.zeros_like(energy)

    if isinstance(formula, str):
        compound = CompoundParser(formula)
        for i in range(compound["nElements"]):
            elem = AtomicNumberToSymbol(compound["Elements"][i])
            count = compound["nAtoms"][i]
            f2_abs += f2_chantler(elem, energy) * count

    elif isinstance(formula, list):
        f2_abs = np.zeros_like(energy)
        for elem in formula:
            f2_abs += f2_chantler(elem, energy)

    return f2_abs


### fake signal:


def get_fake_xas(
    formula: str | list[str],
    absorber: str | list[str] = None,
    edge: str | list[str] = None,
    energy_range: tuple[float, float] | None = None,
    npoints: int = 1000,
    pre: int = 200,
    post: int = 800,
    exafs: bool = True,
    nscans: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate unique fake xas-like signals for a given formula and edge(s)/
    energy range.
    If `absorber` and `edge` are provided, then `energy_range` is not needed/
     if `energy_range` provided, `absorber` and `edge` will be ignored.

    Arguments:
        formula (str|list[str]): Formula/list of elements for fake scan.
        absorber (str|list[str], Optional): Absorbing atom(s) in scan.
        edge (str|list[str], Optional): Absorbing edge(s) to cover in energy range.
        energy_range (tuple[float, float], Optional): Energy range to span.
        npoints (int, Optional): Number of datapoints per scan. 1000 by default.
        pre (int, Optional): Energy (eV) before first absorption edge to start the scan,
                    `200` eV by default.
        post (int, Optional): Energy (eV) after last absorption edge to end the scan,
                    `800` eV by default.
        xafs (bool, Optional): Whether to include exafs-like signal (`True` by default).
        nscans (int, Optional): Number of scans to generate (`10` by default).

    Returns:
        tuple (tuple): tuple containing:
            energy (np.ndarray): (1d) Energy array.
            xafs (np.ndarray): (nscans * npoints) Array of fake spectra.
    """

    if absorber is None and edge is not None:
        raise ValueError("Provide absorbing atom.")
    if edge is None and absorber is not None:
        raise ValueError("Provide absorbing edge.")

    if energy_range is not None:
        if absorber is not None or edge is not None:
            logger.warning("energy_range being used instead of absorber/edge provided.")

    # make energy
    if energy_range:
        energy = _get_padded_x(energy_range[0], energy_range[-1], npoints)

    elif absorber is not None and edge is not None:
        sorted_d = xray_edge_data(absorber, edge)
        energy = get_full_energy_range(sorted_d, npoints, pre, post)

    xafs = np.empty((nscans, npoints))
    # make f2
    f2_abs = get_f2(formula, energy)

    # make nscan*npoint grid of f2
    xafs[:] = f2_abs

    if not exafs:
        return energy, xafs

    # get values to add xafs between
    e0s, pre_edges = find_edges(f2_abs)
    pre_edges = np.delete(pre_edges, 0)
    pre_edges = np.append(pre_edges, npoints - 1)

    xafs = make_signal(energy, xafs, e0s, pre_edges)
    if nscans == 1:
        xafs = xafs[0]
    return energy, xafs
