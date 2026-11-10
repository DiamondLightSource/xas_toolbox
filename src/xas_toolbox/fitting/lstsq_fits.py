import logging
from typing import Any

import numpy as np
from scipy.optimize import nnls

from xas_toolbox.io import XasMeasurement
from xas_toolbox.utils.typing import _Array1dFloat, _ArrayNdFloat, _ArrayNdInt

from .fit_filters import check_energy_overlap, interpolate_fit_data

log = logging.getLogger(__name__)


def nnls_fit(
    data: XasMeasurement | tuple[_Array1dFloat, _ArrayNdFloat],
    components: list[XasMeasurement | tuple[_Array1dFloat, _Array1dFloat]],
    component_names: list[str] | None = None,
):
    """
    Perform non-negative least squares fit of components to data.
    """
    data_tmp, comps_tmp = check_energy_overlap(data, components)
    data_tmp, comps_tmp = interpolate_fit_data(data_tmp, comps_tmp)
    ncomp = len(comps_tmp)
    x = data_tmp["x"]
    y = data_tmp["y"]
    references = np.empty((len(x), ncomp))
    if component_names is None:
        component_names = [f"component {i + 1}" for i in range(ncomp)]

    for i in range(ncomp):
        references[:, i] = comps_tmp[i]["y"]
    if y.ndim == 1:
        concs: np.ndarray = nnls(references, y)[0]
        concs /= np.sum(concs)
        solution = {component_names[i]: concs[i] for i in range(ncomp)}

    else:
        solution = {}
        for i in range(y.shape[0]):
            soln: np.ndarray = nnls(references, y[i, :])[0]
            soln /= np.sum(soln)
            solution[f"scan {i + 1}"] = {
                component_names[i]: soln[i] for i in range(ncomp)
            }
        concs = group_nnls_solution(solution)
        solution, concs = filter_nnls_solution(component_names, concs, y, x, references)
    return solution, concs


def group_nnls_solution(solution: dict):
    ntime = len(solution.keys())
    ncomp = len(solution[list(solution.keys())[0]].keys())
    grouped = np.empty((ntime, ncomp))
    i = 0
    for _, v in solution.items():
        j = 0
        for _, v2 in v.items():
            grouped[i, j] = v2
            j += 1
        i += 1
    return grouped


def filter_nnls_solution(
    component_names: list[str],
    concentrations: _ArrayNdFloat | _ArrayNdInt,
    ydata: _ArrayNdFloat,
    xdata: _Array1dFloat,
    refdata: _ArrayNdFloat,
):
    """
    Filter nnls solution and make it look a bit more human-readable.
    """
    solution: dict[str, Any] = {"fit data": {"y": ydata, "x": xdata}}
    concentrations = concentrations.T
    concs_new, names_out = [], []
    for i in range(concentrations.shape[0]):
        if (len(list(set(concentrations[i, :]))) == 1) and (
            list(set(concentrations[i, :]))[0] == 0
        ):
            log.info(f"Removing {component_names[i]} from fit (does not contribute).")
        else:
            concs_new.append(concentrations[i, :])
            names_out.append(component_names[i])
    concs_new = np.array(concs_new).T
    for i in range(concentrations.shape[1]):
        yfit = np.sum(concentrations[:, i] * refdata, axis=1)
        solution[f"time {i}"] = {
            "fitted values": {
                names_out[n]: concs_new[i][n] for n in range(len(names_out))
            },
            "fitted_data": yfit,
        }
    return solution, concs_new
