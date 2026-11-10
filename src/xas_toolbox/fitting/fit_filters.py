"""
Data processing required to be ran fitting is done.
"""

import numpy as np

from xas_toolbox.io import XasMeasurement
from xas_toolbox.utils.maths.crop import crop_xy
from xas_toolbox.utils.maths.interpolate import interpolate_with_bounds
from xas_toolbox.utils.typing import _Array1dFloat, _ArrayNdFloat


def calc_overlap_score(
    target: _Array1dFloat,
    reference: _Array1dFloat,
) -> float:
    """
    Get the fractional similarity between two 1D arrays.

    Arguments:
        arr: Target array.
        reference: Reference array.

    Returns:
        score: Overlap of two arrays as a fraction of`target`.

    """
    overlap = np.intersect1d(target, reference)
    score = len(overlap) / len(target)
    return score


def check_energy_overlap(
    data: XasMeasurement | tuple[_Array1dFloat, _ArrayNdFloat],
    components: list[XasMeasurement | tuple[_Array1dFloat, _Array1dFloat]] | None,
    tol: float = 0.3,
):
    """
    Filter a set of data and components to find a common energy range.
    Data that has less than 30% (default) similarity with the data's energy range will
    be removed. <br/>
    If the experimental data spans the reference data it is kept in the set.

    Arguments:
        data: Experimental data.
        components: List of reference datasets.
        tol (Optional): Tolerance for removing components. Higher = lower tolerance.
    Returns:
        tuple: tuple containing:
            data: Same data from arguments.
            components_out: list of filtered components.
    """
    if isinstance(data, tuple):
        x = data[0]
    else:
        x = data.energy
    espace_data = np.linspace(x[0], x[-1], 10)

    if components is not None:
        flagged_components = []
        for i in range(len(components)):
            tmp = components[i]
            if isinstance(tmp, tuple):
                x = tmp[0]
            else:
                x = tmp.energy
            espace = np.linspace(x[0], x[-1], 10)
            score = calc_overlap_score(espace_data, espace)
            if score < tol:
                if espace[0] < espace_data[0] and espace[-1] > espace_data[-1]:
                    continue
                else:
                    flagged_components.append(i)
        components_out = [
            components[i] for i in range(len(components)) if i not in flagged_components
        ]
    else:
        components_out = None

    return data, components_out


def get_energy_ranges(
    data: XasMeasurement | _Array1dFloat | tuple[_Array1dFloat, _ArrayNdFloat] | None,
    components: list[XasMeasurement]
    | list[_Array1dFloat]
    | list[tuple[_Array1dFloat, _ArrayNdFloat]],
) -> tuple[float, float]:
    """
    Get highest and lowest starting/ending energies for data and \
    list of components.

    Arguments:
        data: Optional experimental (spectrum or energy or x,y values).
        components: Reference data (list of spectrum or energy or x,y values).

    Returns:
        tuple: tuple containing:
            Emin: Largest starting energy across all scans.
            Emax: Lowest ending energy across all scans.

    """
    min_e, max_e = [], []
    if data is not None:
        if isinstance(data, np.ndarray):
            min_e.append(data[0])
            max_e.append(data[-1])
        elif isinstance(data, tuple):
            min_e.append(data[0][0])
            max_e.append(data[0][-1])
        else:
            min_e.append(data.energy[0])
            max_e.append(data.energy[-1])

    for c in components:
        if isinstance(c, np.ndarray):
            min_e.append(c[0])
            max_e.append(c[-1])
        elif isinstance(c, tuple):
            min_e.append(c[0][0])
            max_e.append(c[0][-1])
        else:
            min_e.append(c.energy[0])
            max_e.append(c.energy[-1])

    return np.max(min_e), np.min(max_e)


def crop_fit_data(
    data: XasMeasurement | tuple[_Array1dFloat, _ArrayNdFloat] | None,
    components: list[XasMeasurement] | None | list[tuple[_Array1dFloat, _Array1dFloat]],
    bounds: tuple[float, float],
):
    """
    Crop data optional and fit components by given bounds in
    energy.

    Arguments:
        data (Optional): A spectrum or tuple of (x, y) data.
        components (Optional): A list of spectra or tuples of (x, y) data.
        bounds: Tuple of bounds (in x) to crop data by.

    Returns:
        tuple (tuple): tuple containing:
            data_out: None if no data or cropped data as dict of x and y data.
            comps_out: None if no components or cropped components as list of dicts
              of x and y data.
            target_length: Minimum length across the data and components.
    """
    npoints = []
    if data is not None:
        data_out = {}
        if isinstance(data, tuple):
            x = data[0]
            y = data[-1]
        else:
            x = data.energy
            y = data.mu
        xout, yout = crop_xy(x, y, bounds, mode="value")
        data_out = {"x": xout, "y": yout}
        npoints.append(len(xout))
    else:
        data_out = None

    if components is not None:
        comps_out = []
        for c in components:
            if isinstance(c, tuple):
                x = c[0]
                y = c[-1]
            else:
                x = c.energy
                y = c.mu
            xout, yout = crop_xy(x, y, bounds, mode="value")
            comps_out.append({"x": xout, "y": yout})
            npoints.append(len(xout))
    else:
        comps_out = None

    target_length = int(np.min(npoints))
    return data_out, comps_out, target_length


def interpolate_fit_data(
    data: XasMeasurement | tuple[_Array1dFloat, _ArrayNdFloat] | None,
    components: list[XasMeasurement] | list[tuple[_Array1dFloat, _Array1dFloat]],
):
    """
    For optional data to fit and list of components perform the following filtering
    steps:<br/>
    &nbsp;(1) Filter components to remove ones which don't span the same energy as
      the experimental data. <br/>
    &nbsp;(2) Crop *data* and *components* to be in the smallest common energy range
      (so no data is interpolated onto a larger range that it spans). <br/>
    &nbsp;(3) Interpolate *data* and *components* to have an equal number of points
      equal to the lowest resolution spectrum.

    Arguments:
        data (Optional): Data to fit to as XasMeasurement or (x, y) tuple.
        components: List of components as XasMeasurement or (x, y) tuple.

    Returns:
        tuple (tuple): tuple containing:
            dtmp: None (if no data) or dict of filtered x,y fit data.
            ctmp: List of dicts of filtered x, y component data.
    """
    if isinstance(data, tuple):
        Ebounds = get_energy_ranges(data[0], components)  # noqa: N806
    else:
        Ebounds = get_energy_ranges(data, components)  # noqa: N806

    dtmp, ctmp, tl = crop_fit_data(data, components, Ebounds)

    for i in range(len(ctmp)):
        x, y = interpolate_with_bounds(
            x=ctmp[i]["x"], y=ctmp[i]["y"], npoints=tl, bounds=Ebounds
        )
        ctmp[i]["x"] = x
        ctmp[i]["y"] = y
    if data is not None:
        x, y = interpolate_with_bounds(
            x=dtmp["x"], y=dtmp["y"], npoints=tl, bounds=Ebounds
        )
        dtmp["x"] = x
        dtmp["y"] = y

    return dtmp, ctmp
