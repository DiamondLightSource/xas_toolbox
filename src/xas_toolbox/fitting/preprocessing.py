"""
Data processing required to be ran fitting is done.
"""

from typing import Any

import numpy as np

from xas_toolbox.io import XasMeasurement


def calc_overlap_score(
    target: np.ndarray[tuple[Any, ...], np.dtype[Any]],
    reference: np.ndarray[tuple[Any, ...], np.dtype[Any]],
) -> float:
    """
    Get the fractional similarity between two arrays.

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
    data: XasMeasurement, components: list[XasMeasurement] | None
) -> tuple[XasMeasurement, list[XasMeasurement] | None]:
    """
    Filter a set of data and components to find a common energy range.
    Data that has less than 30% similarity with the data's energy range will be
    removed. <br>
    If the experimental data spans the reference data it is kept in the set.

    Arguments:
        data: Experimental data.
        components: List of reference datasets.
    Returns:
        tuple: tuple containing:
            data: Same data from arguments.
            components_out: list of filtered components.
    """

    espace_data = np.linspace(data.energy[0], data.energy[-1], 10)
    espace_data = np.array([round(n, -2) for n in espace_data])

    if components is not None:
        flagged_components: list[int] = []
        for i in range(len(components)):
            espace = np.linspace(components[i].energy[0], components[i].energy[-1], 10)
            espace = np.array([round(n, 2) for n in espace])
            score = calc_overlap_score(espace_data, espace)
            if score < 0.3:
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
    data: XasMeasurement | None, components: list[XasMeasurement]
) -> tuple[float, float]:
    """
    Get highest and lowest starting/ending energies for data and \
    list of components.

    Arguments:
        data: Optional experimental data.
        components: Reference data.

    Returns:
        tuple: tuple containing:
            Emin: Largest starting energy across all scans.
            Emax: Lowest ending energy across all scans.

    """
    min_e, max_e = [], []
    if data:
        min_e.append(data.energy[0])
        max_e.append(data.energy[-1])
    for c in components:
        min_e.append(c.energy[0])
        max_e.append(c.energy[-1])

    return np.max(min_e), np.min(max_e)
