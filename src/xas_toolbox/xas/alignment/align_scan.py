import numpy as np
from xas_toolbox.xas.edges import compare_e0s, calc_e0
from xas_toolbox.io import XasMeasurement

import logging; logger = logging.getLogger(__name__)

def calc_energy_shift(x:np.ndarray, y:np.ndarray, symbol:str|None=None, edge:str|None=None)\
    ->tuple[np.ndarray, float|np.ndarray]:
    """
    Apply a linear shift to energy axis based upon the difference between the calculated
    and tabulated values for <i>e0</i> in the reference spectrum.

    Arguments:
        x (np.ndarray): (1d or nd) Array of energy values.
        y (np.ndarray): (1d or nd) Array of reference absorption values.
        symbol (str|None, Optional): Absorbing atom for reference spectrum.
        edge (str|None, Optional): Absorbing edge for reference spectrum.

    Returns:
        x_out (np.ndarray): Shifted energy values.
        delta_e (float|np.ndarray): Value(s) for the linear energy shifts.
    """
    if x.ndim == 1:
        if y.ndim == 1:
            e0_pred, e0_calc = compare_e0s(x, y, symbol, edge)
            delta_e = e0_pred - e0_calc
            if delta_e == 0: logger.info("No shift from predicted e0 value.")
            x_out = x + delta_e
        else: raise NotImplementedError("Unable to align stacks of reference absorption data.")
    else:
        x_out = x.copy()
        delta_e = np.empty((y.shape[0]))
        e0_pred, e0_calc = compare_e0s(x[0,:], y[0,:], symbol, edge)
        delta_e0 = e0_pred - e0_calc
        delta_e[0] = delta_e0

        for i in range(y.shape[0]):
            e0_tmp, idx_tmp, eno = calc_e0(x[i,:], y[i,:])
            e0_tmp = e0_tmp[0]
            delta_ei = e0_pred - e0_tmp
            x_out[i,:] += delta_ei
            delta_e[i] = delta_ei
    return x_out, delta_e


def align_scan(scan:XasMeasurement)->XasMeasurement:
    """
    Align the energy axis for a single scan such that the edge
    jump found in the reference spectrum matches the tabulated 
    value given provided edge and absorber data.

    Arguments:
        scan (XasMeasurement): Single scan object, it must have attribute 
                                `refData.murefer`.

    Returns:
        new_scan (XasMeasurement): Copy of the scan with new shifted energy in 
                                    `auxdata.energy` and `energy` properties. Shift
                                    is given in `meta.energy_shift`.
    """
    symbol, edge = None, None

    x = scan.energy
    if hasattr(scan, "refData"):
        y = scan.refData.murefer
    else:
        raise AttributeError("Scan has no reference data.")
    if hasattr(scan, "meta"):
        symbol = scan.meta.ref_symbol
        if symbol is None:
            logger.info("No ref_symbol used, trying symbol.")
            symbol = scan.meta.symbol
        edge = scan.meta.ref_edge
        if edge is None:
            logger.info("No ref_edge, trying edge.")
            edge = scan.meta.edge
    else: raise AttributeError("Scan has no meta-data.")

    if symbol is None: logger.info("No absorbing atom in metadata.")
    if edge is None: logger.info("No absorbing edge in meta-data.")

    x_out, delta_x = calc_energy_shift(x, y, symbol, edge)

    scan_out = XasMeasurement(get_value=scan._get_value)
    scan_out.energy = x_out; scan_out.auxData.energy = x_out
    scan_out.meta.energy_shift = delta_x
    return scan_out
