from xas_toolbox.utils.maths.interpolate import interpolate_data
import numpy as np
from xas_toolbox.io import XasMeasurement
from xas_toolbox.io.measurement_types import (TransMeasurement, FluorMeasurement, RefMeasurement,
                                                       CommonMeasurement)
import logging; logger = logging.getLogger(__name__)

def get_target_energy(x:np.ndarray)->np.ndarray:
    """
    Given nscans * npoints energy axis, find min, max and npoints
    for each scan. <b>This is meant for use with scans that are 
    close in energy-range but just have some small differences.</b>

    Args:
        x (np.ndarray): nd array of energy values for a series of scans.

    Returns:
        x_t (np.ndarray): 1d array of energy spanning min. range shared by all `x` and with 
                          energy resolution matching lowest value in the set `x`.
    """
    startvals = np.nanmin(x, axis=1); endvals = np.nanmax(x, axis=1)
    e_start = np.max(startvals); e_stop = np.min(endvals)
    
    if e_stop <= e_start: raise ValueError("Scans too dissimilar to correct energy axis.")

    Edensity = 0
    for i in range(x.shape[0]):
        masked = np.ma.masked_invalid(x[i,:]).compressed()

        density = (masked[-1]-masked[0])/len(masked)
        if density >= Edensity: Edensity = density

    npts = int((e_stop-e_start)/Edensity)
    x_t = np.linspace(e_start, e_stop, npts)

    return x_t

def interp_stack(xval:np.ndarray, yval:np.ndarray|None,x_t:np.ndarray)\
    ->np.ndarray|None:
    """
    Interpolate a stack of x and y data onto target axis (assumes that
    fill value for differently sized elements of x and y is `NaN`).

    Arguments:
        xval (np.ndarray): (nd) Array of x (energy) values.
        yval (np.ndarray): (nd) Array of y (absorption) values.
        x_t (np.ndarray): 1d x-array to interpolate the data onto.

    Returns:
        y_out (np.ndarray|None): The interpolated y data.
    """
    if yval is None: return

    y_out = np.empty((yval.shape[0], len(x_t)))

    for i in range(yval.shape[0]):
        xtmp = np.ma.masked_invalid(xval[i,:]).compressed()
        ytmp = np.ma.masked_invalid(yval[i,:]).compressed()
        interp_tmp = interpolate_data(xtmp, ytmp, x_t)
        y_out[i,:] = interp_tmp
    return y_out

def align_stack(scan:XasMeasurement)->XasMeasurement:
    """
    Find common energy axis for stacked data and interpolate `mu`
    such that new datapoints are not introduced into the data.

    Arguments:
        scan (XasMeasurement): Stacked scan object.

    Returns:
        scan_out (XasMeasurement): Copy of input scan with properties `energy`
                                    and `mu` having the stack routine applied to 
                                    them (all other properties/attrs will remain the 
                                    same).

    Note/Example:
        If wanting to apply this routine on a different dataset within an `XasMeasurement`:
        ```
        stack = read_data(paths)
        stack = align_stack(stack)
        # e.g. align transmission data for fluorescence data:
        mutrans = stack.transData.mutrans
        #this works because original energy is kept the same:
        mutrans_interp = interp_stack(data.auxData.energy, mutrans, data.energy)
        # replace the old value with the new one:
        stack.transData.mutrans = mutrans_interp
        ```
    """
    if scan.energy.ndim == 1:
        logger.warning("Single energy axis cannot be stacked")
        return scan
    energy_out = get_target_energy(scan.energy)
    mu_out = interp_stack(scan.energy, scan.mu, energy_out)
    scan_out = XasMeasurement(get_value=scan._get_value)
    scan_out.energy = energy_out; scan_out.mu = mu_out
    return scan_out
