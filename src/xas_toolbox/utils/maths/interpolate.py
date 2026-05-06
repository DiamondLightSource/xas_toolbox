import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from .filter import _remove_negative_gradient

def interpolate_data(x:np.ndarray, y:np.ndarray, xtarget:np.ndarray,
                   method:str="Pchip")->np.ndarray:
    """
    Interpolate y (previously on x-axis) to the target axis xtarget.

    Arguments:
        x (np.ndarray): Original x-data.
        y (np.ndarray): Original y-data.
        xtarget (np.ndarray): New x-data for y to be mapped to.
        method (str, optional): "Pchip" or "linear", interpolation method to use.
    Returns:
        fx (np.ndarray): Interpolated 
    """
    x, y = _remove_negative_gradient(x, y)
    if "pchip" in method.lower():
        if len(y.shape) == 1:
            fx = PchipInterpolator(x, y)(xtarget)
        else: fx = PchipInterpolator(x, y, axis=1)(xtarget)
    elif method.lower() == "linear":
        if len(y.shape) == 1:
            fx = interp1d(x, y, kind="linear", fill_value="extrapolate")(xtarget)
        else:
            fx = interp1d(x,y, kind="linear", axis=1, fill_value="extrapolate")(xtarget)
    else:
        raise NotImplementedError(f"{method} not available (current options: `Pchip`, `linear`)")
    return fx

def interpolate_with_bounds(x:np.ndarray, y:np.ndarray, npoints:int, bounds:tuple,
                            method:str="Pchip"):
    x, y = _remove_negative_gradient(x, y)
    xtarget = np.linspace(*bounds, npoints)
    fx = interpolate_data(x, y, xtarget, method)
    return fx, xtarget

