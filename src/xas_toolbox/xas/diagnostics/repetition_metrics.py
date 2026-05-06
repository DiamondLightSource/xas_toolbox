import numpy as np
from larch.xafs import autobk, set_xafsGroup
from xas_toolbox.xas.edges import calc_e0
from larch import Group
from xas_toolbox.io import XasMeasurement

from larch.xafs import pre_edge # to replace out later..

def repetition_average(data:XasMeasurement|Group)->Group:
    """
    Create a mean spectrum as larch `Group`.

    Arguments:
        data (XasMeasurement|Group): Stack of XAS data to perform averaging on.

    Returns:
        mean_group (Group): Group with attributes `mu` (averaged), `energy`, `e0`, `e0_avr`, `norm`,
                            `flat`, `norm_avr`, `flat_avr`.
    """

    if not hasattr(data, "energy"): raise AttributeError("data has no attribute energy")
    if not hasattr(data, "mu"): raise AttributeError("data has no attribute mu")
    if data.mu.ndim <= 1: raise ValueError("must have multiple scans.")
    mu_mean = np.mean(data.mu, axis=0)
    mean_group = set_xafsGroup(None)
    mean_group.energy = data.energy; mean_group.mu = mu_mean
    autobk(mean_group)

    e0_calc, *_ = calc_e0(mean_group.energy, mean_group.mu)
    mean_group.e0_avr = e0_calc

    e0_calc, *_ = calc_e0(mean_group.energy, data.mu)
    mean_group.e0 = e0_calc

    flat, norm = np.empty_like(data.mu), np.empty_like(data.mu)
    for i in range(data.mu.shape[0]):
        tmp = set_xafsGroup(None)
        pre_edge(data.energy, data.mu[i,:], group=tmp)
        flat[i,:] = tmp.flat
        norm[i,:] = tmp.norm

    mean_group.flat = flat; mean_group.norm = norm
    mean_group.flat_avr = np.mean(flat, axis=0)
    mean_group.norm_avr = np.mean(norm, axis=0)

    return mean_group


    
