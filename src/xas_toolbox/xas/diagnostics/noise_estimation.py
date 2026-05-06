import numpy as np; from scipy.signal import savgol_filter
from larch.xafs import autobk, set_xafsGroup, estimate_noise
from larch import Group
from xas_toolbox.io import XasMeasurement

def noise_estimate_poly(y:np.ndarray, winlength:int=7, polyorder:int=3)\
    ->np.ndarray|float:
    """
    Estimate noise for stack of data. <br>
    Noise is taken to be the standard deviation of a
    given spectrum - smoothed spectrum (using savitsky-golay
    filter).

    Parameters:
        y (np.ndarray): Absorption (nspectra x number of energy points)
        winlength (int, Optional): Length of savgol filter window.
        polyorder (int, Optional): Polynomial order used in savgol filter.

    Returns:
        noise (np.ndarray|float): Estimated noise.
    """
    if y.ndim > 1:
        n_scans = y.shape[0]; noise = np.zeros(n_scans)

        for i in range(n_scans):
            ymean = y[:i+1,:].mean(axis=0)
            ysmooth = savgol_filter(ymean, winlength, polyorder)
            noise[i] = np.std(ymean-ysmooth)
    else:
        noise = np.std(y - savgol_filter(y, winlength, polyorder))
    return noise

def noise_estimate_fft(x:np.ndarray, y:np.ndarray)\
                ->tuple[np.ndarray|float, np.ndarray|float, Group, Group]:
    """
    Estimate k-space noise and k_max using larch for repetition
    scans.

    Arguments:
        x (np.ndarray): Energy axis.
        y (np.ndarray): Nscans * npoints absorption values.
    
    Returns:
        tuple (tuple): tuple containing:
            k_noise (np.ndarray|float): Array of estimated k-space noise for each scan.
            kmax (np.ndarray|float): Array of estimate kmax for FT/analysis for each scan.
            scan_0 (Group): Larch group of first scan's data with autobk applied.
            group_tmp (Group): Larch group of summed repetition avlues with autobk applied.
    """
    if y.ndim > 1:
        kmax = np.zeros(y.shape[0])
        knoise = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            group_tmp = set_xafsGroup(None)
            group_tmp.energy = x
            group_tmp.mu = y[0:i+1,:].sum(axis=0)
            autobk(group_tmp)
            noise_group_tmp = set_xafsGroup(None)
            estimate_noise(group_tmp.k, group_tmp.chi, group=noise_group_tmp, 
                        kweight=2)
            kmax[i] = noise_group_tmp.kmax_suggest
            knoise[i] = noise_group_tmp.epsilon_k
            if i == 0:
                scan_0 = group_tmp
    else:
        group_tmp = set_xafsGroup(None)
        group_tmp.energy = x
        group_tmp.mu = y
        autobk(group_tmp)
        noise_group_tmp = set_xafsGroup(None)
        estimate_noise(group_tmp.k, group_tmp.chi, group=noise_group_tmp, 
                        kweight=2)
        kmax = noise_group_tmp.kmax_suggest
        knoise = noise_group_tmp.epsilon_k
        scan_0 = group_tmp

    return knoise, kmax, scan_0, group_tmp

def snr_metrics(data:Group|XasMeasurement)\
    ->tuple[np.ndarray|float, np.ndarray|float, np.ndarray|float, Group,Group]:
    """
    Get estimates of noise and kmax for a series of spectra using fitting with a polynomial
    or larch's `estimate_noise` function.

    Arguments:
        data (Group|XasMeasurement): Stack of spectra to analyze, they must share a common energy axis.

    Returns:
        tuple (tuple): tuple containing:
            noise_poly (np.ndarray|float): Estimated noise from subtracting a polynomial from scans.
            knoise (np.ndarray|float): Estimated noise from larch's `estimate_noise()`.
            kmax (np.ndarray|float): Estimated kmax from larch's `estimage_noise()`.
            group_0 (Group): The first scan with `autobk` applied.
            avr_group (Group): Sum of scans with `autobk` applied at end.
    """
    if not hasattr(data, "energy"): raise AttributeError("Data has no attribute energy")
    if not hasattr(data, "mu"): raise AttributeError("Data has no attribute mu")

    if data.energy.ndim > 1: raise NotImplementedError("Cannot produce snr metrics for un-aligned scans.")
    
    noise_poly = noise_estimate_poly(data.mu)

    knoise, kmax, group_0, avr_group = noise_estimate_fft(data.energy, data.mu)

    return noise_poly, knoise, kmax, group_0, avr_group
