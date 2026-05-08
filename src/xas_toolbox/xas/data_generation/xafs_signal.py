import numpy as np

_rng = np.random.default_rng(25)

def damped_sine(x:np.ndarray, nscans:int)->np.ndarray:
    """
    Return a damped sine wave where:
    <ol>
     Decay factor is chosen to have a half-life betweeen 1/2 and 1/5th of data length,<br>
     Phase is chosen from normal distribution between [pi/4, pi/7)<br>
     Angle is chosen to have a frequency between 5-20 waveforms in the energy length.
    </ol>
    There is a cut-off such that if energy length is over 100 points,
    it's fixed at 100 for these calculations.
    (this is to avoid very long running signals)

    Arguments:
        x (np.ndarray): X-axis for sine.
        nscans (int): Number of scans to generate sine waves for.

    Returns:
        out (np.ndarray): Normalised, damped sine wave.
    """  # noqa: E501

    npoints = len(x)
    xn = np.empty((nscans, len(x)))
    xn[:] = x

    # may want this to be energy-based.
    if npoints > 100:
        npoints = 100

    out = np.zeros((nscans, len(x)))

    wherestop = _rng.choice(np.arange(2, 5, 0.5), nscans)
    npts = np.empty((nscans)); npts.fill(npoints)  # noqa: E702, UP034
    halflife = npts/wherestop
    decayf =  np.divide(xn, halflife[:,None], where=halflife[:,None] != 0)
    decay = 2**-(decayf)

    phase = _rng.normal(np.pi/4, np.pi/7, nscans)

    freq = _rng.choice(np.linspace(5, 10, 10), nscans)
    omega1 = _rng.normal(2*np.pi, np.pi, nscans)
    omega = omega1*freq/npoints

    x_sine = np.multiply(xn, omega[:,None], where=omega[:,None]!=0)
    phase_out = np.add(x_sine, phase[:,None], where=phase[:,None]!=0)

    out[:] = np.multiply(decay, np.sin(phase_out))

    return out/np.abs(np.min(out))



def make_signal(x:np.ndarray, y:np.ndarray, lower:np.ndarray,
                 upper:np.ndarray, npaths:int=5):
    y2 = y.copy()
    n_edge = len(upper)
    nscans = y.shape[0]

    for i in range(n_edge):
        lo = lower[i]
        hi = upper[i]
        y_tmp = y2[:,lo:hi]
        x_tmp = x[lo:hi]
        signal_sum = np.zeros_like(y_tmp)
        # j_h = y_tmp[0,lo:hi]
        for i in range(npaths):  # noqa: B007
            signal_sum += damped_sine(x_tmp, nscans)
        damping = _rng.choice(np.arange(0.1, 0.5, 0.1), nscans)
        y_tmp += np.multiply(signal_sum/np.abs(np.nanmax(signal_sum)),
                                               damping[:,None], where=damping[:,None]!=0)  # noqa: E501

        y2[:,lo:hi] = y_tmp
        ints = _rng.choice(np.arange(1, 1.2, 0.1), nscans)
        y2[:,lo:lo+5] = np.multiply(np.abs(y2[:,lo:lo+5]), ints[:,None], where=ints[:,None]!=0)  # noqa: E501

    return y2
