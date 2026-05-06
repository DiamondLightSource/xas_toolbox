import numpy as np

round_n = lambda a, n: np.array([round(x, n) for x in a])

def _remove_negative_gradient(x:np.ndarray, y:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """
    Remove points from x,y data if x is decreasing at any point.
    """
    dx = np.diff(x)
    if not any(dx <= 0):
        return x, y
    bad = np.where(dx <= 0)[0]
    np.delete(x, bad, axis=0)
    np.delete(y, bad, axis=0)
    return x, y

def resize_with_replacement(data:list[np.ndarray], placeholder:float|int=np.nan)->np.ndarray:
    """
    For a list of variable-length arrays resize to
    one array of size (maximum length, length of list).
    Values are padded with NaNs to make lengths the same.

    Arguments:
        data (list[np.ndarray]): List of varying-length arrays.
    
    Returns:
        out (np.ndarray): Array of the data, empty values replaced by NaN.
    """
    lmax = np.max([len(d) for d in data])
    out = np.empty((len(data), lmax))
    for i in range(len(data)):
        d = data[i]
        if len(d) != lmax:
            diff = lmax - len(d)
            nans = [placeholder]*diff
            out[i] = np.concat((d, nans))
        else:
            out[i] = d
    return out

def norm_abs_matrix(values:np.ndarray|list)->np.ndarray:
    """
    Shift all values in a matrix such that all values are positive.
    Normalise the matrix over it's 0th axis to ensure all values
    at each point values[i,:] sum to 1.

    Arguments:
        values (np.ndarray|list): Matrix to normalise. (length * values)
    
    Returns:
        normalised (np.ndarray): Normalised, all-positive new matrix.
    """
    values = np.array(values)
    normalised = np.empty_like((values), dtype=float)

    if len(values.shape) == 1:
        lvtmp = np.min(values)
        if lvtmp < 0: lv = np.abs(lvtmp)
        else: lv = 0
        values = [v+lv for v in values]
        normalised = np.array([x/np.sum(values) for x in values])

    else:
        for i in range(values.shape[0]):
            tmp = values[i]
            nvals = np.where(tmp < 0)[0]
            if len(nvals) >=1: lv = np.min(tmp[nvals])
            else: lv = 0
            values[i] = tmp - lv
        for i in range(values.shape[1]):
            normalised[:,i] = np.array([x/np.sum(values[:,i]) for x in values[:,i]])
    return normalised

