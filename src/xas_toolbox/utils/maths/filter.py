from typing import Literal

import numpy as np

round_n = lambda a, n: np.array([round(x, n) for x in a])  # noqa: E731


def _remove_negative_gradient(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def resize_with_replacement(
    data: list[np.ndarray], placeholder: float | int = np.nan
) -> np.ndarray:
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
            nans = [placeholder] * diff
            out[i] = np.concat((d, nans))
        else:
            out[i] = d
    return out


def normalise_matrix(
    matrix: np.ndarray | list, axis: Literal[0, -1] = 0, copy: bool = False
) -> np.ndarray:
    """
    Shift all values in a matrix so that minimum value is non-negative. \\
    Elements along `axis` are normalised so that they sum to 1.

    Arguments:
        matrix (np.ndarray): Array to normalise.
        axis (Literal[0, -1], Optional): Axis over which to normalise (`default=0`).
        copy (bool, Optional): Whether to save normalised matrix as a copy\
              (`default=False`).

    Returns:
        out (np.ndarray): New array if `copy=True`.
    """
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if not copy:
        if np.min(matrix) < 0:
            matrix += np.abs(np.min(matrix))
        if axis == 0:
            matrix /= np.sum(matrix, axis=0)[None, :]
        if axis == -1:
            matrix /= np.sum(matrix, axis=-1)[:, None]
        return matrix
    else:
        out = np.copy(matrix)
        if np.min(out) < 0:
            out += np.abs(np.min(out))
        if axis == 0:
            out /= np.sum(out, axis=0)[None, :]
        if axis == -1:
            out /= np.sum(out, axis=-1)[:, None]
        return out


def remove_nans(*arrays: np.ndarray) -> list[np.ndarray]:
    """
    If a NaN or inf. value exists in one of the arrays
    in the set, remove the point from all of them.

    Arguments:
        *arrays (tuple[np.ndarray]): List of arrays.

    Returns:
        arrays (list[np.ndarray]): List of arrays with NaN/inf removed.
    """
    arrays = list(arrays)
    bad_idxs = [np.where(~np.isfinite(a)) for a in arrays]
    for val in bad_idxs:
        if len(val) == 1:
            for i in range(len(arrays)):
                try:
                    arrays[i] = np.delete(arrays[i], val)
                except IndexError:
                    continue

        if len(val) > 1:
            for i in range(len(arrays)):
                for v in val:
                    try:
                        arrays[i] = np.delete(arrays[i], v)
                    except IndexError:
                        continue
    return arrays


def replace_nans(
    a: np.ndarray | list,
    method: Literal["Average", "Replace"] = "Average",
    val: float | int = 5,
) -> np.ndarray:
    """
    Replace a NaN/inf value in an array with either an average of
    values around it or a user-defined value.

    Arguments:
        a (np.ndarray | list[float|int]): Array to act on.
        method ("Average" | "Replace"): Whether to use mean values around
           % of a's length or replace with provided value.
        val (float | int): % of a's length to calculate mean/value to replace NaNs with.

    Returns:
        a (np.ndarray): NaN/inf-filtered array.
    """
    if isinstance(a, list):
        a = np.array(a)
    mask = ~np.isfinite(a)
    if True not in mask:
        return a

    if method == "Replace":
        a[mask] = val
        return a

    wbad = np.array(np.where(mask))
    npts = int(val / 100) * len(a)
    if npts == 0:
        npts = 1

    if a.ndim == 1:
        for idx in wbad:
            lo = int(idx[0] - npts)
            if lo <= 0:
                lo = 0
            hi = int(idx[0] + npts)
            if hi >= len(a) - 1:
                hi = len(a) - 1
            a[idx] = np.nanmean(a[lo:hi], dtype=a.dtype)

    else:
        for idx in wbad.T:
            lo = int(idx[-1] - npts)
            if lo <= 0:
                lo = 0
            hi = int(idx[-1] + npts)
            if hi >= a.shape[1] - 1:
                hi = a.shape[1] - 1
            a[idx[0], idx[-1]] = np.nanmean(a[idx[0], lo:hi], dtype=a.dtype)

    return a
