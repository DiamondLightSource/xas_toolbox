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


def unit_pos_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Shift all values in a matrix so that the minimum value (if negative)
    becomes 0. Along axis 1 all elements are normalised to give a unit norm.

    Arguments:
        matrix (np.ndarray): Matrix of values.

    Returns:
        matrix (np.ndarray): Positive matrix with unit norm along axis=1.

    Note:
        - If used for mixing profile assumed shape is
        timesteps/nscans * number components.
    """
    if np.min(matrix) > 0:
        matrix += np.abs(np.min(matrix))
    matrix = np.divide(matrix.T, np.linalg.norm(matrix, ord=1, axis=1)).T

    return matrix
