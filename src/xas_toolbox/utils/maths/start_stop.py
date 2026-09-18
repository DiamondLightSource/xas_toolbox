import numpy as np


def _get_start_stop(
    coords: np.ndarray | list, diff: np.ndarray, min_spacing: float
) -> list[tuple]:
    """
    Get list of tuples of coordinates where minimum spacing > min_spacing

    Parameters:
        coords (np.ndarray|list): list of coordinates.
        diff (np.ndarray): list of differences between some array to filter the coords
        based-off.
        min_spacing (float): minimum difference between coordinates.

    Returns:
        ss (list[tuple]): grouped coordinates.
    """
    ss = []
    size = len(diff)  # noqa: E702
    lookup = diff <= min_spacing
    start, stop = 0, 0
    while start < size:
        slice = lookup[start:]
        if np.all(slice):
            ss.append((coords[start], coords[-1] + 1))
            break  # noqa: E701, E702
        stop = np.argmin(slice) + start
        ss.append((coords[start], coords[stop] + 1))
        start = stop + 1  # noqa: E702
        if start == size:
            ss.append((coords[start], coords[start] + 1))  # noqa: E701
    return ss


def _mean_start_stop(
    coords: np.ndarray | list, diff: np.ndarray, min_spacing: float = 50
) -> np.ndarray:
    """
    For a list of points, cluster them together with cluster
    distance dependent on *min_spacing value*.
    For pairs in a cluster return for each the mean of the two values.

    Parameters:
        coords (np.ndarray | list): Coordinates to cluster.
        diff (np.ndarray): Some array of differences to filter coords on.
        min_spacing (int): Minimum allowed distance between points.

    Returns:
        averaged (np.ndarray): The clusters points.
    """
    ss_full = _get_start_stop(coords, diff, min_spacing)
    averaged = np.empty(len(ss_full), dtype=np.int64)
    for i in range(len(ss_full)):
        averaged[i] = int(np.mean(ss_full[i]))  # noqa: E701

    return averaged
