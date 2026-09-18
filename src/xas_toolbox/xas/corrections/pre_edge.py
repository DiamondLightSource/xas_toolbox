import logging
from typing import Literal

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from xas_toolbox.xas.edges import calc_e0, edge_crop

log = logging.getLogger(__name__)


def pre_edge_single_edge(
    x: np.ndarray,
    y: np.ndarray,
    e0_idx: int,
    bounds_lo: tuple[int, int],
    bounds_hi: tuple[int, int],
    pre_order: int,
    post_order: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pre- and post-edge background finding for a single
    e0 value and single array of absorption data (y = 1d).

    Arguments:
        x (np.ndarray): Energy array (1d).
        y (np.ndarray): Absorption array (1d).
        e0_idx (int): Index of where e0 is.
        bounds_lo (tuple[int, int]): Lower and upper bounds on pre-edge\
         region.
        bounds_hi (tuple[int, int]): Lower and upper bounds on post-edge\
         region.
        pre_order (int): Order of pre-edge polynomial.
        post_order (int): Order of post-edge polynomial.

    Returns:
        tuple (tuple[np.ndarray, np.ndarray, float]): tuple containing:
            poly_pre (np.ndarray): Pre-edge background fit (same length as x).
            poly_post (np.ndarray): Post-edge background fit (same length as x).
            edge_step (np.ndarray): Calculated edge jump at e0_idx position.
    """

    poly_pre = Polynomial.fit(
        x[bounds_lo[0] : bounds_lo[-1]], y[bounds_lo[0] : bounds_lo[-1]], deg=pre_order
    )

    poly_post = Polynomial.fit(
        x[bounds_hi[0] : bounds_hi[-1]], y[bounds_hi[0] : bounds_hi[-1]], deg=post_order
    )

    edge_step = poly_post(x[e0_idx]) - poly_pre(x[e0_idx])

    if isinstance(edge_step, np.ndarray):
        edge_step = edge_step[0]

    return poly_pre(x), poly_post(x), edge_step


def pre_edge_bkg(
    x: np.ndarray,
    y: np.ndarray,
    e0: float | int | list[float | int],
    e0_idx: int | list[int],
    pre_order: int,
    post_order: int,
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray[float]]:
    """
    Find pre-edge and post-edge baselines for a scan by\
    fitting a polynomial within bounds.

    Arguments:
        x (np.ndarray): Energy axis (1d).
        y (np.ndarray): Absorption data (1d or nd).
        e0 (float | int | list[float|int]): e0 value(s) in energy.
        e0_idx (int | list[int]): Indices for e0 values.
        pre_order (int): Order for pre-edge polynomial.
        post_order (int): Order for post-edge polynomial.

    Returns:
        tuple (tuple[np.ndarray, np.ndarray, float | np.ndarray[float]]): \
            tuple containing:
            poly_pre (np.ndarray): Pre-edge polynomial.
            poly_post (np.ndarray): Post-edge polynomial.
            edge_step (float | np.ndarray[float]): Estimated edge jump\
                  value(s) (i.e. the difference between `poly_post` and \
                    `poly_pre` at `e0`.).
    """

    if y.ndim == 1:
        e0_round_up = np.where(x >= e0 + 30)[0][0]
        _lo = int(e0_idx / 2)
        pre_hi = np.where(y[_lo:e0_idx] <= np.median(y[_lo:e0_idx]))[0][-1] + _lo
        post_lo = (
            np.where(y[e0_round_up:] >= np.median(y[e0_round_up:]))[0][0] + e0_round_up
        )

        bounds_lo = (0, pre_hi)
        bounds_hi = (post_lo, len(x))
        poly_pre, poly_post, edge_step = pre_edge_single_edge(
            x, y, e0_idx, bounds_lo, bounds_hi, pre_order, post_order
        )

    # non uniform multi-edge stacks of data will need different treatment.
    if y.ndim > 1:
        poly_pre = np.empty_like(y)
        poly_post = np.empty_like(y)
        edge_step = np.empty(y.shape[0])

        ymed = np.median(y, axis=0)
        e0med = np.median(e0)
        e0_idxmed = int(np.median(e0_idx))

        e0_round_up = np.where(x >= e0med + 30)[0][0]
        _lo = int(e0_idxmed / 2)

        pre_hi_med = (
            np.where(ymed[_lo:e0_idxmed] <= np.median(ymed[_lo:e0_idxmed]))[0][-1] + _lo
        )
        post_lo_med = (
            np.where(ymed[e0_round_up:] >= np.median(ymed[e0_round_up:]))[0][0]
            + e0_round_up
        )

        if isinstance(e0, list):
            if len(set(e0)) > 1:
                for i in range(y.shape[0]):
                    # make sure only first edge looked at for stacks.
                    if isinstance(e0_idx[i], np.ndarray):
                        e0_idx_tmp = e0_idx[i][0]
                    else:
                        e0_idx_tmp = e0_idx[i]

                    _low = pre_hi_med + e0_idx_tmp - e0_idxmed
                    _hi = post_lo_med + e0_idx_tmp - e0_idxmed

                    bounds_lo = (0, int(_low))
                    bounds_hi = (int(_hi), len(x) - 1)
                    poly_pre[i, :], poly_post[i, :], edge_step[i] = (
                        pre_edge_single_edge(
                            x,
                            y[i, :],
                            e0_idx[i],
                            bounds_lo,
                            bounds_hi,
                            pre_order,
                            post_order,
                        )
                    )
            else:
                e0 = e0[0]
                bounds_lo = (0, pre_hi_med)
                bounds_hi = (post_lo_med, len(x) - 1)
                for i in range(y.shape[0]):
                    poly_pre[i, :], poly_post[i, :], edge_step[i] = (
                        pre_edge_single_edge(
                            x,
                            y[i, :],
                            e0_idx[i],
                            bounds_lo,
                            bounds_hi,
                            pre_order,
                            post_order,
                        )
                    )

    return poly_pre, poly_post, edge_step


def norm_and_flatten(
    y: np.ndarray,
    pre_bkg: np.ndarray,
    post_bkg: np.ndarray,
    edge_step: float | np.ndarray,
    ie0: int | np.ndarray[int],
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalised and optionally flatten data given pre-
    and post-edge backgrounds.

    Arguments:
        y (np.ndarray): Absorption data (1d or nd).
        pre_bkg (np.ndarray): Pre-edge background (same shape as `y`).
        post_bkg (np.ndarray): Post-edge background (same shape as `y`).
        edge_step (float | np.ndarray): Estimated edge step(s).
        ie0 (int | np.ndarray[int]): Indices for detected e0.
        flatten (bool, Optional): Whether to return flattened data \
            as well as normalised.

    Returns:
        tuple (tuple[np.ndarray, np.ndarray|None]): tuple containing:
            ynorm (np.ndarray): Normalised absorption.
            yflat (np.ndarray | None): Flattened absorption.
    """

    if y.ndim == 1:
        ynorm = np.divide(y - pre_bkg, edge_step)
        _res = (post_bkg - pre_bkg) / edge_step
        if flatten is True:
            yflat = ynorm - _res + _res[ie0]
            yflat[:ie0] = ynorm[:ie0]
        else:
            yflat = None
    else:
        if edge_step.ndim > 1:
            edge_step = edge_step[:, 0]

        if isinstance(ie0, np.ndarray) and ie0.ndim > 1:
            ie0 = np.array(ie0[:, 0], dtype=int)

        ynorm = (y - pre_bkg).T / edge_step

        resid = (post_bkg - pre_bkg).T / edge_step

        if ynorm.shape != y.shape:
            ynorm = ynorm.T

        if flatten is True:
            yflat = ynorm - resid.T
            for i in range(y.shape[0]):
                yflat[i, :] += resid[ie0[i], i]
                yflat[i, : ie0[i]] = ynorm[i, : ie0[i]]

            if yflat.shape != y.shape:
                yflat = yflat.T

        else:
            yflat = None

    return ynorm, yflat


def pre_edge(
    x: np.ndarray,
    y: np.ndarray,
    e0: float | int | list[float | int] | None = None,
    pre_order: int = 1,
    post_order: int = 2,
    output: Literal["Full"] | None = None,
    mode: Literal["First", "Crop"] = "First",
) -> tuple[
    np.ndarray[float] | np.ndarray[np.ndarray[float]] | list[np.ndarray[float]],
    list[np.ndarray[float]] | None,
    list[np.ndarray[float]] | None,
]:
    """
    Perform a pre-edge background subtraction routine on absorption data.

    Arguments:
        x (np.ndarray): Energy axis.
        y (np.ndarray): Absorption data (1d or nd).
        e0 (float | int | list[float | int] | None, Optional): Value(s) (in energy)\
                for e0. Can be left as `None`.
        pre_order (int, Optional): Order for pre-edge polynomial (default = 1).
        post_order (int, Optional): Order for post-edge polynomial (default = 2).
        outout ("Full" | None, Optional): Whether to return just `norm`, `flat` and\
                `edge_step` (`output = None`) or `pre_bkg` and `post_bkg` too\
                      (`output="Full"`).
         mode ("First" | "Crop", Optional): For single scans with muliple edges detected
                whether to use the first value (`mode="First"`) of e0 or apply an edge\
                    -cropping\
                 routine and perform the background subtraction on each split scan\
                      (`mode="Crop"`).

    Returns:
        tuple (tuple[np.ndarray, np.ndarray, float|np.ndarray, None|np.ndarray,\
              None|np.ndarray]):
            tuple containing:
                norm (np.ndarray): Normalised absorption.
                flat (np.ndarray): Flattened absorption.
                edge_step (float | np.ndarray[float]): Estimated edge step value(s).
                pre_bkg (np.ndarray | None): Pre-edge background (if `output="Full"`).
                post_bkg (np.ndarray | None): post-edge baseline (if `output="Full"`).

                x (np.ndarray | None): Cropped energy (if single scan + multiple edges).
                y (np.ndarray | None): Cropped absorption (if single scan + multiple
                    edges).
    """
    if e0 is None:
        e0, e0_idx, _ = calc_e0(x, y)

    else:
        if y.ndim == 1:
            e0_idx = np.where(x >= e0)[0][0]
        if y.ndim > 1:
            e0_idx = [np.where(x >= e0[i])[0][0] for i in range(len(e0))]

    if y.ndim == 1:
        if isinstance(e0, list):
            log.info("Multiple Edges Detected!")
            if mode == "Crop":
                norm, flat, edge_step = [], [], []
                pre_bkg, post_bkg = [], []
                x, y = edge_crop(x, y)
                for i in range(len(x)):
                    e0_idx_tmp = np.where(x[i] >= e0[i])[0][0]

                    _pre_bkg, _post_bkg, _edge_step = pre_edge_bkg(
                        x[i], y[i], e0[i], e0_idx_tmp, pre_order, post_order
                    )
                    _norm, _flat = norm_and_flatten(
                        y[i], _pre_bkg, _post_bkg, _edge_step, e0_idx_tmp
                    )
                    norm.append(_norm)
                    flat.append(_flat)
                    pre_bkg.append(_pre_bkg)
                    post_bkg.append(_post_bkg)
                    edge_step.append(_edge_step)

                log.info("Also returning new x, y")
                if output is not None:
                    return norm, flat, edge_step, pre_bkg, post_bkg, x, y
                else:
                    return norm, flat, edge_step, x, y
            else:
                e0 = e0[0]
                e0_idx = e0_idx[0]
                pre_bkg, post_bkg, edge_step = pre_edge_bkg(
                    x, y, e0, e0_idx, pre_order, post_order
                )
                norm, flat = norm_and_flatten(y, pre_bkg, post_bkg, edge_step, e0_idx)

    elif y.ndim > 1:
        pre_bkg, post_bkg, edge_step = pre_edge_bkg(
            x, y, e0, e0_idx, pre_order, post_order
        )

        norm, flat = norm_and_flatten(y, pre_bkg, post_bkg, edge_step, e0_idx)
    if output is not None:
        return norm, flat, edge_step, pre_bkg, post_bkg

    else:
        return norm, flat, edge_step
