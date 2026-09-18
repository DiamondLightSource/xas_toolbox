import numpy as np

from xas_toolbox.xas.data_generation import (
    add_gaussian_noise,
    add_poisson_noise,
    make_fake_spectrum,
)
from xas_toolbox.xas.diagnostics import (
    gesd_outlier_detection,
    normal_outlier_detection,
    remove_edge_jump,
)


def test_1d():
    spectrum = make_fake_spectrum("Cu3Sn", "Cu", "K", nscans=1)
    x, y0 = spectrum.energy, spectrum.mu
    y_g = add_gaussian_noise(x, y0, 0.002 * np.max(y0), 0)
    y = add_poisson_noise(y_g, 0.01, 0.05 * np.max(y_g))

    outs = normal_outlier_detection(y_g, filter_size=int(len(y_g) / 20), tol=2, order=2)
    assert isinstance(outs, np.ndarray)
    assert outs.dtype == int
    assert len(outs) > 0

    outs = gesd_outlier_detection(y, filter_size=20)

    assert isinstance(outs, np.ndarray)
    assert outs.dtype == int
    assert len(outs) > 0

    outs = remove_edge_jump(x, y, outs)
    assert len(outs) > 0
    assert isinstance(outs, np.ndarray)

    # fake data with no noise should have no outliers.
    outs = normal_outlier_detection(y0, filter_size=int(len(y_g) / 20), tol=2, order=2)
    outs = remove_edge_jump(x, y, outs)
    assert len(outs) == 0


def test_nd():
    spectrum = make_fake_spectrum("Fe3C", "Fe", "K", nscans=30)
    x, y0 = spectrum.energy, spectrum.mu
    y_g = add_gaussian_noise(x, y0, 0.002 * np.max(y0), 0)
    y = add_poisson_noise(y_g, 0.01, 0.05 * np.max(y_g))

    outs = normal_outlier_detection(
        y_g, filter_size=int(y_g.shape[1] / 20), tol=2, order=2
    )
    assert isinstance(outs, list)
    assert len(outs) > 0

    for o in outs:
        assert isinstance(o, np.ndarray)
        assert o.dtype == int

    outs = remove_edge_jump(x, y, outs)
    assert isinstance(outs, list)
    assert len(outs) > 0
    assert isinstance(outs[0], np.ndarray) and outs[0].dtype == int

    outs = gesd_outlier_detection(y_g, filter_size=20)
    assert isinstance(outs, list)
    assert len(outs) > 0
    assert outs[0].dtype == int
    assert isinstance(outs[0], np.ndarray)

    outs = normal_outlier_detection(
        y0, filter_size=int(y_g.shape[1] / 20), tol=2, order=2
    )
    outs = remove_edge_jump(x, y0, outs)
    for o in outs:
        assert isinstance(o, np.ndarray)
        assert len(o) == 0
