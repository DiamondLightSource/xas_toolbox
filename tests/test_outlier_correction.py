import numpy as np

from xas_toolbox.xas.corrections import correct_outliers
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

    outs1 = normal_outlier_detection(y, filter_size=int(len(y_g) / 20), tol=2, order=2)

    outs2 = gesd_outlier_detection(y, filter_size=20)

    y_corr1 = correct_outliers(y, outs1)
    y_corr2 = correct_outliers(y, outs2)

    assert y_corr1.ndim == y.ndim
    assert len(y_corr1) == len(y) == len(y_corr2) > 0


def test_nd():
    spectrum = make_fake_spectrum("Fe3C", "Fe", "K", nscans=30)
    x, y0 = spectrum.energy, spectrum.mu
    y_g = add_gaussian_noise(x, y0, 0.002 * np.max(y0), 0)
    y = add_poisson_noise(y_g, 0.01, 0.05 * np.max(y_g))

    outs = normal_outlier_detection(
        y, filter_size=int(y_g.shape[1] / 20), tol=2, order=2
    )

    y_corr = correct_outliers(y, outs)
    assert y_corr.shape == y.shape

    outs2 = remove_edge_jump(x, y, outs)
    y_corr2 = correct_outliers(y, outs2)
    assert y_corr2.shape == y.shape
