import numpy as np

from xas_toolbox.xas.corrections import pre_edge
from xas_toolbox.xas.data_generation import (
    add_gaussian_noise,
    add_poisson_noise,
    make_fake_spectrum,
)


def test_1d_single_edge():
    spectrum = make_fake_spectrum("FeCO3", "Fe", "K", nscans=1)
    norm, flat, step, pre, post = pre_edge(
        spectrum.energy, spectrum.mu, e0=None, pre_order=1, post_order=2, output="Full"
    )
    for tmp in [norm, flat]:
        assert len(tmp) == len(spectrum.mu)
        assert tmp.ndim == spectrum.mu.ndim

    assert isinstance(step, float)
    assert len(pre) == len(spectrum.mu) == len(post)
    assert pre.ndim == spectrum.mu.ndim == post.ndim
    assert step > 0


def test_1d_single_edge_noisy():
    spectrum = make_fake_spectrum("Li3As", "As", "k", nscans=1)
    y_poisson = add_poisson_noise(spectrum.mu, 0.005, max(spectrum.mu) * 0.01)
    y_gauss = add_gaussian_noise(
        spectrum.energy, spectrum.mu, max(spectrum.mu) * 0.005, 0
    )
    y_combined = add_gaussian_noise(
        spectrum.energy, y_poisson, max(spectrum.mu) * 0.005, 0
    )

    for y in [y_poisson, y_gauss, y_combined]:
        norm, flat, step = pre_edge(
            spectrum.energy, y, e0=None, pre_order=1, post_order=2, mode="First"
        )
        assert len(norm) == len(flat) == len(y)
        assert isinstance(step, float)
        assert norm.ndim == flat.ndim == y.ndim
        assert step > 0


def test_1d_multi_edge():
    spectrum = make_fake_spectrum("FeCoCO2", ["Fe", "Co"], "K", nscans=1)
    norm, flat, step = pre_edge(spectrum.energy, spectrum.mu, mode="First")
    assert len(norm) == len(flat) == len(spectrum.mu)
    assert isinstance(step, float)

    norm, flat, step, xc, yc = pre_edge(spectrum.energy, spectrum.mu, mode="Crop")
    assert isinstance(norm, list)
    assert isinstance(flat, list)
    assert isinstance(step, list)
    assert len(norm) == len(flat) == len(step) == 2

    for i in range(len(norm)):
        assert len(norm[i]) == len(flat[i])
        assert step[i] > 0


def test_1d_multi_edge_noisy():
    spectrum = make_fake_spectrum("FeCoCO2", ["Fe", "Co"], "K", nscans=1)
    y_poisson = add_poisson_noise(spectrum.mu, 0.005, max(spectrum.mu) * 0.01)
    y_gauss = add_gaussian_noise(
        spectrum.energy, spectrum.mu, max(spectrum.mu) * 0.005, 0
    )
    y_combined = add_gaussian_noise(
        spectrum.energy, y_poisson, max(spectrum.mu) * 0.005, 0
    )

    for y in [y_poisson, y_gauss, y_combined]:
        norm, flat, step, xc, yc = pre_edge(spectrum.energy, y, mode="Crop")
        assert isinstance(norm, list)
        assert isinstance(flat, list)
        assert isinstance(step, list)
        assert len(norm) == len(flat) == len(step)
        # relaxing edge detection constraints for multi-edge noisy data..
        assert 1 < len(norm) <= 3

        for i in range(len(norm)):
            assert len(norm[i]) == len(flat[i])
            assert step[i] > 0


def test_nd_single_edge():
    spectrum = make_fake_spectrum("CdTe", "Cd", "K", nscans=200)
    norm, flat, step, pre, post = pre_edge(
        spectrum.energy, spectrum.mu, e0=None, pre_order=1, post_order=2, output="Full"
    )

    assert len(step) == spectrum.mu.shape[0]
    assert min(step) > 0
    assert pre.shape == post.shape == spectrum.mu.shape
    assert norm.shape == flat.shape == spectrum.mu.shape


def test_nd_single_edge_noisy():
    spectrum = make_fake_spectrum("TcO2", "Tc", "K", nscans=240)
    y_poisson = add_poisson_noise(spectrum.mu, 0.005, np.max(spectrum.mu) * 0.02)
    y_combined = add_gaussian_noise(
        spectrum.energy, y_poisson, np.max(spectrum.mu) * 0.005, 0
    )
    for y in [spectrum.mu, y_poisson, y_combined]:
        norm, flat, step = pre_edge(spectrum.energy, y)
        assert norm.shape == y.shape == flat.shape
        assert len(step) == y.shape[0]


def test_nd_multi_edge():
    spectrum = make_fake_spectrum("CuZnO", ["Cu", "Zn"], "K", nscans=1000)
    norm, flat, step, pre, post = pre_edge(spectrum.energy, spectrum.mu, output="Full")

    assert norm.shape == flat.shape == spectrum.mu.shape
    assert pre.shape == post.shape == spectrum.mu.shape
