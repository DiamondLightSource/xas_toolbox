from xas_toolbox.xas.data_generation import (
    add_gaussian_noise,
    add_poisson_noise,
    make_fake_spectrum,
)
from xas_toolbox.xas.edges import calc_e0


def test_detect_one_edge():
    spectrum_1 = make_fake_spectrum(
        formula="Fe", absorber="Fe", edge="K", exafs=True, post=800, pre=30, npoints=800
    )
    spectrum_2 = make_fake_spectrum(
        formula="Se",
        absorber="Se",
        edge="K",
        npoints=1000,
        pre=30,
        post=900,
        exafs=True,
        nscans=1,
    )
    spectrum_3 = make_fake_spectrum(
        formula="CuO2",
        absorber="Cu",
        edge="K",
        post=600,
        pre=40,
        npoints=1000,
        exafs=True,
        nscans=50,
    )
    spectrum_4 = make_fake_spectrum(
        formula="FeCO2",
        absorber="Fe",
        edge="K",
        post=1000,
        pre=30,
        npoints=1200,
        exafs=True,
        nscans=1,
    )
    spectra = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    for s in spectra:
        e0, e0_idx, num_e0 = calc_e0(s.energy, s.mu)
        if isinstance(num_e0, int):
            assert num_e0 == 1
        else:
            assert list(set(num_e0)) == [1]


def test_detect_multi_edge():
    sp1 = make_fake_spectrum(
        formula="FeCrO20", absorber=["Fe", "Cr"], edge="k", nscans=1
    )
    sp2 = make_fake_spectrum(
        formula="Nb2Y3O2Zr", absorber=["Y", "Nb", "Zr"], edge="K", nscans=1
    )

    spectra = [sp1, sp2]
    target = [2, 3]
    for i in range(len(spectra)):
        e0, e0_idx, num_e0 = calc_e0(spectra[i].energy, spectra[i].mu)
        assert target[i] - 1 <= num_e0 <= target[i] + 1


def test_detect_with_noise():
    spectrum_1 = make_fake_spectrum(
        formula="FeSeSi2",
        absorber="Fe",
        edge="K",
        npoints=800,
        pre=30,
        post=800,
        exafs=True,
        nscans=1,
    )
    spectrum_2 = make_fake_spectrum(
        formula="Se",
        absorber="Se",
        edge=["K", "L1"],
        npoints=1000,
        pre=100,
        post=900,
        exafs=True,
        nscans=1,
    )
    spectrum_3 = make_fake_spectrum(
        formula="CuNiGa",
        absorber=["Cu", "Ni", "Ga"],
        edge="K",
        post=600,
        pre=100,
        npoints=1000,
        exafs=True,
        nscans=1,
    )
    spectra = [spectrum_1, spectrum_2, spectrum_3]

    edge_nos = [1, 2, 3]
    i = 0
    for s in spectra:
        gs = add_gaussian_noise(s.energy, s.mu, max(s.mu) / 100, 0)
        ps = add_poisson_noise(s.mu, 0.01, max(s.mu) / 20)
        no_tmp = edge_nos[i]
        for y in [gs, ps]:
            e0, e0idx, num_e0 = calc_e0(s.energy, y)
            assert no_tmp - 1 < num_e0 <= no_tmp + 1
        i += 1


def test_detect_increasing_noise():
    spectrum_1 = make_fake_spectrum(
        formula="Fe", absorber="Fe", edge="K", npoints=800, nscans=1
    )
    spectrum_2 = make_fake_spectrum(
        formula="Se",
        absorber="Se",
        edge=["K", "L1"],
        npoints=1000,
        pre=100,
        post=900,
        nscans=1,
    )
    spectrum_3 = make_fake_spectrum(
        formula="CuONi2Ga4",
        absorber=["Cu", "Ni", "Ga"],
        edge="K",
        post=600,
        pre=100,
        npoints=1000,
        nscans=1,
    )
    spectra = [spectrum_1, spectrum_2, spectrum_3]
    edge_nos = [1, 2, 3]
    for _i in range(100):
        amp_factor = 1 / 1000
        # have an error term? 1/amp_factor/100?
        freq = 1 / 1000
        for j in range(len(spectra)):
            s = spectra[j]
            gs = add_gaussian_noise(s.energy, s.mu, max(s.mu) * amp_factor, 0)
            ps = add_poisson_noise(s.mu, freq, max(s.mu) * amp_factor)
            for y in [gs, ps]:
                e0, e0_idx, num_e0 = calc_e0(s.energy, y)
                assert 0 < num_e0 <= edge_nos[j] + (1 / amp_factor) * 0.1


def test_detect_decreasing_edge_height():
    spectrum = make_fake_spectrum(
        formula="Co", absorber="Co", edge="K", npoints=800, pre=200, post=800, nscans=1
    )
    e0, e0_idx, edge_no = calc_e0(spectrum.energy, spectrum.mu)

    # reduce amp.
    amp = 0.8
    for _i in range(10):
        e0, en, ec = calc_e0(spectrum.energy, spectrum.mu * amp)
        assert ec == 1
        amp -= 0.1
    ytmp = spectrum.mu * 0.5
    for _i in range(100):
        e0, en, ec = calc_e0(spectrum.energy, ytmp)
        ytmp *= 0.5
        assert ec == 1
