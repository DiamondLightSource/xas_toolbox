from xas_toolbox.fitting.fit_filters import check_energy_overlap, interpolate_fit_data
from xas_toolbox.fitting.lstsq_fits import nnls_fit
from xas_toolbox.xas.corrections import pre_edge
from xas_toolbox.xas.data_generation import make_fake_spectrum


class TestFilters:
    formulae = ["MoS2", "Mo", "MoO3", "CaMoO4", "CuO"]
    absorbers = ["Mo", "Mo", "Mo", "Mo", "Cu"]

    def make_spectral_data(self, mode="spectrum"):
        Cu_data = make_fake_spectrum("Cu", "Cu", "K", nscans=1)  # noqa: N806
        Mo_data = make_fake_spectrum("Mo", "Mo", "K", nscans=1)  # noqa: N806
        ref_spectra = [
            make_fake_spectrum(self.formulae[i], self.absorbers[i], "K", nscans=1)
            for i in range(len(self.absorbers))
        ]
        if mode == "spectrum":
            return Mo_data, Cu_data, ref_spectra
        Mo_data = (Mo_data.energy, Mo_data.mu)  # noqa: N806
        Cu_data = (Cu_data.energy, Cu_data.mu)  # noqa: N806
        ref_spectra = [(r.energy, r.mu) for r in ref_spectra]
        return Mo_data, Cu_data, ref_spectra

    def test_overlap_spectrum(self):
        Mo_data, Cu_data, ref_spectra = self.make_spectral_data(mode="spectrum")  # noqa: N806
        d, c = check_energy_overlap(Mo_data, ref_spectra)
        assert (len(c)) == len(ref_spectra) - 1
        assert isinstance(d, type(Mo_data))
        assert isinstance(c, type(ref_spectra))
        assert isinstance(c[0], type(ref_spectra[0]))

        d, c = check_energy_overlap(Cu_data, ref_spectra)
        assert len(c) == 1

    def test_overlap_xy(self):
        Mo_data, Cu_data, ref_spectra = self.make_spectral_data(mode="MODE")  # noqa: N806
        d, c = check_energy_overlap(data=Mo_data, components=ref_spectra)
        assert (len(c)) == len(ref_spectra) - 1
        assert isinstance(d, type(Mo_data))
        assert isinstance(c, type(ref_spectra))
        assert isinstance(c[0], type(ref_spectra[0]))

        d, c = check_energy_overlap(Cu_data, ref_spectra)
        assert len(c) == 1

    def test_interp_spectrum(self):
        Mo_data = make_fake_spectrum("Mo", "Mo", "K", nscans=1)  # noqa: N806
        npoints = [1000, 1000, 800, 900, 950]
        ref_spectra = [
            make_fake_spectrum(
                self.formulae[i], self.absorbers[i], "K", nscans=1, npoints=npoints[i]
            )
            for i in range(len(self.absorbers))
        ]
        d, c = check_energy_overlap(data=Mo_data, components=ref_spectra)
        d, c = interpolate_fit_data(data=d, components=c)
        assert list(d.keys()) == ["x", "y"]
        assert len(d["x"]) == len(d["y"]) <= min(npoints)
        for ci in c:
            assert (list(ci.keys())) == ["x", "y"]
            assert len(ci["x"]) == len(ci["y"]) <= min(npoints)

    def test_interp_components_spectrum(self):
        npoints = [1000, 910, 800, 900, 950]
        ref_spectra = [
            make_fake_spectrum(
                self.formulae[i], self.absorbers[i], "K", nscans=1, npoints=npoints[i]
            )
            for i in range(len(self.absorbers) - 1)
        ]
        d, c = interpolate_fit_data(data=None, components=ref_spectra)
        assert d is None
        assert len(c) == len(ref_spectra)
        for ci in c:
            assert len(ci["x"]) == len(ci["y"]) <= min(npoints)

    def test_interp_components_spectrum_xy(self):
        npoints = [1000, 910, 800, 900, 950]
        ref_spectra = [
            make_fake_spectrum(
                self.formulae[i], self.absorbers[i], "K", nscans=1, npoints=npoints[i]
            )
            for i in range(len(self.absorbers) - 1)
        ]
        ref_spectra = [(r.energy, r.mu) for r in ref_spectra]
        d, c = interpolate_fit_data(data=None, components=ref_spectra)
        assert d is None
        assert len(c) == len(ref_spectra)
        for ci in c:
            assert len(ci["x"]) == len(ci["y"]) <= min(npoints)

    def test_interp_spectrum_xy(self):
        Mo_data = make_fake_spectrum("Mo", "Mo", "K", nscans=1)  # noqa: N806
        Mo_data = (Mo_data.energy, Mo_data.mu)  # noqa: N806
        npoints = [1000, 1000, 800, 900, 950]
        ref_spectra = [
            make_fake_spectrum(
                self.formulae[i], self.absorbers[i], "K", nscans=1, npoints=npoints[i]
            )
            for i in range(len(self.absorbers))
        ]
        ref_spectra = [(r.energy, r.mu) for r in ref_spectra]

        d, c = check_energy_overlap(data=Mo_data, components=ref_spectra)
        d, c = interpolate_fit_data(data=d, components=c)
        assert list(d.keys()) == ["x", "y"]
        assert len(d["x"]) == len(d["y"]) <= min(npoints)
        for ci in c:
            assert (list(ci.keys())) == ["x", "y"]
            assert len(ci["x"]) == len(ci["y"]) <= min(npoints)


def test_nnls_1d_spectrum():
    formulae = ["MoS2", "Mo", "MoO3", "CaMoO4"]
    ref_data = []
    for f in formulae:
        tmp = make_fake_spectrum(f, "Mo", "k", nscans=1)
        norm, flat, step = pre_edge(tmp.energy, tmp.mu)
        ref_data.append((tmp.energy, flat))

    for i in range(len(formulae)):
        data = ref_data[i]
        solution, concentrations = nnls_fit(
            data=data, components=ref_data, component_names=formulae, filter=False
        )
        assert len(concentrations) == len(formulae)
        assert list(solution.keys()) == ["fit data", "fitted values", "fitted data"]
        assert round(concentrations[i], 2) > 0.99

        solution, concentrations = nnls_fit(
            data=data, components=ref_data, component_names=formulae, filter=True
        )
        assert len(concentrations) <= len(formulae)
        assert list(solution.keys()) == ["fit data", "fitted values", "fitted data"]


def test_nnls_nd():
    formulae = ["Cr", "Cr2Na2O7", "YCrO3"]
    ref_data = []
    for f in formulae:
        tmp = make_fake_spectrum(f, "Cr", "k", nscans=1)
        norm, flat, step = pre_edge(tmp.energy, tmp.mu)
        ref_data.append((tmp.energy, flat))

    for i in range(len(formulae)):
        data = make_fake_spectrum(formulae[i], "Cr", "K", nscans=30)
        norm, flat, step = pre_edge(data.energy, data.mu)
        data = (data.energy, norm)
        solution, concentrations = nnls_fit(
            data=data, components=ref_data, component_names=formulae, filter=False
        )
        assert concentrations.shape[1] == len(formulae)
        assert concentrations.shape[0] == norm.shape[0]

        solution, concentrations = nnls_fit(
            data=data, components=ref_data, component_names=formulae, filter=True
        )
        assert concentrations.shape[1] <= len(formulae)
        assert concentrations.shape[0] == norm.shape[0]
