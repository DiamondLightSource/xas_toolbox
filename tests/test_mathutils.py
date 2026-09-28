import numpy as np

from xas_toolbox.io import XasMeasurement
from xas_toolbox.utils.maths.filter import normalise_matrix, resize_with_replacement
from xas_toolbox.utils.maths.interpolate import interpolate_with_bounds
from xas_toolbox.utils.maths.start_stop import _get_start_stop, _mean_start_stop
from xas_toolbox.xas.data_generation import make_fake_spectrum, mix_scans


class TestInterpolation:
    od_data = make_fake_spectrum("FeO2", "Fe", "k", nscans=1)
    od_data2 = make_fake_spectrum("FeClH2", "Fe", "k", nscans=1)
    xnd, ynd = mix_scans(
        profiles=np.vstack((np.linspace(0, 1, 100), np.linspace(1, 0, 100))).T,
        absorption=[od_data.mu, od_data2.mu],
        energy=od_data.energy,
    )

    def getval(self, value: str):
        if value == "energy":
            return self.xnd
        if value in ["mu", "mutrans"]:
            return self.ynd

    def test_interp_nd(self):

        nd_data = XasMeasurement(get_value=self.getval, mode="transmission")
        data = nd_data
        ebounds = (data.energy[10], data.energy[-20])
        npoints = 1000
        fx, x = interpolate_with_bounds(data.energy, data.mu, npoints, ebounds)
        assert len(x) == npoints
        assert x[0] == ebounds[0] and x[-1] == ebounds[-1]
        assert fx.shape[1] == len(x)
        assert fx.shape[0] == data.mu.shape[0]

    def test_interp_1d(self):
        data = self.od_data
        ebounds = (data.energy[40], data.energy[-10])
        npoints = 300
        fx, x = interpolate_with_bounds(data.energy, data.mu, npoints, ebounds)

        assert len(x) == npoints and len(fx) == len(x)
        assert x[0] == ebounds[0] and x[-1] == ebounds[-1]


class TestStartStop:
    def test_base_example(self):
        arr = [0, 10, 15, 25, 30]
        diff = np.diff(arr)
        ss = _get_start_stop(arr, diff, min_spacing=6)
        assert len(ss) == round(len(arr) / 2) + 1
        ss = _get_start_stop(arr, diff, min_spacing=2)
        assert len(ss) == len(arr)
        ss = _get_start_stop(arr, diff, min_spacing=50)
        assert len(ss) == 1
        assert len(ss[0]) == 2

        ssmean = _mean_start_stop(arr, diff, min_spacing=5)
        assert len(ssmean) == round(len(arr) / 2) + 1
        ssmean = _mean_start_stop(arr, diff, min_spacing=2)
        assert len(ssmean) == len(arr)
        ssmean = _mean_start_stop(arr, diff, min_spacing=100)
        assert len(ssmean) == 1
        assert ssmean[0] == np.median(arr)


def test_replacement():
    vals = [[0, 2, 3], [4, 5], [2, 3, 6], [1]]
    out = resize_with_replacement(vals)
    assert out.shape[0] == len(vals)
    assert out.shape[1] == np.max([len(v) for v in vals])
    assert str(out[-1][-1]) == "nan"

    out = resize_with_replacement(vals, 100)
    assert out.shape[0] == len(vals)
    assert out[-1][-1] == 100


def test_normalise_matrix():
    rng = np.random.default_rng(10)
    c_init = rng.normal(0.5, 0.5, (100, 10))
    normalise_matrix(c_init, axis=0)
    sums = np.sum(c_init, axis=0)
    for s in sums:
        assert round(s, 0) == 1

    c_2 = rng.normal(0.5, 0.5, (50, 10))
    normalise_matrix(c_2, axis=-1)
    sums = np.sum(c_2, axis=-1)
    for s in sums:
        assert round(s, 0) == 1
