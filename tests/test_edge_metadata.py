from xas_toolbox.xas.data_generation import make_fake_spectrum
from xas_toolbox.xas.edges.find_e0 import get_edge_symbol


def test_detect_1d_no_symbol():
    spectrum = make_fake_spectrum("CuO", "Cu", "K", nscans=1)
    edge = get_edge_symbol(spectrum.energy, spectrum.mu, symbol=None)
    assert isinstance(edge, tuple)
    assert edge[0] == "Cu"
    assert edge[1].upper() == "K"


def test_detect_1d_symbol():
    spectrum = make_fake_spectrum("HoO2", "Ho", "L1", nscans=1)
    edge = get_edge_symbol(spectrum.energy, spectrum.mu, "Tb")
    assert isinstance(edge, str)


def test_detect_stack():
    s = make_fake_spectrum("PtRu", "Pt", "L3", nscans=30)
    edge = get_edge_symbol(s.energy, s.mu, symbol=None)
    assert isinstance(edge, tuple)
    assert isinstance(edge[0], str)
    assert isinstance(edge[1], str)

    s = make_fake_spectrum("AsO5", "As", "K", nscans=10)
    edge = get_edge_symbol(s.energy, s.mu, symbol="As")
    assert len(list(set(edge))) == 1
    assert list(set(edge)) == ["K"]


def test_detect_multi():
    x = []
    y = []
    for _ in range(5):
        spectrum = make_fake_spectrum("FeO2", "Fe", "K", nscans=1)
        x.append(spectrum.energy)
        y.append(spectrum.mu)
    edge = get_edge_symbol(x, y)
    assert isinstance(edge, list)
    for e in edge:
        assert isinstance(e[0], str)
        assert isinstance(e[1], str)
    assert len(edge) == len(x)

    edge = get_edge_symbol(x, y, "Fe")
    assert len(list(set(edge))) == 1
    assert list(set(edge)) == ["K"]
