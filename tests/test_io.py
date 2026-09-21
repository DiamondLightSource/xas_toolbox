import pytest

from xas_toolbox.io import XasMeasurement, read_data
from xas_toolbox.xas.data_generation import make_fake_spectrum


@pytest.fixture
def ascii_file(tmp_path_factory: pytest.TempPathFactory):
    import numpy as np

    spectrum = make_fake_spectrum("UPH4", "U", "L2", nscans=1)
    comments = ["#---------------------------------", "#  ['energy', 'mutrans']"]
    data = np.vstack((spectrum.energy, spectrum.mu)).T

    data_dir = tmp_path_factory.mktemp("data")
    ascii_path = data_dir.joinpath("data.dat")
    with open(ascii_path, "w") as f:
        [f.writelines(f"{c}\n") for c in comments]
        [f.write(f"{d}\n".replace("[", "").replace("]", "")) for d in data]
        f.close()

    return ascii_path


@pytest.fixture
def xdi_file(tmp_path_factory: pytest.TempPathFactory):

    # using some fields from the xdi example at:
    # https://github.com/XraySpectroscopy/XAS-Data-Interchange/blob/master/specification/spec.md

    data_dir = tmp_path_factory.mktemp("data")
    xdi_path = data_dir.joinpath("data.xdi")

    headers = ["# energy", "i0", "itrans", "mutrans"]
    data = [
        "8779.0  149013.7  550643.089065  -1.3070486",
        "8789.0  144864.7  531876.119084  -1.3006104",
        "8799.0  132978.7  489591.10592  -1.3033816",
        "8809.0  125444.7  463051.104096  -1.3059724",
        "8819.0  121324.7  449969.103983  -1.3107085",
        "8829.0  119447.7  444386.117562  -1.3138152",
        "8839.0  119100.7  440176.091039  -1.3072055",
        "8849.0  117707.7  440448.106567  -1.3195882",
        "8859.0  117754.7  442302.10637  -1.3233895",
        "8869.0  117428.7  441944.116528  -1.3253521",
        "8879.0  117383.7  442810.120466  -1.327693",
        "8889.0  117185.7  443658.11566  -1.3312944",
    ]

    comments = [
        "# XDI/1.0 GSE/1.0",
        "# Column.1: energy eV",
        "# Column.2: i0",
        "# Column.3: itrans",
        "# Column.4: mutrans",
        "# Element.edge: K",
        "# Element.symbol: Cu",
        "#-----",
    ]
    with open(xdi_path, "w") as f:
        [f.writelines(f"{c}\n") for c in comments]
        [f.writelines(f"{h} ") for h in headers]
        [f.write(f"{d}\n") for d in data]
        f.close()

    return xdi_path


@pytest.fixture
def b18_file(tmp_path_factory: pytest.TempPathFactory):
    # making example b18 file with some random data in it.
    import h5py

    spectrum = make_fake_spectrum("CuO2", "Cu", "K", nscans=1)
    data_dir = tmp_path_factory.mktemp("data")
    h5_path = data_dir.joinpath("data.nxs")
    with h5py.File(h5_path, "w") as f:
        entry1 = f.create_group("entry1")
        instrument = entry1.create_group("instrument")
        instrument.create_dataset("name", data="b18")

        ionchambers = entry1.create_group("qexafs_counterTimer01")
        ionchambers.create_dataset("lnI0It", data=spectrum.mu)
        ionchambers.create_dataset("qexafs_energy", data=spectrum.energy)
        f.close()
    return h5_path


def test_read_ascii(ascii_file):
    tst = read_data(ascii_file)
    assert isinstance(tst, XasMeasurement)
    assert hasattr(tst, "mu")
    assert hasattr(tst, "energy")
    assert tst.mode == "transmission"


def test_read_xdi(xdi_file):
    tst = read_data(xdi_file, mode="fluorescence")
    assert isinstance(tst, XasMeasurement)
    assert hasattr(tst, "mu")
    assert hasattr(tst, "energy")

    tst = read_data(xdi_file, mode="transmission")
    assert isinstance(tst, XasMeasurement)
    assert hasattr(tst, "mu")
    assert hasattr(tst, "energy")


def test_read_nexus(b18_file):
    tst = read_data(b18_file)
    assert tst.mode == "transmission"
    assert hasattr(tst, "mu")
    assert hasattr(tst, "energy")
    assert hasattr(tst, "transData")
