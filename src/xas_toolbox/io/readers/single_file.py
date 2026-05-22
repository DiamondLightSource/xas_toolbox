from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py

from xas_toolbox.io.filetypes import (
    B18Reader,
    I20_1Reader,
    I20Reader,
    LegacyB18Reader,
    XdiReader,
)
from xas_toolbox.utils.scan_data import ScanData


def _find_instrument(path: Path) -> str:
    """
    Find instrument used for a scan when nexus file format used.
    """
    with h5py.File(path, "r") as f:
        if "entry1/instrument/name" in f.keys():
            instrument = f["entry1/instrument/name"][...].astype("T")
            return str(instrument)
        else:
            raise AttributeError("Instrument not found.")


def _load_nxs(
    path: Path,
) -> tuple[Callable[[ScanData], Any], B18Reader | I20Reader | I20_1Reader]:
    """
    For a nexus file, set the correct reader class (from `beamlines`)
    and get function to be used in making the `XasMeasurement` object.

    Args:
        path (Path): Path to file.

    Returns:
        tuple (tuple): tuple containing:
            get_data_fn (Callable[[ScanData], Any]): Function to request data from the
            file.
            reader (B18Reader | I20Reader | I20_1Reader): File path/data storage object.
    """
    instrument = _find_instrument(path)
    if instrument == ["b18"]:
        reader = LegacyB18Reader(path)
    if instrument == "b18":
        reader = B18Reader(path)
    elif instrument == "i20":
        reader = I20Reader(path)
    elif instrument == "i20-1":
        reader = I20_1Reader(path)
    else:
        raise NotImplementedError(f"{instrument} not supported.")

    get_data_fn = reader.get_value

    return get_data_fn, reader


def _load_misc(path: Path) -> tuple[Callable[[ScanData], Any], XdiReader]:
    """
    Return correct data store object and getter function for
    xdi/dat files.

    Args:
        path (Path): Path to file.

    Returns:
        tuple (tuple): Tuple containing:
            get_data_fn (Callable[[ScanData], Any]): Function to request data from the
            file.
            reader (XdiReader): File path/data storage object.
    """
    if path.suffix == ".dat":
        raise NotImplementedError("Cannot currently read ascii files.")
        # reader = AsciiReader(path)
    elif path.suffix == ".xdi":
        reader = XdiReader(path)
    get_data_fn = reader.get_value

    return get_data_fn, reader
