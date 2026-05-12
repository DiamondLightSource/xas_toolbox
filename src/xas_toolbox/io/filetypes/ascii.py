"""
Reader for ASCII (.dat) files.
"""

# todo: get rid of larch dep. here!
import logging
from pathlib import Path

import numpy as np
from larch.io import read_ascii

from xas_toolbox.utils.scan_data import ScanData

logger = logging.getLogger(__name__)


class AsciiReader:
    """
    Class to read and hold data for .dat files.
    """

    def __init__(self, path: Path):
        """
        Read in the data, since its a .dat file all
        data needs to be read in at once. <br>
        Naming may differ from the XDI standards too. <br>
        Not able to read metadata because it is all stored in an unstructured
        header.
        """
        self.path = path

        self.data = read_ascii(path)
        self.rename = {"mutrans": ["lnI0It", "xmu"], "mufluor": "ff_i0", "energy": "e"}

    def get_value(self, val: ScanData) -> np.ndarray | None:
        """
        Get a requested value from .dat file if it exists and is a valid
        parameter.

        Arguments:
            val (ScanData): Value to return.

        Returns:
            out (np.ndarray | None): Value from the file.
        """
        out = None
        if val == "path":
            return self.path

        if hasattr(self.data, val):
            out = getattr(self.data, val)
        else:
            valsnew = self.rename[val]
            if isinstance(valsnew, str):
                if hasattr(self.data, valsnew):
                    out = getattr(self.data, valsnew)
            else:
                for v in valsnew:
                    if hasattr(self.data, v):
                        out = getattr(self.data, v)
                        break
        if out is None:
            logger.warning(f"{val} not found in scan.")

        return out

    def _get_dims(self, val: ScanData) -> tuple[int] | None:
        """
        Get the shape of a requested value.

        Arguments:
            val (ScanData): Value (must be a `ScanData` member).

        Returns:
            out (tuple[int] | None): Shape of the value or `None` if value not found.
        """
        out = self.get_value(val)
        if out is not None:
            return out.shape
        else:
            return
