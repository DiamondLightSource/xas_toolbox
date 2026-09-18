"""
Reader for ASCII (.dat) files.
"""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from xas_toolbox.utils.scan_data import ScanData

logger = logging.getLogger(__name__)


class AsciiReader:
    """
    Class to read and hold data for .dat files.
    """

    def __init__(self, path: Path):
        self.path = path

        data, cols = self.read_cols()

        self.set_data(data, cols)

    def read_cols(self):
        """
        Very simply read data and column names from .dat
        file using numpy.

        Returns:
            data (np.ndarray): Array of all columns in file.
            cols (list[str]): List of detected column names.
        """
        data = np.genfromtxt(self.path)
        comments = []
        with open(self.path) as f:
            stop = 0
            for line in f.readlines():
                if "#--" in line:
                    stop += 1
                if stop >= 1:
                    stop += 1
                    comments.append(line)
                if stop > 2:
                    break
        cols = re.findall("[A-Z]?[a-z]+", comments[-1])
        return data, cols

    def set_data(self, data: np.ndarray, colnames: list[str]):
        """
        Set attributes on `self` corresponding  to the column
        names found in `read_cols()`.

        Arguments:
            data (np.ndarray): Array of column values.
            colnames (list[str]): List of column names.
        """
        if "mu" in colnames and "mufluor" not in colnames:
            self.mode = "transmission"
        else:
            self.mode = "fluorescence"

        if data.shape[1] == len(colnames) or data.shape[0] > data.shape[1]:
            data = data.T

        for i in range(len(colnames)):
            name = colnames[i]
            if self.mode == "transmission" and name in ["mu", "xmu"]:
                self.mutrans = data[i, :]

            self.__setattr__(name, data[i, :])

    def get_value(self, val: ScanData | str) -> Any:
        """
        If value present in column names then it is returned.

        Arguments:
            val (Union[ScanData, str]): Value to try and return.

        Returns:
            out (Any|None): `None` if value not present or the value.
        """
        if val == "path":
            return self.path
        if not hasattr(self, val):
            logger.warning(f"{val} not found in scan.")
        else:
            return getattr(self, val)

    def _get_dims(self, val: ScanData) -> tuple | None:
        """
        Get the shape of a requested value.

        Arguments:
            val (ScanData): Value (must be a `ScanData` member).

        Returns:
            out (tuple[int] | None): Shape of the value or `None`
              if value not found.
        """
        out = self.get_value(val)
        if out is not None:
            return out.shape
        else:
            return
