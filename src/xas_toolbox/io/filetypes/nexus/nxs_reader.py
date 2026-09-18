"""
Base reader for nexus files.
"""

import logging
from pathlib import Path

import h5py
import lxml.etree as etree
import numpy as np

from xas_toolbox.utils.scan_data import ElementMeta, ScanData, ScanMeta

logger = logging.getLogger(__name__)


class NxsReader:
    """
    Base object for reading nexus files. <br>
    This makes dictionaries for storing internal paths to scan data (transmission,
    fluorescence, reference + mcas) and metadata (scan + element). <br>
    It also gives base methods for getting data from each of these fields and a general
    `get_value(value)` function to be used by `XasMeasurement`.

    Attrs:
        path (Path): Path to nexus file.
        scanMeta (dict): Dictionary of paths to <i>start_time, end_time</i> and
                        <i>repetition_files</i>.
        elementMeta (dict): Empty dictionary to be given paths to <i>symbol, edge,
          reference
        </i>
                            and <i>ref_edge</i>.
        transParams (dict): Empty dictionary to be given paths to <i>mutrans, energy, i0
        </i> and
                            <i>itrans</i>.
        fluorParams (dict): Empty dictionary to be given paths to <i>mufluor, energy</i
        > and
                            <i>ifluor</i>.
        referParams (dict): Empty dictionary to be given paths to <i>murefer</i> and
                            <i>irefer</i>.
        mcaParams (dict): Empty dictionary to be given paths to <i>dtc_factors,
                          raw_scaler_in_window</i> and <i>mcas</i>.
    """

    def __init__(self, path: Path):

        self.path = path

        self.scanMeta = {
            "start_time": "entry1/start_time",
            "end_time": "entry1/end_time",
            "repetition_files": "entry1/before_scan/files_in_repetition_scan",
            "path": self.path,
        }

        self.monoParams = {"i0": None, "energy": None}

        self.elementMeta = {
            # could have functions instead of labels being passed through here?
            "symbol": None,
            "edge": None,
            "reference": None,
            "ref_edge": None,
        }

        self.transParams = {"mutrans": None, "energy": None, "i0": None, "itrans": None}

        self.fluorParams = {"mufluor": None, "energy": None, "ifluor": None}

        self.referParams = {"murefer": None, "irefer": None}

        self.mcaParams = {
            "dtc_factors": None,
            "raw_scaler_in_window": None,
            "mcas": None,
        }

    def _find_detector(self) -> str | None:
        """
        Find fluorescence detector used in a scan if multiple configurations
        exist (e.g. for b18 and i20-1).
        """
        det = None
        if hasattr(self, "_detectors"):
            with h5py.File(self.path, "r") as f:
                for d in self._detectors:
                    d_tmp = f"entry1/{d}"
                    if d_tmp in f.keys():
                        det = d
                        break
        return det

    def _get_ScanData(self, val: ScanData) -> np.ndarray | None:  # noqa: N802
        """
        Get a value from the file if it matches any of the values in `ScanData`.<br>
        This is to be used for numeric data in the file (i.e. not metadata).

        Arguments:
            val (ScanData): Value from the nexus file to return.

        Returns:
            out (np.ndarray | None): Value as an array. If the item isn't in the file
                                    `None` will be returned.
        """
        out = None
        _val = getattr(ScanData, val)
        tosearch = _val.mode
        if tosearch is None:
            tosearch = "monoParams"
        if tosearch:
            d_tosearch = self.__getattribute__(tosearch)
            if (val in d_tosearch.keys()) and (d_tosearch[val] is not None):
                with h5py.File(self.path, "r") as f:
                    out = f[d_tosearch[val]][...]
            else:
                logger.warning(f"{val} is not present in scan.")

        return out

    def _get_ScanMeta(self, val: ScanMeta) -> str | list:  # noqa: N802
        """
        Get a requested value from the file if it's classed as
        scan-metadata (i.e. `start_time`, `end_time` or
        `repetition_files`).
        """
        out = None

        if val not in self.scanMeta.keys():
            logger.warning(f"{val} not found in file")
            return

        if val == "path":
            return self.scanMeta["path"]

        _path = self.scanMeta[val]

        if _path is None:
            return None

        with h5py.File(self.path, "r") as f:
            if _path in f.keys():
                out = f[_path][...].astype("T")
            else:
                logger.warning(f"{val} not found in file")
        if hasattr(out, "ndim"):
            if out.ndim == 0:
                out = str(out)  # single values are returned as string
                if out == "":
                    out = None
            else:
                out = out[0].split("\n")  # repetition files need to be split by \n
                if out == [""]:
                    out = None  # repetition files need to be split by \n
        return out

    def _get_ElementMeta(self, val: ElementMeta) -> str | None:  # noqa: N802
        """
        Get requested ElementMeta item from file. <br>
        This will need to be different for i20-1 files.
        """
        out = None

        if (val not in self.elementMeta.keys()) or (self.elementMeta[val] is None):
            logger.warning(f"{val} not found in file")
            return

        parser = etree.XMLParser(recover=True)
        _path = self.elementMeta[val][0]
        xml_name = self.elementMeta[val][-1]

        with h5py.File(self.path, "r") as f:
            if _path not in f.keys():
                return
            meta = f[_path][...].item()
            root = etree.fromstring(meta, parser=parser)
            out = root.findall(xml_name)[0].text

        return out

    def get_value(
        self, val: ScanData | ElementMeta | ScanMeta
    ) -> np.ndarray | str | list | None:

        if val in ScanData.__members__.keys():
            return self._get_ScanData(val)

        elif val in ElementMeta.__members__.keys():
            return self._get_ElementMeta(val)

        elif val in ScanMeta.__members__.keys():
            return self._get_ScanMeta(val)

        else:
            return ValueError("Value not a valid XAS parameter.")

    def _get_dims(self, val: ScanData) -> tuple[int]:
        """
        Get the shape of a requested value.

        Arguments:
            val (ScanData): Value (must be a `ScanData` member).

        Returns:
            out (tuple[int] | None): Shape of the value or `None` if value not found.
        """
        _val = getattr(ScanData, val)
        tosearch = _val.mode
        if tosearch is None:
            tosearch = "monoParams"
        if tosearch:
            dtmp = self.__getattribute__(tosearch)
            if (val in dtmp.keys()) and (dtmp[val] is not None):
                with h5py.File(self.path, "r") as f:
                    nshape = f[dtmp[val]].shape
                return nshape
            else:
                logger.info(f"{val} not found in {self.path}")
                return
