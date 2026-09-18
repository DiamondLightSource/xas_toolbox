"""
Reader for legacy beamline i20-1 (EDE).
"""

import logging
import re

import h5py

from xas_toolbox.utils.scan_data import ElementMeta

from .nxs_reader import NxsReader

logger = logging.getLogger(__name__)


class I20_1Reader(NxsReader):  # noqa: N801
    """
    Class to hold relevant paths to XAS data for beamline i20-1 (EDE).

    Attributes:
        transParams (dict): Paths to transmission data: <b>mutrans, energy, i0, itrans
        </b>.
        fluorParams (dict): Paths to fluorescence data: <b>mufluor, energy, ifluor</b>.
        referParams (dict): Paths to reference data: <b>murefer, irefer</b>, these will
          all be
                            `None`.
        mcaParams (dict): Paths to MCA data <b>dtc_factors, raw_scaler_in_window, mcas<
        /b>,
                            these will all be `None`.
    """

    def __init__(self, path):
        super().__init__(path)

        # detector options:
        self._detectors = ["frelon", "xh"]

        # i20-1 specific paths:
        self.scanMeta["repetition_files"] = None
        self.elementMeta["_path"] = [
            "entry1/metaData",
            "polynomial",
            "referenceDataFileName",
        ]

        self.monoParams["energy"] = "entry1/lnI0It/energy"
        self.monoParams["i0"] = "entry1/metaData/i0"

        self.transParams["mutrans"] = "entry1/lnI0It/data"
        self.transParams["itrans"] = "entry1/metaData/it"

        self._make_fluor_paths()

    def _make_fluor_paths(self) -> None:
        """
        If xh/frelon used in measurement, populate the fluorParams
        dict with the correct nexus file paths to data.
        """
        det = self._find_detector()
        if det is not None:
            self.fluorParams["mufluor"] = f"entry1/{det}/data"
            self.fluorParams["energy"] = f"entry1/{det}/energy"
            self.fluorParams["ifluor"] = f"entry1/{det}/it"
        else:
            logger.info("Alt. detector cannot be found.")

    def _make_elementMeta(self) -> None:  # noqa: N802

        mdetails = self.elementMeta["_path"]
        with h5py.File(self.path, "r") as f:
            if mdetails[0] in f.keys():
                meta = f[mdetails[0]]
                if mdetails[1] in meta.attrs.keys():
                    info = meta.attrs[f"{mdetails[1]}"][:].decode("utf-8")
                else:
                    return

                sample = list(filter(lambda p: mdetails[-1] in p, info.split(",")))
                if len(sample) < 1:
                    return
                sdetails = re.findall("[A-Z][a-z]?_[A-Z]\\d?", sample[0])[0]
                symbol, edge = sdetails.split("_")
                self.elementMeta["symbol"] = symbol
                self.elementMeta["edge"] = edge
                self.elementMeta["reference"] = symbol
                self.elementMeta["ref_edge"] = edge

            else:
                return

    def _get_ElementMeta(self, val: ElementMeta) -> str | None:  # noqa: N802
        """
        Get element metadata from scan (e.g. absorbing edge and element).
        """
        if val not in self.elementMeta.keys():
            logger.warning(f"{val} not present in file")
            return

        if self.elementMeta[val] is None:
            self._make_elementMeta()

        return self.elementMeta[val]
