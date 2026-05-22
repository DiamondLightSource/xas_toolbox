"""
Reader for beamline b18.
"""

import logging
import os
from pathlib import Path

import h5py
import numpy as np

from .nxs_reader import NxsReader

logger = logging.getLogger(__name__)


class B18Reader(NxsReader):
    """
    Class to hold relevant paths to XAS data for beamline b18.

    Attributes:
        path (Path): Path to file.
        scanMeta (dict): Paths to scan metadata (currently <b>start_time, end_time,
                                                            repetition_files</b>).
        elementMeta (dict): Paths to element metadata (currently <b>symbol, edge, reference,
                                                             ref_edge</b>).
        transParams (dict): Paths to transmission data: <b>mutrans, energy, i0, itrans</b>.
        fluorParams (dict): Paths to fluorescence data: <b>mufluor, energy, ifluor</b>
                            (will all be `None` if the scan is in transmission mode).
        referParams (dict): Paths to reference data: <b>murefer, irefer</b>
                            (will all be `None` if there is no recorded reference).
        mcaParams (dict): Paths to MCA data <b>dtc_factors, raw_scaler_in_window, mcas</b>
                            (will all be `None` if the scan is in transmission mode).
    """  # noqa: E501

    def __init__(self, path: Path):
        super().__init__(path)

        # options for fluorescence detecotrs:
        self._detectors = ["qexafs_FFI0_xspress4Odin", "qexafs_FFI0_xspress3X"]

        # setting beamline-specific paths here.
        self.monoParams["energy"] = "entry1/qexafs_counterTimer01/qexafs_energy"
        self.monoParams["i0"] = "entry1/qexafs_counterTimer01/I0"

        self.transParams["mutrans"] = "entry1/qexafs_counterTimer01/lnI0It"
        self.transParams["itrans"] = "entry1/qexafs_counterTimer01/It"

        self.referParams["murefer"] = "entry1/qexafs_counterTimer01/lnItIref"
        self.referParams["irefer"] = "entry1/qexafs_counterTimer01/IRef"

        self.elementMeta["symbol"] = ["entry1/before_scan/QEXAFS_Parameters", "element"]
        self.elementMeta["edge"] = ["entry1/before_scan/QEXAFS_Parameters", "edge"]

        self._make_fluor_paths()

    def _make_fluor_paths(self):
        """
        Using the fluorescence detector found in the file, populate the
        `fluorParams` and `mcaParams` dictionaries with paths to data.
        If the scan is in transmission mode this will not do anything.
        """
        det = self._find_detector()

        if det is not None:
            self.fluorParams["mufluor"] = f"entry1/{det}/FFI0"
            self.fluorParams["energy"] = f"entry1/{det}/qexafs_energy"

            det2 = det.replace("FFI0", "").replace("qexafs", "").replace("_", "")

            self.fluorParams["ifluor"] = f"entry1/{det2}/FF"
            self.mcaParams["dtc_factors"] = f"entry1/{det2}/dtc factors"
            self.mcaParams["raw_scaler_in_window"] = (
                f"entry1/{det2}/raw scaler in-window"  # noqa: E501
            )

            mca_key = None
            with h5py.File(self.path, "r") as f:
                for k in f[f"entry1/{det2}"].keys():
                    if ".h5" in k:
                        mca_key = k

            exp_dir = str(self.path).split("nexus")[0]
            mca_path = exp_dir + det2 + f"/{mca_key}"
            if os.path.exists(mca_path):
                mca_key = mca_path
            else:
                logger.info("Can't find mca meta file.")

            self.mcaParams["mcas"] = mca_key
        else:
            logger.info("No fluorescence data found.")

    def _get_mca_value(self) -> np.ndarray | None:
        """
        For b18 data the MCA scalars are kept in a different file,
        this function will find them (is it can..) and return
        them concaternated to an Nd-array.
        """
        if self.mcaParams["mcas"] is None:
            logger.warning("Scan is not in fluorescence mode.")
            return

        with h5py.File(self.mcaParams["mcas"], "r") as f:
            scalars = list(filter(lambda k: "scalar" in k, f.keys()))
            dims = [f[n].shape for n in scalars]

        # making empty array:
        dim_y = len(scalars)  # no. scalars
        dim_z = max([s[0] for s in dims])  # energy length
        dim_x = max([s[-1] for s in dims])  # no. elements

        mcas = np.zeros((dim_x, dim_y, dim_z))
        with h5py.File(self.mcaParams["mcas"], "r") as f:
            for i in range(dim_y):
                s = scalars[i]
                shape_tmp = f[s].shape
                mcas[: shape_tmp[-1], i, : shape_tmp[0]] = f[s][...].T
        return mcas

    def _get_ScanData(self, val):  # noqa: N802
        if val.lower() == "mcas":
            return self._get_mca_value()  # noqa: E701
        else:
            return super()._get_ScanData(val)
