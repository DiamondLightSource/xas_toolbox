"""
Reader for beamline i20.
"""

import logging
from .nxs_reader import NxsReader

logger = logging.getLogger(__name__)

class I20Reader(NxsReader):
    """
    Class to hold relevant paths to XAS data for beamline i20.

    Attributes:
        path (str): Path to file.
        scanMeta (dict): Paths to scan metadata (currently <b>start_time, end_time, 
                                                            repetition_files</b>).
        elementMeta (dict): Paths to element metadata (currently <b>symbol, edge, reference,
                                                             ref_edge</b>).
        transParams (dict): Paths to transmission data: <b>mutrans, energy, i0, itrans</b>.
        fluorParams (dict): Paths to fluorescence data: <b>mufluor, energy, ifluor</b>.
        referParams (dict): Paths to reference data: <b>murefer, irefer</b> 
                            (will all be `None` if there is no recorded reference).
        mcaParams (dict): Paths to MCA data <b>dtc_factors, raw_scaler_in_window, mcas</b>.
    """
    def __init__(self, path):
        super().__init__(path)

        # inputting beamline-specific paths here.
        self.elementMeta["symbol"] = ["entry1/before_scan/XAS_Parameters", "element"]
        self.elementMeta["edge"] = ["entry1/before_scan/XAS_Parameters", "edge"]

        self.monoParams["energy"] = "entry1/ionchambers/bragg1WithOffset"
        self.monoParams["i0"] = "entry1/instrument/ionchambers/I0"

        self.transParams["mutrans"] = "entry1/ionchambers/lnI0It"
        self.transParams["itrans"] = "entry1/ionchambers/It"

        self.fluorParams["mufluor"] = "entry1/instrument/xspress4FFI0/FFI0"
        self.fluorParams["energy"] = "entry1/xspress4FFI0/bragg1WithOffset"
        self.fluorParams["ifluor"] = "entry1/instrument/xspress4/FF"

        self.referParams["murefer"] = "entry1/ionchambers/lnItIref"
        self.referParams["irefer"] = "entry1/instrument/ionchambers/I0"

        self.mcaParams["mcas"] = "entry1/xspress4/MCAs"
        self.mcaParams["dtc_factors"] = "entry1/xspress4/dtc factors"
        self.mcaParams["raw_scaler_in_window"] = "entry1/xspress4/raw scaler in-window"
