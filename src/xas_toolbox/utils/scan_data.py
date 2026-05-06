"""
Utils for classfying different scan data types.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Literal

@dataclass
class ScanParams:
    """
    Dataclass for what type of scan mode is used for parameters within
    an XAS-file. 

    Attributes:
        valname (str): Name of the data being read in (e.g. mutrans).
        mode (Literal["transParams","fluorParams", "referParams", "mcaParams"] | None):
                      What data acquisition mode the value relates to (set to `None` if
                      not known).
    """
    valname: str
    mode: Literal["transParams", "fluorParams", "referParams", "mcaParams"]

@dataclass
class MetaParams:
    """
    Dataclass for different types of scan metadata within an XAS-file.

    Attributes:
        valname (str): Name of property (e.g. ref_edge).
        mode (Literal["scanMeta", "elementMeta"] | None): What
                     type of metadata the value corresponds to.
    """
    valname: str
    mode: Literal["scanMeta", "elementMeta", None]

class ScanData(ScanParams, Enum):
    """
    Class to capture various scan data an XAS measurement can
    have and what acquisition mode these values correspond to.

    Members:
        `energy, angle,`\n
        `i0, itrans, ifluor, irefer,`\n
        `mutrans, mufluor, murefer,`\n
        `dtc_factors, raw_scaler_in_window, mcas`
    """
    energy = "energy", None # "monoParams"?
    angle = "angle", None # "monoParams"?
    i0 = "i0", None # "???"?
    itrans = "itrans", "transParams"
    ifluor = "ifluor", "fluorParams"
    irefer = "irefer", "referParams"
    mutrans = "mutrans", "transParams"
    mufluor = "mufluor", "fluorParams"
    murefer = "murefer", "referParams"
    dtc_factors = "dtc_factors", "mcaParams"
    raw_scaler_in_window = "raw_scaler_in_window", "mcaParams"
    mcas = "mcas", "mcaParams"

class ScanMeta(MetaParams, Enum):
    """
    Class to capture scan metadata from XAS measurements.

    Members:
        `start_time, end_time, repetition_files`
    """
    start_time = "start_time", "scanMeta"
    end_time = "end_time", "scanMeta"
    repetition_files = "repetition_files", "scanMeta"
    path = "path", "scanMeta"

class ElementMeta(MetaParams, Enum):
    """
    Class to capture element (sample) metadata from XAS
    measurements.

    Members:
        `symbol, edge,`\n
        `ref_symbol, ref_edge`
    """
    symbol = "symbol", "elementMeta"
    edge = "edge", "elementMeta"
    ref_symbol = "ref_symbol", "elementMeta"
    ref_edge = "ref_edge", "elementMeta"
