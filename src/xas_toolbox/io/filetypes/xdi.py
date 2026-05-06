"""
Reader for XDI (.xdi) files.
"""

import re; import numpy as np
from xas_toolbox.utils.scan_data import ScanData, ScanMeta, ElementMeta
from pathlib import Path
import logging
from typing import Union

logger = logging.getLogger(__name__)

class XdiReader:
    """
    Class to read and hold data for .xdi files.
    """
    def __init__(self, path:Path):

        self.path = path

        self._read_from_columns()
        self._populate_colmap()

    def _get_colmap(self, col_labels:list)->dict:
        """
        Get a value <-> column number mapping for .xdi data. <br>
        Scan variables can be spelt differently in files, if a new 
        variant is found add to `alt_names`.

        Arguments:
            col_labels (list): List of column label names from the
                                xdi file header.

        Returns:
            colmap (dict): Mapping of variable name to column number
                            with checks for alternative naming.
        """
        # some .xdi files don't follow convention?
        # add alternate spellings here.
        alt_names = {"mutrans": "xmu",
                     "itrans": "it",
                     "irefer": "ir"}
        colmap = {}
        for k in ScanData:
            vname = k.valname
            colval = list(filter(lambda l: (vname in l.lower()), col_labels))
            if len(colval) < 1:
                if vname in alt_names.keys():
                    colval = list(filter(lambda l: (alt_names[vname] in l.lower()),
                                         col_labels))
                else:
                    logger.info(f"Cannot find {vname} in file.")
                    colval = None
            if colval:
                colval = colval[0].split(":")[0]
                colmap[vname] = int(re.findall("\d", str(colval))[0]) - 1
            else: colmap[vname] = None

        return colmap
    
    def _populate_colmap(self)->None:
        """
        Change `colmap` to be a mapping of column indices
        to actual arrays.
        """
        
        i0 = self._colmap["i0"]
        if i0: i0 = self._scandata[i0]
        it = self._colmap["itrans"]
        if it: it = self._scandata[it]
        iff = self._colmap["ifluor"]
        if iff: iff = self._scandata[iff]
        ir = self._colmap["irefer"]
        if ir: ir = self._scandata[ir]

        for k, v in self._colmap.items():
            if (k in ScanMeta.__members__.keys()) or (k in ElementMeta.__members__.keys()):
                continue
            if v is not None:
                self._colmap[k] = self._scandata[v,:]
            if v is None:
                if k == "mutrans":
                    self._colmap[k] = -np.log(it/i0)
                if k == "mufluor":
                    if iff is not None:
                        # actually:
                        self._colmap[k] = i0/iff
                if k == "murefer":
                    if ir is not None:
                        self._colmap[k] = -np.log(ir/i0)
    
    def _read_from_columns(self)->None:
        """
        Get (1) Mapping of column indices to XDI parameters (`self._colmap`),
            (2) Block of data contained in xdi file (`self._scandata`).
        """
        with open(self.path, "r") as f:
            xdi = f.read()
        header, cols = re.split("-{3,}", xdi)
        header = header.split("\n"); cols = cols.split("\n")
        col_labels = list(filter(lambda v: ("column" in v.lower()), 
                                                header))
        elm_labels = list(filter(lambda v: ("element" in v.lower()), header))
        
        self._colmap = self._get_colmap(col_labels)

        for k, v in {"symbol": "element.symbol", "edge": "element.edge",
                     "ref_symbol": "element.reference", "ref_edge": "element.ref_edge"}.items():
            tmp = [e for e in elm_labels if v in e.lower()]
            if len(tmp) >= 1: self._colmap[f"{k}"] = tmp[0].split(": ")[-1]
            else: self._colmap[f"{k}"] = None


        dpattern = r"\d+.\d+"
        scandata = [np.array(re.findall(dpattern, f),dtype=np.float64) for f in cols[:-1]]
        self._scandata = np.array([s for s in scandata if len(s)> 1]).T

    def get_value(self, val:Union[ScanData, ScanMeta, ElementMeta])->np.ndarray | None:
        """
        Get a requested value from .xdi file it it exists/can be made 
        and is a valid parameter.

        Arguments:
            val (ScanData): Value to return.

        Returns:
            out (np.ndarray | None): Value from the file.
        """
        out = None
        if val == "path":
            return self.path

        if val not in self._colmap.keys():
            logger.warning(f"{val} not found in scan.")
            return None
        if self._colmap[val] is not None:
            out = self._colmap[val]
        else:
            logger.warning(f"{val} not present in scan.")

        return out

    def _get_dims(self, val:ScanData)->tuple|None:
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
        else: return
