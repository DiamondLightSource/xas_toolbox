from .single_file import _find_instrument
from xas_toolbox.io.filetypes import B18Reader, I20_1Reader, I20Reader, XdiReader, AsciiReader
from xas_toolbox.utils.scan_data import ScanData, ScanMeta, ElementMeta
import logging
from pathlib import Path
import numpy as np
from typing import Union, Any
from collections import Counter

logger = logging.getLogger(__name__)

class MultipleFileReader:
    """
    Class for handling reading of multiple files.
    """
    def __init__(self, paths:list[Path|str], meta_filter:bool):
        """
        Make the reader from the path list.

        Args:
            paths (list[Path | str]): List of paths to data.
            meta_filter (bool): Whether to remove scans with different absorbing elements
                                and eges from the reader.
        """
        self.paths = paths
        self.readers = None
        self.__filtered = meta_filter

        self._make_readers()
        if meta_filter == True:
            self._filter_by_edge()

    def _make_readers(self)->None:
        """
        Add a dictionary of individual readers with keys
        corresponding to their file paths to `self`.
        """
        self.readers = {}
        for path in self.paths:
            if path.suffix == ".nxs":
                instrument = _find_instrument(path)
                if instrument == "b18": self.readers[f"{path}"] = B18Reader(path)
                elif instrument == "i20": self.readers[f"{path}"] = I20Reader(path)
                elif instrument == "i20-1": self.readers[f"{path}"] = I20_1Reader(path)
                else:
                    logger.warning(f"{path} skipped: {instrument} not supported.")
                    continue

            elif path.suffix == ".dat": self.readers[f"{path}"] = AsciiReader(path)
            elif path.suffix == ".xdi": self.readers[f"{path}"] = XdiReader(path)
            else: logger.warning(f"{path} skipped- filetype not implemented.")
        if self.readers == {}:
            self.readers = None

    def _filter_by_edge(self):
        """
        Remove files with different absorbing atoms + edges from the stack. <br>
        This is done if the `meta_filter` argument is set to `True` (by-default).
        """
        symbol_dict = {}
        for k, v in self.readers.items():
            atsym = v.get_value("symbol")
            edge = v.get_value("edge")
            symbol_dict[k] = {"atsym": atsym, "edge": edge.lower()}
        
        symbols = Counter([v["atsym"] for k, v in symbol_dict.items()])
        edges = Counter([v["edge"] for k, v in symbol_dict.items()])
        symbol = symbols.most_common(1)[0][0]
        edge = edges.most_common(1)[0][0]

        to_rm = []
        for k in self.readers.keys():
            if symbol_dict[k]["atsym"] != symbol:
                logger.warning(f"{k} has different absorber ({symbol_dict[k]['atsym']})")
                to_rm.append(k)
            if (symbol_dict[k]["edge"] != edge) and (k not in to_rm):
               logger.warning(f"{k} has different edge ({symbol_dict[k]['edge']})")
               to_rm.append(k)

        self.readers = {k:v for k, v in self.readers.items() if k not in to_rm}
        self.paths = [p for p in self.paths if f"{p}" not in to_rm]

    def _get_ScanData(self, value:ScanData)->np.ndarray|None:
        """
        Numerical data from the files in the stack are read in this way. <br>
        Currently if lengths between the same values in different scans are different
        the final array is padded with NaN values to accomodate this.

        Arguments:
            value (ScanData): Value to load.
        
        Returns:
            out (np.ndarray|None): Array of stacked data for given value.
        """
        _skip = []
        # remove the scans that don't include the given value.
        nscans = len(self.paths)
        #checking shapes
        dshapes = []
        for k, v in self.readers.items():
            dim_tmp = v._get_dims(value)
            if dim_tmp is not None:
                dshapes.append(v._get_dims(value))
            else:
                logger.info(f"path {k} has no attribute {value}")
                _skip.append(k)
                continue
        relpaths = [p for p in self.paths if f"{p}" not in _skip]
        ndims = list(set([len(d) for d in dshapes]))
        nscans = len(relpaths)

        if len(ndims) == 1 and ndims == [1]:
            # stack of 1d scans with same/varying lengths.
            dlengths = list(set([d[0] for d in dshapes]))
            out = np.empty((nscans, max(dlengths)))
            out.fill(np.nan)

            for i in range(nscans):
                dtmp = self.readers[f"{relpaths[i]}"].get_value(value)
                if dtmp is not None:
                    out[i,:len(dtmp)] = dtmp
                else: continue
            
        else:
            # stack of scans with mixed depths and lengths.
            dlengths = [d[-1] for d in dshapes]
            dwidths = []

            for i in range(nscans):
                if len(dshapes[i]) == 1:
                    dwidths.append(1)
                else: dwidths.append(dshapes[i][0])
            if dwidths == [] or dlengths == []: return None
            out = np.empty((sum(dwidths), max(dlengths)))
            out.fill(np.nan)

            current = 0
            for i in range(nscans):
                dtmp = self.readers[f"{relpaths[i]}"].get_value(value)
                if dtmp is None: continue

                if i < nscans -1:
                    out[current:dwidths[i], :dlengths[i]] = dtmp
                else: out[current:, :dlengths[i]] = dtmp
                current += dwidths[i]

        return out
    
    def _get_ScanMeta(self, value:ScanMeta)->list:
        """
        Scan metadata is read via this, it will always give a list
        of values since different scans comprise the stack.

        Args:
            value (ScanMeta): Value to access.

        Returns:
            vals (list[Any]): List of values for each scan.
        """
        vals = []
        for k, v in self.readers.items():
            vals.append(v.get_value(value))
        return vals

    def _get_ElementMeta(self, value:ElementMeta)->str|list[str]:
        """
        Load element metadata (e.g. absorbing atom, edge) for the stack.
        If the data making the stack has been filtered this will give single
        values for element metadata, else it will give a list for each scan in
        the stack.

        Args:
            value (ElementMeta): Value to access.

        Returns:
            out (str | list[str]): Single value/list of values requested.
        """
        if self.__filtered == True:
            tmp = self.readers[list(self.readers.keys())[0]]
            return tmp.get_value(value)
        else:
            vals = []
            for k, v in self.readers.items():
                vals.append(v.get_value(value))
            return vals

    def get_value(self,val:Union[ScanData, ElementMeta, ScanMeta])\
                        -> Union[np.ndarray, str, list[Any], None]:
        """
        Method for getting data from the stack of data, to be used
        when accessing any of the scan data/meta data.

        Args:
            val (Union[ScanData, ElementMeta, ScanMeta]): Value to access

        Returns:    
            out Union[np.ndarray, str, list[Any], None]: The value requested.
        """
        if val in ScanData.__members__.keys():
            return self._get_ScanData(val)
        elif val in ElementMeta.__members__.keys():
            return self._get_ElementMeta(val)
        elif val in ScanMeta.__members__.keys():
            return self._get_ScanMeta(val)
        
        else: return ValueError("Value not a valid XAS parameter.")
