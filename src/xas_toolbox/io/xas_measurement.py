"""
XAS measurement class here. <br>
Goals are: 
<ul> (1) Be able to "hold" data from ascii, xdi or nexus files in the same way. </ul>
<ul>(2) Be able to enable "lazy" loading of data from nexus files. </ul>
<ul>(3) be able to have arbitrary data set in the XasMeasurement. </ul> 
"""
from collections.abc import Callable
from typing import Any, Union, Literal
from xas_toolbox.utils.scan_data import ScanData, ScanMeta, ElementMeta
from xas_toolbox.io.measurement_types import (TransMeasurement, FluorMeasurement, RefMeasurement, McaMeasurement,
                                 XasMeta, AuxMeasurement)
import numpy as np
import logging; logger = logging.getLogger(__name__)


class XasMeasurement:
    """
    Class for Xas-Measurement data, these could be a single scan, stacked scans with common
    energy axis or a collection of scans with given absorption values and energies.

    Properties:
        mu (np.ndarray): Principal recorded value for absorption.
        energy (np.ndarray): Energy values.

    Attributes:
        mode (Literal["fluorescence", "transmission"]): What measurement mode the scan(s) are in.
        meta (XasMeta): Metadata, could contain absorbing edge, atom and scan start + end time.
        auxData (AuxMeasurement): Collection of measurements with `energy`, `i0` and any auxilliary
                                    measurements (like time, temperature).
        transData (TransMeasurement|None): Collection of measurements associated with transmission-mode data
                                            i.e. `itrans`, `mutrans`.
        fluorData (FluorMeasurement|None): Collection of measurements associated with fluorescence-mode data
                                            i.e. `ifluor`, `mufluor`.
        refData (RefMeasurement|None): Collection of reference data measurements (`murefer`, `irefer`).
        mcaData (McaMeasurement|None) Collection of measurements related to MCA (`MCAs`, `dtc_factors`, ect).
    """
    def __init__(self, get_value:Callable[[Union[ScanData, ScanMeta, ElementMeta, str]], Any],
                 mode:Literal["fluorescence", "transmission"]=None):
        
        """
        Create an `XasMeasurement` object.
        
        Arguments:
            get_value (Callable[[Union[ScanData, ScanMeta, ElementMeta, str]], Any]): Function to use
                                when scan data is requested.
            mode (Literal["fluorescence", "transmission"], Optional): Mode the scan is in (if not set this
                                will be found automatically).

        Example:
            - for a .nxs file which has energy and mu inside an internal path: "data/xy"
            ```
            import h5py; from functools import partial
            from xas_toolbox.io import XasMeasurement

            def get_fn(filename, var):
                if var == "mutrans": var = "mu" #forcing a "mode" assigned to the data.
                with h5py.File(filename) as f:
                if var in f["data/xy"].keys():
                    out = f[f"data/xy/{var}"][...]
                else: out = None
                return out

            path = "a_path.nxs"
            data = XasMeasurement(partial(get_fn, path))
            ```
        """

        self._mu = None; self._energy = None

        self.mode = mode

        self._get_value = get_value

        self.meta = XasMeta(get_value)

        self.auxData = None
        self.transData = None
        self.fluorData = None
        self.refData = None
        self.mcaData = None
        self.set_mode(get_value, mode)


    @property
    def energy(self)->None:
        """
        Set energy !!!
        """
        if self._energy is None:
            # run set-mode(), return self._energy
            self.set_mode(self._get_value, self.mode)
        return self._energy
    
    @energy.setter
    def energy(self, energy):
        if isinstance(energy, str):
            raise AttributeError("Value must be numeric.")
        self._energy = energy

    @property
    def mu(self)->None:
        """
        Set mu !
        """
        if self._mu is None:
            self.set_mode(self._get_value, self.mode)
        return self._mu
    
    @mu.setter
    def mu(self, mu):
        if isinstance(mu, str):
            raise AttributeError("Value not numeric.")
        self._mu = mu


    def set_mode(self, get_value:Callable[[Union[ScanData, ScanMeta, ElementMeta, str]], Any],
                                mode:Literal["fluorescence", "transmission"]=None):
        """
        Determine what the principal values to go in `self.energy, self.mu` are.
        """
        self.auxData = AuxMeasurement(get_value)
        self.transData = TransMeasurement(get_value)
        self.fluorData = FluorMeasurement(get_value)
        self.refData = RefMeasurement(get_value)
        self.mcaData = McaMeasurement(get_value)

        self._energy = self.auxData.energy

        if mode is None:
            self.mode = "fluorescence"

        if self.mode == "fluorescence":
            self._mu = self.fluorData.mufluor
            if self._mu is None:
                if self.fluorData.ifluor and self.auxData.i0:
                    self._mu = self.auxData.i0/self.fluorData.ifluor
                else:
                    logger.warning("Cannot find mu (fluorescence)")
        if self._mu is None:
            logger.info("Setting mode to transmission")
            self.mode = "transmission"

        if self.mode == "transmission":
            self._mu = self.transData.mutrans
            if self._mu is None:
                if self.transData.itrans and self.auxData.i0:
                    self._mu = -np.log(self.transData.itrans/self.auxData.i0)
                else:
                    logger.warning("Cannot find mu (transmission).")

        if self._mu is None:
            raise AttributeError("Cannot find experimental absorption.")
            



