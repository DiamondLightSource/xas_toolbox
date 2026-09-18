"""
Classes for detection-mode specific measurements (i.e. fluorescence, transmission,
reference + metadata).
"""

from collections.abc import Callable
from typing import Any

from xas_toolbox.utils.scan_data import ElementMeta, ScanData, ScanMeta


class AuxMeasurement:
    """
    Class for acquisition-mode agnostic scan data. <br>
    In `scan_data_types` I have set these to have `mode=None`.<br>
    In the nxs file reader they are in a dict. called `monoParams`.

    Properties (for now):
        i0: monitor intensity.
        energy: mono energy.
        angle: mono angle.
        time: time axis

    To add ?
        - temperature: temperature axis
        - ...
    """

    def __init__(self, get_value: Callable[[ScanData | str], Any]):

        self._get_value = get_value
        self._i0 = None
        self._energy = None
        self._angle = None
        self._time = None

    @property
    def i0(self) -> None:
        """
        Set `_i0` internally if no initial value. <br>
            Will return `_i0` if it's already been set.
        """
        if self._i0 is None:
            self._i0 = self._get_value("i0")
        return self._i0

    @i0.setter
    def i0(self, i0):
        if isinstance(i0, str):
            raise AttributeError("Value must be numeric.")
        self._i0 = i0

    @property
    def energy(self) -> None:
        """
        Set `_energy` internally if no initial value. <br>
            Will return `_energy` if it's already been set.
        """
        if self._energy is None:
            self._energy = self._get_value("energy")
        return self._energy

    @energy.setter
    def energy(self, energy):
        if isinstance(energy, str):
            raise AttributeError("Value must be numeric.")
        self._energy = energy

    @property
    def angle(self) -> None:
        """
        Set `_angle` internally if no initial value. <br>
            Will return `_angle` if it's already been set.
        """
        if self._angle is None:
            self._angle = self._get_value("angle")
        return self._angle

    @angle.setter
    def angle(self, angle):
        if isinstance(angle, str):
            raise AttributeError("Value must be numeric.")
        self._angle = angle

    @property
    def time(self) -> None:
        """
        Set `_time` internally if no initial value. <br>
        Will return `_time` it it's already been set.
        """
        if self._time is None:
            self._time = self._get_value("time")
        return self._time

    @time.setter
    def time(self, time):
        if isinstance(time, str):
            raise AttributeError("Value must be numeric.")
        self._time = time


class FluorMeasurement:
    """
    Class for fluorescence-specific measurements.

    Properties:
        ifluor: fluorescence intensity
        mufluor: mu fluorescence
    """

    def __init__(self, get_value: Callable[[ScanData | str], Any]):

        self._get_value = get_value

        self._ifluor = None
        self._mufluor = None

    @property
    def ifluor(self) -> None:
        """
        Set `_ifluor` internally if no initial value. <br>
            Will return `_ifluor` if it's already been set.
        """
        if self._ifluor is None:
            self._ifluor = self._get_value("ifluor")
        return self._ifluor

    @ifluor.setter
    def ifluor(self, ifluor):
        if isinstance(ifluor, str):
            raise AttributeError("Value must be numeric.")
        self._ifluor = ifluor

    @property
    def mufluor(self) -> None:
        """
        Set `_mufluor` internally if no initial value. <br>
            Will return `_mufluor` if it's already been set.
        """
        if self._mufluor is None:
            self._mufluor = self._get_value("mufluor")
        return self._mufluor

    @mufluor.setter
    def mufluor(self, mufluor):
        if isinstance(mufluor, str):
            raise AttributeError("Value must be numeric.")
        self._mufluor = mufluor


class TransMeasurement:
    """
    Class for transmission-specific measurements.

    Properties:
        itrans: transmission intensity
        mutrans: mu transmission
    """

    def __init__(self, get_value: Callable[[ScanData | str], Any]):

        self._get_value = get_value

        self._itrans = None
        self._mutrans = None

    @property
    def itrans(self) -> None:
        """
        Set `_itrans` internally if no initial value. <br>
            Will return `_itrans` if it's already been set.
        """
        if self._itrans is None:
            self._itrans = self._get_value("itrans")
        return self._itrans

    @itrans.setter
    def itrans(self, itrans):
        if isinstance(itrans, str):
            raise AttributeError("Value must be numeric.")
        self._itrans = itrans

    @property
    def mutrans(self) -> None:
        """
        Set `_mutrans` internally if no initial value. <br>
            Will return `_mutrans` if it's already been set.
        """
        if self._mutrans is None:
            self._mutrans = self._get_value("mutrans")
        return self._mutrans

    @mutrans.setter
    def mutrans(self, mutrans):
        if isinstance(mutrans, str):
            raise AttributeError("Value must be numeric.")
        self._mutrans = mutrans


class RefMeasurement:
    """
    Class for reference-specific measurements.

    Properties:
        irefer: reference intensity
        murefer: mu reference
    """

    def __init__(self, get_value: Callable[[ScanData | str], Any]):

        self._get_value = get_value

        self._irefer = None
        self._murefer = None

    @property
    def irefer(self) -> None:
        """
        Set `_irefer` internally if no initial value. <br>
            Will return `_irefer` if it's already been set.
        """
        if self._irefer is None:
            self._irefer = self._get_value("irefer")
        return self._irefer

    @irefer.setter
    def irefer(self, irefer):
        if isinstance(irefer, str):
            raise AttributeError("Value must be numeric.")
        self._irefer = irefer

    @property
    def murefer(self) -> None:
        """
        Set `_murefer` internally if no initial value. <br>
            Will return `_murefer` if it's already been set.
        """
        if self._murefer is None:
            self._murefer = self._get_value("murefer")
        return self._murefer

    @murefer.setter
    def murefer(self, murefer):
        if isinstance(murefer, str):
            raise AttributeError("Value must be numeric.")
        self._murefer = murefer


class McaMeasurement:
    """
    Class for MCA-related measurements.

    Properties:
        dtc_factors: Deadtime correction factor.
        raw_scaler_in_window: Counts in window (ROI).
        mcas: MCA frames.
    """

    def __init__(self, get_value: Callable[[ScanData | str], Any]):

        self._get_value = get_value

        self._dtc_factors = None
        self._mcas = None
        self._raw_scaler_in_window = None

    @property
    def dtc_factors(self) -> None:
        """
        Set `_dtc_factors` internally if no initial value. <br>
            Will return `_dtc_factors` if it's already been set.
        """
        if self._dtc_factors is None:
            self._dtc_factors = self._get_value("dtc_factors")
        return self._dtc_factors

    @dtc_factors.setter
    def dtc_factors(self, dtc_factors):
        if isinstance(dtc_factors, str):
            raise AttributeError("Value must be numeric.")
        self._dtc_factors = dtc_factors

    @property
    def raw_scaler_in_window(self) -> None:
        """
        Set `_raw_scaler_in_window` internally if no initial value. <br>
            Will return `_raw_scaler_in_window` if it's already been set.
        """
        if self._raw_scaler_in_window is None:
            self._raw_scaler_in_window = self._get_value("raw_scaler_in_window")
        return self._raw_scaler_in_window

    @raw_scaler_in_window.setter
    def raw_scaler_in_window(self, raw_scaler_in_window):
        if isinstance(raw_scaler_in_window, str):
            raise AttributeError("Value must be numeric.")
        self._raw_scaler_in_window = raw_scaler_in_window

    @property
    def mcas(self) -> None:
        """
        Set `_mcas` internally if no initial value. <br>
            Will return `_mcas` if it's already been set.
        """
        if self._mcas is None:
            self._mcas = self._get_value("mcas")
        return self._mcas

    @mcas.setter
    def mcas(self, mcas):
        if isinstance(mcas, str):
            raise AttributeError("Value must be numeric.")
        self._mcas = mcas


class XasMeta:
    """
    Grouping all "metadata" together for now.<br>

    Properties:
        path: Path(s) to file (not editable).
        repetition_files: Repetition file paths (not editable).
        start_time: Scan start time(s) (not editable).
        end_time: Scan end time(s) (not editable).
        symbol: Absorbing atom.
        ref_symbol: Reference absorbing atom.
        edge: Absorption edge.
        ref_edge: Reference absorption edge.
    """

    def __init__(self, get_value: Callable[[ScanMeta | ElementMeta | str], Any]):
        self._get_value = get_value

        self._path = None
        self._repetition_files = None
        self._start_time = None
        self._end_time = None

        self._symbol = None
        self._ref_symbol = None
        self._edge = None
        self._ref_edge = None

    @property
    def path(self) -> None:
        """
        Set `_path` internally if no initial value. <br>
            Will return `_path` if it's already been set.
        """
        if self._path is None:
            self._path = self._get_value("path")
        return self._path

    @path.setter
    def path(self, path):
        if path is not None:
            raise ValueError("Can't change path value.")
        self._path = path

    @property
    def repetition_files(self) -> None:
        """
        Set `_repetition_files` internally if no initial value. <br>
            Will return `_repetition_files` if it's already been set.
        """
        if self._repetition_files is None:
            self._repetition_files = self._get_value("repetition_files")
        return self._repetition_files

    @repetition_files.setter
    def repetition_files(self, repetition_files):
        if repetition_files is not None:
            raise ValueError("Can't change repetition_files value.")
        self._repetition_files = repetition_files

    @property
    def start_time(self) -> None:
        """
        Set `_start_time` internally if no initial value. <br>
            Will return `_start_time` if it's already been set.
        """
        if self._start_time is None:
            self._start_time = self._get_value("start_time")
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        if start_time is not None:
            raise ValueError("Can't change start_time value.")
        self._start_time = start_time

    @property
    def end_time(self) -> None:
        """
        Set `_end_time` internally if no initial value. <br>
            Will return `_end_time` if it's already been set.
        """
        if self._end_time is None:
            self._end_time = self._get_value("end_time")
        return self._end_time

    @end_time.setter
    def end_time(self, end_time):
        if end_time is not None:
            raise ValueError("Can't change end_time value.")
        self._end_time = end_time

    @property
    def symbol(self) -> None:
        """
        Set `_symbol` internally if no initial value. <br>
            Will return `_symbol` if it's already been set.
        """
        if self._symbol is None:
            self._symbol = self._get_value("symbol")
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        if isinstance(symbol, str) or symbol is None:
            self._symbol = symbol
        else:
            raise ValueError("symbol must be str.")

    @property
    def ref_symbol(self) -> None:
        """
        Set `_ref_symbol` internally if no initial value. <br>
            Will return `_ref_symbol` if it's already been set.
        """
        if self._ref_symbol is None:
            self._ref_symbol = self._get_value("ref_symbol")
        return self._ref_symbol

    @ref_symbol.setter
    def ref_symbol(self, ref_symbol):
        if isinstance(ref_symbol, str) or ref_symbol is None:
            self._ref_symbol = ref_symbol
        else:
            raise ValueError("ref_symbol must be str.")

    @property
    def edge(self) -> None:
        """
        Set `_edge` internally if no initial value. <br>
            Will return `_edge` if it's already been set.
        """
        if self._edge is None:
            self._edge = self._get_value("edge")
        return self._edge

    @edge.setter
    def edge(self, edge):
        if isinstance(edge, str) or edge is None:
            self._edge = edge
        else:
            raise ValueError("edge must be str.")

    @property
    def ref_edge(self) -> None:
        """
        Set `_ref_edge` internally if no initial value. <br>
            Will return `_ref_edge` if it's already been set.
        """
        if self._ref_edge is None:
            self._ref_edge = self._get_value("ref_edge")
        return self._ref_edge

    @ref_edge.setter
    def ref_edge(self, ref_edge):
        if isinstance(ref_edge, str) or ref_edge is None:
            self._ref_edge = ref_edge
        else:
            raise ValueError("ref_edge must be str.")
