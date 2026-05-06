from xas_toolbox.io.xas_measurement import XasMeasurement
from .single_file import _load_misc, _load_nxs
from .multi_file import MultipleFileReader
from pathlib import Path
from typing import Literal

def load_single_file(path:str|Path, mode:Literal["fluorescence", "transmission"]=None)\
                                                                    ->XasMeasurement:

    if isinstance(path, str): path = Path(path)
    if path.suffix == ".nxs":
        get_fn, _ = _load_nxs(path)
    else:
        get_fn, _ = _load_misc(path)

    out = XasMeasurement(get_value=get_fn, mode=mode)

    return out

def load_multiple_files(paths:list[str|Path], meta_filter:bool=True,\
         mode:Literal["fluorescence", "transmission"]=None)->XasMeasurement:
    
    paths = [Path(p) for p in paths]
    readers = MultipleFileReader(paths, meta_filter=meta_filter)
    get_fn = readers.get_value

    out = XasMeasurement(get_value=get_fn, mode=mode)

    return out

def read_data(path:str|Path|list[str|Path], meta_filter:bool=True,\
               mode:Literal["fluorescence", "transmission"]=None)->XasMeasurement:
    """
    Read a single file/list of files.

    Arguments:
        path (str | Path | list[str|Path]): File path(s) to data. 
        meta_filter (bool, Optional): If multiple files provided whether to remove
                                      files with different absorbing edges + elements.
        mode (Literal["fluorescence", "transmission"], Optional): Acquisition mode for the scan. If set to `None`
                                        the mode is automatically found.

    Returns:
        out (XasMeasurement): Object with XAS data in.
    """
    if isinstance(path, list):
        return load_multiple_files(path, meta_filter=meta_filter, mode=mode)
    else:
        return load_single_file(path, mode)
