_edges = ('K','L1','L2','L3','M1','M2','M3','M4','M5')

_lines = ("KA1_LINE","KA2_LINE","KB1_LINE","KB2_LINE","KB3_LINE","KB4_LINE","KB5_LINE","LA1_LINE","LA2_LINE",
          "LB1_LINE","LB2_LINE","LB3_LINE","LB4_LINE","LB5_LINE","LB6_LINE","LB7_LINE","LB9_LINE","LB10_LINE","LB15_LINE",
          "LB17_LINE","LG1_LINE","LG2_LINE","LG3_LINE","LG4_LINE","LG5_LINE","LG6_LINE","LG8_LINE","LE_LINE","LL_LINE",
          "LS_LINE","LT_LINE","LU_LINE","LV_LINE","MA1_LINE","MA2_LINE","MB_LINE","MG_LINE")

_iupac_lines = ("KL3_LINE", "KL2_LINE", "KM3_LINE" ,"KN3_LINE",
"KM2_LINE" ,"KN5_LINE", "KM5_LINE", "L3M5_LINE", "L3M4_LINE", "L2M4_LINE",
"L3N5_LINE" ,"L1M3_LINE", "L1M2_LINE" ,"L3O45_LINE" ,"L3N1_LINE" ,"L3O1_LINE", 
"L1M5_LINE", "L1M4_LINE", "L3N4_LINE", "L2M3_LINE" ,"L2N4_LINE" ,"L1N2_LINE", 
"L1N3_LINE", "L1O3_LINE", "L2N1_LINE" ,"L2O4_LINE" ,"L2O1_LINE", "L2M1_LINE", 
"L3M1_LINE", "L3M3_LINE", "L3M2_LINE", "L3N6_LINE", "L2N6_LINE", "M5N7_LINE", 
"M5N6_LINE", "M4N6_LINE", "M3N5_LINE")

def _index_iupac_lines(edge:str)->list:
    edge_len = len(edge)
    return [l for l in _iupac_lines if l[:edge_len]]

import xraylib
def _get_transition_dict(edge:str)->tuple[list[int], list[str]]:
    rel_lines = _index_iupac_lines(edge)
    out = ([getattr(xraylib, l) for l in rel_lines], rel_lines)
    return out

_transitions = {edge:_get_transition_dict(edge) for edge in ["K", "L1", "L2", "L3", "M"]}

import inspect
import numpy as np
from scipy.interpolate import CubicSpline

def to_density(fn):
    """
    If a function is called with rho, t as density and thickness, calculate
    surface density and use that instead.
    """
    def wrapper(*args, **kwargs):
        b_args = inspect.signature(fn).bind(*args,**kwargs)
        if "rho" in b_args.arguments.keys() and b_args.arguments["rho"] is not None:
            b_args.arguments["surf_dens"] = b_args.arguments["rho"]*b_args.arguments["t"]
        return fn(**b_args.arguments)
    return wrapper
