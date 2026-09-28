"""
Submodule for data-correction related functions.
"""

from .deglitch import median_deglitch
from .outlier_removal import correct_outliers
from .pre_edge import norm_and_flatten, pre_edge, pre_edge_bkg

__all__ = [
    "median_deglitch",
    "pre_edge",
    "pre_edge_bkg",
    "norm_and_flatten",
    "correct_outliers",
]
