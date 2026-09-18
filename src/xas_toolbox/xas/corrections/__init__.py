"""
Submodule for data-correction related functions.
"""

from .deglitch import median_deglitch
from .pre_edge import pre_edge

__all__ = ["pre_edge", "median_deglitch"]
