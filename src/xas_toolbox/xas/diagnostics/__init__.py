"""
Submodule for xas data diagnostic tools e.g. noise estimation and outlier flagging.
"""

from .outlier_detection import (
    detect_outlier_scans,
    gesd_outlier_detection,
    normal_outlier_detection,
    remove_edge_jump,
)

__all__ = [
    "gesd_outlier_detection",
    "normal_outlier_detection",
    "remove_edge_jump",
    "detect_outlier_scans",
]
