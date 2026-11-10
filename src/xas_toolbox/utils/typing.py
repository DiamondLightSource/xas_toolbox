from typing import Any, TypedDict

from numpy import complex128, dtype, float64, int_, ndarray, uint

# AliasTypes for 1d and nd arrays of float/ints.


_Array1dFloat = ndarray[tuple[Any], dtype[float64]]
_Array1dInt = ndarray[tuple[Any], dtype[uint | int_]]
_Array1dComplex = ndarray[tuple[Any], dtype[complex128]]

_Array1d = _Array1dFloat | _Array1dInt | _Array1dComplex

_ArrayNdFloat = ndarray[tuple[Any, ...], dtype[float64]]
_ArrayNdInt = ndarray[tuple[Any, ...], dtype[uint | int_]]
_ArrayNdComplex = ndarray[tuple[Any, ...], dtype[complex128]]

_ArrayNd = _ArrayNdFloat | _ArrayNdInt | _ArrayNdComplex


class _XYdict(TypedDict):
    x: _Array1d
    y: _ArrayNd
