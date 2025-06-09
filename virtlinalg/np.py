"""
NumPy backend for VLA
"""

import numpy.typing as npt

from . import Matrices

class NumpyMatrices(Matrices):
    """
    Adapter for NumPy types
    """
    def __init__(self, np_matrices: npt.NDArray):
        self._np_matrices = np_matrices

    @property
    def T(self) -> 'NumpyMatrices':
        return NumpyMatrices(self._np_matrices.transpose(-1, -2))

    def __matmul__(self, other: 'NumpyMatrices') -> 'NumpyMatrices':
        return NumpyMatrices(self._np_matrices @ other._np_matrices)

    def unwrap(self) -> npt.NDArray:
        """
        Return wrapped array
        """
        return self._np_matrices

def wrap(np_matrices: npt.NDArray) -> NumpyMatrices:
    """
    Wrap NumPy array into VLA matrices
    """
    return NumpyMatrices(np_matrices)


def unwrap(vla_matrices: NumpyMatrices) -> npt.NDArray:
    """
    Unwrap NumPy array from VLA Matrices
    """
    return vla_matrices.unwrap()
