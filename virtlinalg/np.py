"""
NumPy backend for VLA
"""

import numpy as np
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
        if isinstance(other, NumpyMatrices):
            return NumpyMatrices(self._np_matrices @ other._np_matrices)
        return NotImplemented

    def __rmatmul__(self, other: 'NumpyMatrices') -> 'NumpyMatrices':
        if isinstance(other, NumpyMatrices):
            return NumpyMatrices(other._np_matrices @ self._np_matrices)
        return NotImplemented

    def __add__(self, other: 'NumpyMatrices') -> 'NumpyMatrices':
        if isinstance(other, NumpyMatrices):
            return NumpyMatrices(self._np_matrices + other._np_matrices)
        return NotImplemented

    def inv(self) -> 'NumpyMatrices':
        return NumpyMatrices(np.linalg.inv(self._np_matrices))

    def right_eye(self) -> 'NumpyMatrices':
        return NumpyMatrices(np.eye(self._np_matrices.shape[-1]))

    @property
    def n_rows(self) -> int:
        return self._np_matrices.shape[-2]

    @property
    def n_cols(self) -> int:
        return self._np_matrices.shape[-1]

    def __neg__(self) -> 'NumpyMatrices':
        return NumpyMatrices(-self._np_matrices)

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

def wrap_vectors(np_vectors: npt.NDArray) -> NumpyMatrices:
    """
    Wrap batches of NumPy vectors into VLA matrices
    """
    return NumpyMatrices(np_vectors[..., None])


def unwrap_vectors(vla_matrices: NumpyMatrices) -> npt.NDArray:
    """
    Unwrap batches of NumPy vectors from VLA Matrices
    """
    return np.squeeze(vla_matrices.unwrap(), axis=-1)
