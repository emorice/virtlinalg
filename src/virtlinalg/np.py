"""
NumPy backend for VLA
"""

import numpy as np
import numpy.typing as npt

from . import Matrices

class _NumpyMatrices(Matrices):
    """
    Adapter for NumPy types
    """
    def __init__(self, np_matrices: npt.NDArray):
        self._np_matrices = np_matrices

    @property
    def T(self) -> '_NumpyMatrices':
        return _NumpyMatrices(np.swapaxes(self._np_matrices, -1, -2))

    def __matmul__(self, other: '_NumpyMatrices') -> '_NumpyMatrices':
        if isinstance(other, _NumpyMatrices):
            return _NumpyMatrices(self._np_matrices @ other._np_matrices)
        return NotImplemented

    def __rmatmul__(self, other: '_NumpyMatrices') -> '_NumpyMatrices':
        if isinstance(other, _NumpyMatrices):
            return _NumpyMatrices(other._np_matrices @ self._np_matrices)
        return NotImplemented

    def __mul__(self, other: '_NumpyMatrices') -> '_NumpyMatrices':
        if isinstance(other, _NumpyMatrices):
            return _NumpyMatrices(self._np_matrices * other._np_matrices)
        return NotImplemented

    def __add__(self, other: '_NumpyMatrices') -> '_NumpyMatrices':
        if isinstance(other, _NumpyMatrices):
            return _NumpyMatrices(self._np_matrices + other._np_matrices)
        return NotImplemented

    def __rtruediv__(self, other: float) -> '_NumpyMatrices':
        return _NumpyMatrices(other / self._np_matrices)

    def inv(self) -> '_NumpyMatrices':
        return _NumpyMatrices(np.linalg.inv(self._np_matrices))

    def right_eye(self) -> '_NumpyMatrices':
        return _NumpyMatrices(np.eye(self._np_matrices.shape[-1]))

    @property
    def n_rows(self) -> int:
        return self._np_matrices.shape[-2]

    @property
    def n_cols(self) -> int:
        return self._np_matrices.shape[-1]

    def __neg__(self) -> '_NumpyMatrices':
        return _NumpyMatrices(-self._np_matrices)

    def unwrap(self) -> npt.NDArray:
        """
        Return wrapped array
        """
        return self._np_matrices

def wrap(np_matrices: npt.NDArray) -> _NumpyMatrices:
    """
    Wrap NumPy array into VLA matrices
    """
    return _NumpyMatrices(np_matrices)

def unwrap(vla_matrices: _NumpyMatrices) -> npt.NDArray:
    """
    Unwrap NumPy array from VLA Matrices
    """
    return vla_matrices.unwrap()

def wrap_vectors(np_vectors: npt.NDArray) -> _NumpyMatrices:
    """
    Wrap batches of NumPy vectors into VLA matrices
    """
    return _NumpyMatrices(np_vectors[..., None])

def unwrap_vectors(vla_matrices: _NumpyMatrices) -> npt.NDArray:
    """
    Unwrap batches of NumPy vectors from VLA Matrices
    """
    return np.squeeze(vla_matrices.unwrap(), axis=-1)

def wrap_scalars(np_scalars: npt.NDArray) -> _NumpyMatrices:
    """
    Wrap batches of NumPy scalars into VLA matrices
    """
    return _NumpyMatrices(np_scalars[..., None, None])

def unwrap_scalars(vla_matrices: _NumpyMatrices) -> npt.NDArray:
    """
    Unwrap batches of NumPy scalars from VLA Matrices
    """
    return np.squeeze(vla_matrices.unwrap(), axis=(-2, -1))

# Trivial import/export for non-numpy backend interface
def from_numpy(np_array: npt.NDArray) -> npt.NDArray:
    """
    Import from NumPy
    """
    return np_array

def to_numpy(be_matrices: npt.NDArray) -> npt.NDArray:
    """
    Export to NumPy
    """
    return be_matrices
