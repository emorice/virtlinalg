"""
Torch backend
"""

import numpy.typing as npt

from .backends import BackendNotFoundError
try:
    import torch
except ModuleNotFoundError as e:
    raise BackendNotFoundError from e


from . import Matrices

class _TorchMatrices(Matrices):
    """
    Adapter for Torch types
    """
    def __init__(self, torch_matrices):
        self._torch_matrices = torch_matrices

    @property
    def T(self) -> '_TorchMatrices':
        return _TorchMatrices(
                torch.transpose(self._torch_matrices, -1, -2)
                )

    def unwrap(self) -> torch.Tensor:
        """
        Return wrapped tensor
        """
        return self._torch_matrices

    def __matmul__(self, other: '_TorchMatrices') -> '_TorchMatrices':
        if isinstance(other, _TorchMatrices):
            # Same as @ operator but the function form is better documented
            return _TorchMatrices(torch.matmul(
                self._torch_matrices, other._torch_matrices
                ))
        return NotImplemented

    def __add__(self, other: '_TorchMatrices') -> '_TorchMatrices':
        if isinstance(other, _TorchMatrices):
            return _TorchMatrices(torch.add(
                self._torch_matrices, other._torch_matrices
                ))
        return NotImplemented

    def __neg__(self) -> '_TorchMatrices':
        return _TorchMatrices(-self._torch_matrices)

    def right_eye(self) -> '_TorchMatrices':
        """
        Conformable identity matrix on right side

        This copies the dtype of the original matrices.
        """
        # Concerning dtypes: from quick testing on 2.7.1, adding or multiplying
        # element-wise does automatic conversions, but matrix multiplication
        # errors if you even multiply a float and a double. I couldn't find
        # where the latter behavior is documented.

        return _TorchMatrices(torch.eye(
            self._torch_matrices.shape[-1],
            dtype=self._torch_matrices.dtype,
            ))

    @property
    def n_rows(self) -> int:
        return self._torch_matrices.shape[-2]

    @property
    def n_cols(self) -> int:
        return self._torch_matrices.shape[-1]

    def inv(self) -> '_TorchMatrices':
        """
        Matrix inverse

        Contrarily to regular torch, integer matrices are accepted and
        converted.
        """
        # Trick. This should leave all floating points dtypes unchanged but
        # convert int matrices to the default floating point type
        floating = 1. * self._torch_matrices
        # pylint: disable=not-callable # FP
        return _TorchMatrices(torch.linalg.inv(floating))

def wrap(torch_matrices: torch.Tensor) -> _TorchMatrices:
    """
    Wrap Torch tensor into VLA matrices
    """
    return _TorchMatrices(torch_matrices)

def wrap_vectors(torch_vectors: torch.Tensor) -> _TorchMatrices:
    """
    Wrap batches of Torch vectors into VLA matrices
    """
    return _TorchMatrices(torch.unsqueeze(torch_vectors, dim=-1))

def unwrap(vla_matrices: _TorchMatrices) -> torch.Tensor:
    """
    Unwrap Torch tensor from VLA Matrices
    """
    return vla_matrices.unwrap()

def unwrap_vectors(vla_matrices: _TorchMatrices) -> torch.Tensor:
    """
    Unwrap batches of Torch vectors from VLA Matrices
    """
    return torch.squeeze(vla_matrices.unwrap(), dim=-1)

def from_numpy(np_array: npt.NDArray) -> torch.Tensor:
    """
    Import from NumPy
    """
    return torch.from_numpy(np_array)

def to_numpy(torch_tensor: torch.Tensor) -> npt.NDArray:
    """
    Export to NumPy
    """
    return torch_tensor.numpy()
