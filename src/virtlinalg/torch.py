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
