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

def wrap(torch_matrices: torch.Tensor) -> _TorchMatrices:
    """
    Wrap Torch tensor into VLA matrices
    """
    return _TorchMatrices(torch_matrices)

def unwrap(vla_matrices: _TorchMatrices) -> torch.Tensor:
    """
    Unwrap Torch tensor from VLA Matrices
    """
    return vla_matrices.unwrap()

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
