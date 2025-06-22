"""
Common backend utils
"""

from typing import Protocol
import numpy.typing as npt
from .matrices import Matrices

class Backend[T, M: Matrices](Protocol):
    """
    Interface of backend modules
    """
    def wrap(self, be_matrices: T) -> M:
        """
        Wrap backend arrays into a subtype of VLA matrices
        """

    def wrap_vectors(self, be_vectors: T) -> M:
        """
        Wrap batches of backend vectors into VLA matrices
        """

    def wrap_scalars(self, be_scalars: T) -> M:
        """
        Wrap batches of scalars into VLA matrices
        """

    def unwrap(self, vla_matrices: M) -> T:
        """
        Unwrap backend array from matching subtype of VLA Matrices
        """

    def unwrap_vectors(self, vla_matrices: M) -> T:
        """
        Unwrap batches of backend vectors from VLA Matrices
        """

    def unwrap_scalars(self, vla_matrices: M) -> T:
        """
        Unwrap batches of backend scalars from VLA Matrices
        """

    def from_numpy(self, np_array: npt.NDArray) -> T:
        """
        Import from NumPy
        """

    def to_numpy(self, be_matrices: T) -> npt.NDArray:
        """
        Export to NumPy
        """

class BackendNotFoundError(ModuleNotFoundError):
    """
    Raised by VLA backend modules when loading a backend dependency fails.

    This allows to programmatically determine if loading a VLA backend failed
    because that backend is not installed or because of a bug.
    """
