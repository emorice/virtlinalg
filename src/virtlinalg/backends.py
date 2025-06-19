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

    def unwrap(self, vla_matrices: M) -> T:
        """
        Unwrap backend array from matching subtype of VLA Matrices
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
