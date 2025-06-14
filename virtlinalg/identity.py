"""
Identity application
"""

from .matrices import Matrices, VirtualMaps

class Identity[M: Matrices](VirtualMaps[M]):
    """
    Identity map of a given rank
    """
    def __init__(self, rank: int):
        self._rank = rank

    def __matmul__(self, other: M) -> M:
        return other

    def __rmatmul__(self, other: M) -> M:
        return other

    def inv(self):
        return self

def identity(rank: int) -> Identity:
    """
    Identity map of a given rank
    """
    return Identity(rank)
