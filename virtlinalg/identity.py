"""
Identity application
"""

from .matrices import Matrices, Maps

class Identity[M: Matrices](Maps[M]):
    """
    Identity map of a given rank
    """
    def __init__(self, rank: int):
        self._rank = rank

    def __matmul__(self, other: M) -> M:
        return other

def identity(rank: int) -> Identity:
    """
    Identity map of a given rank
    """
    return Identity(rank)
