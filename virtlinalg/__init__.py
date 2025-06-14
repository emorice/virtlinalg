"""
Virtual Linear Algebra
"""
from .matrices import Matrices, Maps
from .identity import identity
from .low_rank import low_rank_product, low_rank_update

__all__ = [
        "Matrices", "Maps", "identity", "low_rank_product", "low_rank_update",
        "inv"
        ]

def inv[M: Matrices](maps: Maps[M]) -> Maps[M]:
    """
    Attempt to invert maps
    """
    return maps.inv()
