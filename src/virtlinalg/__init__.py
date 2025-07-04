"""
Virtual Linear Algebra
"""
from .matrices import Matrices, Maps
from .identity import identity
from .diagonal import diagonal
from .scaled import scaled
from .low_rank import low_rank_product, low_rank_update
from .backends import BackendNotFoundError, Backend

__all__ = [
        "Matrices", "Maps", "identity", "low_rank_product", "low_rank_update",
        "inv", "BackendNotFoundError", "Backend", "diagonal", "scaled"
        ]

def inv[M: Matrices](maps: Maps[M]) -> Maps[M]:
    """
    Attempt to invert maps
    """
    return maps.inv()
