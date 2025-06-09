"""
Low Rank matrices
"""

from .matrices import Matrices, Maps

class LowRankProduct[M: Matrices](Maps[M]):
    """
    Low Rank product
    """
    def __init__(self, left: M, right: M):
        self._left = left
        self._right = right

    def __matmul__(self, other: M) -> M:
        return self._left @ (self._right @ other)

def low_rank_product[M: Matrices](left: M, right: M) -> LowRankProduct[M]:
    """
    Virtual `left @ right` product that does not actually perform the contraction
    """
    return LowRankProduct(left, right)
