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

class LowRankUpdate[M: Matrices](Maps[M]):
    """
    Low Rank Update `base + left @ center @ right`
    """
    def __init__(self, base: M, left: M, center: M, right: M):
        self._base = base
        self._left = left
        self._center = center
        self._right = right

    def __matmul__(self, other: M) -> M:
        return (
            self._base @ other
            +
            self._left @ (self._center @ (self._right @ other))
            )

def low_rank_product[M: Matrices](left: M, right: M) -> LowRankProduct[M]:
    """
    Virtual `left @ right` product that does not actually perform the contraction
    """
    return LowRankProduct(left, right)

def low_rank_update[M: Matrices](base: M, left: M, center: M, right: M
                                 ) -> LowRankUpdate[M]:
    """
    Virtual `base + left @ center @ right` product that does not explictly
    compute the matrix.
    """
    return LowRankUpdate(base, left, center, right)
