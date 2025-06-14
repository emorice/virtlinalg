"""
Low Rank matrices
"""

from .matrices import Matrices, Maps
from .identity import identity

class _LowRankProduct[M: Matrices](Maps[M]):
    """
    Low Rank product
    """
    def __init__(self, left: M, right: M):
        self._left = left
        self._right = right

    def __matmul__(self, other: M) -> M:
        return self._left @ (self._right @ other)

    def inv(self):
        raise ValueError('Low-rank matrices are not meant to be inverted')

class _LowRankUpdate[M: Matrices](Maps[M]):
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

    def inv(self) -> 'LowRankUpdate':
        # This is were the woodbury comes in
        inv_base = self._base.inv()
        inv_base_left = inv_base @ self._left
        capacitance = (
                self._left.right_eye() +
                # contract the large dim first
                (self._right @ inv_base_left) @ self._center
                )

        return LowRankUpdate(
                base=inv_base,
                left=inv_base_left,
                center=- self._center @ capacitance.inv(),
                right=self._right @ inv_base
                )

def low_rank_product[M: Matrices](left: M, right: M) -> LowRankProduct[M]:
    """
    Virtual `left @ right` product that does not actually perform the contraction
    """
    return _LowRankProduct(left, right)

def low_rank_update[M: Matrices](base: M, left: M, right: M,
                                 center: M | None = None
                                 ) -> LowRankUpdate[M]:
    """
    Virtual `base + left @ center @ right` product that does not explictly
    compute the matrix.
    """
    center_map = identity(left.n_cols) if center is None else center
    return _LowRankUpdate(base, left, right, center_map)
