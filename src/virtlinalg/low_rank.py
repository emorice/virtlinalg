"""
Low Rank matrices
"""

from .matrices import Matrices, Maps, VirtualMaps
from .identity import identity

class _LowRankProduct[M: Matrices](VirtualMaps[M]):
    """
    Low Rank product
    """
    def __init__(self, left: M, right: M):
        self._left = left
        self._right = right

    def __matmul__(self, other: M) -> M:
        return self._left @ (self._right @ other)

    def __rmatmul__(self, other: M) -> M:
        return (other @ self._left) @ self._right

    def inv(self):
        raise ValueError('Low-rank matrices are not meant to be inverted')

class _LowRankUpdate[M: Matrices](VirtualMaps[M]):
    """
    Low Rank Update `base + left @ center @ right`
    """
    def __init__(self, base: Maps[M], left: M, right: M, center: Maps[M]):
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

    def __rmatmul__(self, other: M) -> M:
        return (
            other @ self._base
            +
            ((other @ self._left) @ self._center) @ self._right
            )

    def inv(self) -> '_LowRankUpdate':
        # This is were the woodbury comes in
        inv_base = self._base.inv()
        inv_base_left = inv_base @ self._left
        capacitance = (
                self._left.right_eye() +
                # contract the large dim first
                (self._right @ inv_base_left) @ self._center
                )

        return _LowRankUpdate(
                base=inv_base,
                left=inv_base_left,
                center=- (self._center @ capacitance.inv()),
                right=self._right @ inv_base
                )

def low_rank_product[M: Matrices](left: M, right: M) -> _LowRankProduct[M]:
    """
    Virtual `left @ right` product that does not actually perform the contraction
    """
    return _LowRankProduct(left, right)

def low_rank_update[M: Matrices](base: Maps[M] | M, left: M, right: M,
                                 center: Maps[M] | M | None = None
                                 ) -> _LowRankUpdate[M]:
    """
    Virtual `base + left @ center @ right` product that does not explictly
    compute the matrix.
    """
    center_map = identity(left.n_cols) if center is None else center
    return _LowRankUpdate(base, left, right, center_map)
