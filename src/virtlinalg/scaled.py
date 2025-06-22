"""
Scaled maps
"""

from .matrices import Matrices, VirtualMaps, Maps

class _Scaled[M: Matrices](VirtualMaps[M]):
    """
    Scaled maps
    """
    def __init__(self, scale: M, base: Maps[M]) -> None:
        if scale.n_rows != 1:
            raise ValueError('Expected one row per slice, got '
                + f' {scale.n_rows}')
        if scale.n_cols != 1:
            raise ValueError('Expected one column per slice, got '
                + f' {scale.n_cols}')
        self._scale = scale
        self._base = base

    def __matmul__(self, other: M) -> M:
        """
        Application self @ other
        """
        return self._scale * (self._base @ other)

    def __rmatmul__(self, other: M) -> M:
        """
        Application other @ self
        """
        return self._scale * (other @ self._base)

    def inv(self) -> '_Scaled[M]':
        return _Scaled(1. / self._scale, self._base.inv())


def scaled[M: Matrices](scale: M, base: Maps[M]) -> _Scaled[M]:
    """
    Scale the base map by the given scalar

    Arguments:
        scale: must be a stack of (1 x 1) matrices, each of which will
            represent a scale for a map in the resulting stack.
    """
    return _Scaled(scale, base)
