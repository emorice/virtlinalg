"""
Diagonal maps
"""

from .matrices import Matrices, VirtualMaps

class _Diagonal[M: Matrices](VirtualMaps[M]):
    """
    Diagonal maps
    """
    def __init__(self, diagonals: M) -> None:
        if diagonals.n_cols != 1:
            raise ValueError('Expected one column per slice, got '
                + f' {diagonals.n_cols}')
        self._diagonals = diagonals

    def __matmul__(self, other: M) -> M:
        """
        Application self @ other, mutliplies other row-wise
        """
        # Since _diagonals is (n, 1), element-wise multiplication broadcasts it
        # by repeting the column, which is what we need here
        return self._diagonals * other

    def __rmatmul__(self, other: M) -> M:
        """
        Application other @ self, mutliplies other column-wise
        """
        # Same logic as __mul__ except we transpose to get broadcasting in the
        # other direction
        return self._diagonals.T * other

    def inv(self) -> '_Diagonal[M]':
        return _Diagonal(1. / self._diagonals)


def diagonal[M: Matrices](diagonals: M) -> _Diagonal[M]:
    """
    Create diagonal linear maps out of the given diagonals

    Arguments:
        diagonals: must be a stack of column (n x 1) vectors, each of which will
            represent a diagonal of a map in the resulting stack.
    """
    return _Diagonal(diagonals)
