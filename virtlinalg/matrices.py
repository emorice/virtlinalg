"""
Matrices interface
"""

from typing import Self

class Matrices:
    """
    Minimal backend matrix interface
    """
    @property
    def T(self) -> Self: # pylint: disable=invalid-name # Conventional
        """
        Transpose
        """
        raise NotImplementedError

    def __matmul__(self, other: Self) -> Self:
        """
        Matrix multiplication
        """
        raise NotImplementedError

class Maps[M: Matrices]:
    """
    Minimal Linear map interface
    """
    def __matmul__(self, other: M) -> M:
        """
        Linear map application
        """
        raise NotImplementedError
