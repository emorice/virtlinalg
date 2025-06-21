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

    def __add__(self, other: Self) -> Self:
        """
        Matrix addition
        """
        raise NotImplementedError

    def __neg__(self) -> Self:
        """
        Matrix unary minus
        """
        raise NotImplementedError

    def inv(self) -> Self:
        """
        Attempt inverse
        """
        raise NotImplementedError

    def right_eye(self) -> Self:
        """
        Conformable identity matrix on right side

        When applied to a stack of matrices, there is as of now no required
        shape, as long as it broadcasts back correctly for that backend.

        The dtype or equivalent is also undefined for now, as long as it can be
        applied back for that backend.
        """
        raise NotImplementedError

    @property
    def n_rows(self) -> int:
        """
        Number of rows
        """
        raise NotImplementedError

    @property
    def n_cols(self) -> int:
        """
        Number of columns
        """
        raise NotImplementedError

class VirtualMaps[M: Matrices]:
    """
    Virtual Linear Map interface
    """
    def __matmul__(self, other: M) -> M:
        """
        Linear map application
        """
        raise NotImplementedError

    def __rmatmul__(self, other: M) -> M:
        """
        Transposed linear map application
        """
        raise NotImplementedError

    def inv(self) -> Self:
        """
        Attempt inverse
        """
        raise NotImplementedError

# M is automatically a VirtMaps[M], but I don't think the type system can
# express that, so we define the union explictly
type Maps[M] = M | VirtualMaps[M]
