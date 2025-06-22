"""
Shared backend test cases
"""

from typing import Protocol

import numpy as np
import virtlinalg as vla

class BackendCase(Protocol): # pylint: disable=too-few-public-methods
    """
    Type of generic backend case function
    """
    def __call__[T, M: vla.Matrices](self, backend: vla.Backend[T, M]) -> None:
        ...

backend_cases: list[BackendCase] = []

def case(fun: BackendCase) -> BackendCase:
    """
    Decorator to add a generic backend case
    """
    backend_cases.append(fun)
    return fun

def _mat[T, M: vla.Matrices](backend: vla.Backend[T, M], array_like) -> M:
    """
    Helper to convert array like to numpy then backend then vla
    """
    return backend.wrap(backend.from_numpy(np.array(array_like)))

def _unmat[T, M: vla.Matrices](backend: vla.Backend[T, M], matrices: M):
    """
    Reverse of _mat
    """
    return backend.to_numpy(backend.unwrap(matrices))

@case
def transpose[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can transpose matrices
    """
    vla_matrices = backend.wrap(
        backend.from_numpy(
            np.array([[0., 1.], [2., 3.]])
            )
        )

    vla_results = vla_matrices.T

    assert np.array_equal(
            backend.to_numpy(
                backend.unwrap(vla_results)
                ),
            np.array([[0., 2.], [1., 3.]])
            )

@case
def transpose_3d[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can transpose stack of matrices
    """
    # 2 3-by-5 matrices
    vla_matrices = backend.wrap(
        backend.from_numpy(
            np.arange(30).reshape(2, 3, 5)
            )
        )

    vla_results = vla_matrices.T

    # should now be 2 5-by-3
    assert backend.to_numpy(
                backend.unwrap(vla_results)
                ).shape == (2, 5, 3)

@case
def apply[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can use Matrices as Maps
    """
    matrices = backend.wrap(
            backend.from_numpy(np.array([[0, 1], [2, 3]]))
            )
    vectors = backend.wrap_vectors(
            backend.from_numpy(np.array([4, 5]))
            )

    result = matrices @ vectors

    assert np.array_equal(
            backend.to_numpy(
                backend.unwrap_vectors(result)
                ),
            np.array([5, 23])
            )

@case
def add[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can add matrices with infix
    """
    matrices = _mat(backend, [[0, 1], [2, 3]])
    other = _mat(backend, [[4, 5], [6, 7]])

    result = matrices + other

    assert np.array_equal(
            _unmat(backend, result),
            np.array([[4, 6], [8, 10]])
            )

@case
def use_eye[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can create a conformable identity matrix
    """
    matrices = _mat(backend, [[0, 1], [2, 3]])

    eye = matrices.right_eye()

    result = matrices @ eye

    assert np.array_equal(
            _unmat(backend, matrices),
            _unmat(backend, result)
            )

@case
def shape[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can access the n_rows/n_cols shape
    """
    # 1, 2, 3
    matrices = _mat(backend, [[
        [0, 1, 2],
        [3, 4, 5],
        ]])

    assert matrices.n_rows == 2
    assert matrices.n_cols == 3

@case
def neg[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can take unary minus
    """
    matrices = _mat(backend, [[0, 1], [2, 3]])

    result = - matrices

    assert np.array_equal(
            _unmat(backend, result),
            np.array([[0, -1], [-2, -3]])
            )

@case
def inv[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can take matrix inverse
    """
    matrices = _mat(backend, [[0, 1], [2, 3]])

    result = matrices.inv()

    assert np.allclose(
            _unmat(backend, result),
            np.array([[-3/2, 1/2], [1, 0]])
            )

@case
def mul[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can take element-wise multiplication (Hadamard product)
    """
    matrices =  _mat(backend, [[0, 1], [2, 3]])

    result = matrices * matrices

    assert np.array_equal(
            _unmat(backend, result),
            [[0, 1], [4, 9]]
            )

@case
def truediv[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can take element-wise division with a scalar as numerator
    """
    matrices =  _mat(backend, [[1, 2], [3, 4]])

    result = 1. / matrices

    assert np.allclose(
            _unmat(backend, result),
            [[1, 1/2], [1/3, 1/4]]
            )

@case
def scalar[T, M: vla.Matrices](backend: vla.Backend[T, M]) -> None:
    """
    Can wrap a scalar and use it to scale a matrix
    """
    matrices =  _mat(backend, [[1, 2], [3, 4]])

    scalars = backend.wrap_scalars(
                backend.from_numpy(np.array(5))
                )

    result = scalars * matrices

    assert np.array_equal(
            _unmat(backend, result),
            [[5, 10], [15, 20]]
            )
