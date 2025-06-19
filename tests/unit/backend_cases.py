"""
Shared backend test cases
"""

from typing import Protocol

import numpy as np
import virtlinalg as vla

class BackendCase(Protocol):
    def __call__[T, M: vla.Matrices](self, backend: vla.Backend[T, M]) -> None:
        ...

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

backend_cases: list[BackendCase] = [
        transpose,
        transpose_3d,
        ]
