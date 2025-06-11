"""
NumPy backend
"""

import numpy as np

import virtlinalg.np as vnp

def test_wrap_unwrap() -> None:
    """
    Can wrap and unwrap a numpy array
    """
    np_matrices = np.array([[0., 1.], [2., 3.]])

    vla_matrices = vnp.wrap(np_matrices)

    assert isinstance(vla_matrices, vnp.NumpyMatrices)

    np_matrices_back = vnp.unwrap(vla_matrices)

    assert np_matrices_back is np_matrices

def test_transpose() -> None:
    """
    Can transpose matrices
    """
    np_matrices = np.array([[0., 1.], [2., 3.]])

    vla_matrices = vnp.wrap(np_matrices)

    vla_results = vla_matrices.T

    assert np.array_equal(
            vnp.unwrap(vla_results),
            np.array([[0., 2.], [1., 3.]])
            )
