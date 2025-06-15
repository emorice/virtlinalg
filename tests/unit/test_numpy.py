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

def test_wrap_unwrap_vectors() -> None:
    """
    Can wrap and unwrap a batch of vectors
    """
    # this should be two vectors:
    np_matrices = np.array([[0, 1], [2, 3]])

    # one matrix, or two columns [0, 2], [1, 3]
    vla_matrices = vnp.wrap(np_matrices)
    # a batch of two vectors, [0, 1] and [2, 3]
    vla_vectors = vnp.wrap_vectors(np_matrices)

    assert isinstance(vla_vectors, vnp.NumpyMatrices)

    vla_matrix = vnp.wrap(np.array([[3, 4], [5, 6]]))

    result_vectors = vla_matrix @ vla_vectors
    result_matrices = vla_matrix @ vla_matrices

    assert np.array_equal(
            vnp.unwrap_vectors(result_vectors),
            np.array([[4, 6], [18, 28]])
            )

    assert np.array_equal(
            vnp.unwrap(result_matrices),
            np.array([[8, 15], [12, 23]])
            )

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

def test_apply() -> None:
    """
    Can use Matrices as Maps
    """
    matrices = vnp.wrap(np.array([[0, 1], [2, 3]]))
    vectors = vnp.wrap_vectors(np.array([4, 5]))

    result = matrices @ vectors

    assert np.array_equal(
            vnp.unwrap_vectors(result),
            np.array([5, 23])
            )

def test_shape() -> None:
    """
    Can access matrix shape
    """

    mat_2d = vnp.wrap(np.arange(6).reshape(2, 3))

    assert (mat_2d.n_rows, mat_2d.n_cols) == (2, 3)

    mat_4d = vnp.wrap(np.arange(120).reshape(4, 5, 2, 3))

    assert (mat_4d.n_rows, mat_4d.n_cols) == (2, 3)
