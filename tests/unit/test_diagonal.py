"""
Efficient diagonal matrices
"""

import pytest

import numpy as np
import virtlinalg as vla
import virtlinalg.np as vnp

def test_diagonal():
    """
    Can create a diagonal and multiply by it on both sides
    """

    diag_values = vnp.wrap_vectors(np.array([1, 2]))

    diagonal = vla.diagonal(diag_values)

    operand = vnp.wrap(np.array([[0, 1], [2, 3]]))

    result_left = diagonal @ operand

    # Acts on rows
    assert np.array_equal(
            vnp.unwrap(result_left),
            [[0, 1], [4, 6]]
            )

    result_right = operand @ diagonal

    # Acts on columns
    assert np.array_equal(
            vnp.unwrap(result_right),
            [[0, 2], [2, 6]]
            )

def test_invert_diagonal():
    """
    Can invert a diagonal matrix
    """
    diag_values = vnp.wrap_vectors(np.array([1, 2]))

    diagonal = vla.diagonal(diag_values)

    inv_diagonal = diagonal.inv()

    operand = vnp.wrap(np.array([[0, 1], [2, 3]]))

    result_left = inv_diagonal @ operand

    # Acts on rows
    assert np.allclose(
            vnp.unwrap(result_left),
            [[0, 1], [1, 3/2]]
            )

def test_refuses_bad_shapes():
    """
    Cannot create a matrix out of the wrong vector shape
    """
    # (1, n), bad
    vectors = vnp.wrap(np.array([[0, 1]]))

    with pytest.raises(ValueError):
        vla.diagonal(vectors)

    # (n, n), bad
    vectors2 = vnp.wrap(np.array([[0, 1], [2, 3]]))

    with pytest.raises(ValueError):
        vla.diagonal(vectors2)
