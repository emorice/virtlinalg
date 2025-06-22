
"""
Efficient scaled matrices
"""

import pytest

import numpy as np
import virtlinalg as vla
import virtlinalg.np as vnp

def test_scaled():
    """
    Can create a scaled matrix and multiply by it on both sides
    """

    base = vnp.wrap(np.array([[0, 1], [2, 3]]))
    scalar = vnp.wrap_scalars(np.array(4))

    scaled = vla.scaled(scalar, base)

    operand = vnp.wrap(np.array([[5, 6], [7, 8]]))

    result_left = scaled @ operand
    result_right = operand @ scaled

    assert np.array_equal(
            vnp.unwrap(result_left),
            [[28, 32], [124, 144]]
            )
    assert np.array_equal(
            vnp.unwrap(result_right),
            [[48, 92], [64, 124]]
            )

def test_invert_scaled():
    """
    Can invert a scaled matrix
    """
    base = vnp.wrap(np.array([[0, 1], [2, 3]]))
    scalar = vnp.wrap_scalars(np.array(4))

    scaled = vla.scaled(scalar, base)

    inv_scaled = scaled.inv()

    materialized = inv_scaled @ vnp.wrap(np.eye(2))

    assert np.allclose(
            vnp.unwrap(materialized),
            [[-3/8, 1/8], [1/4, 0]]
            )

def test_refuses_bad_shapes():
    """
    Cannot create a matrix out of the wrong vector shape
    """
    base = vnp.wrap(np.array([[0, 1], [2, 3]]))

    # (1, n), bad
    bad_scalar = vnp.wrap(np.array([[0, 1]]))

    with pytest.raises(ValueError):
        vla.scaled(bad_scalar, base)

    # (n, 1), bad
    bad_scalar2 = vnp.wrap(np.array([[0, 1]]).T)

    with pytest.raises(ValueError):
        vla.scaled(bad_scalar2, base)
