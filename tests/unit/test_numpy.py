"""
NumPy backend
"""

import numpy as np

import virtlinalg as vla
import virtlinalg.np

def test_wrap_unwrap() -> None:
    """
    Can wrap and unwrap a numpy array
    """
    np_matrices = np.array([[0., 1.], [2., 3.]])

    vla_matrices = vla.np.wrap(np_matrices)

    assert isinstance(vla_matrices, vla.np.NumpyMatrices)

    np_matrices_back = vla.np.unwrap(vla_matrices)

    assert np_matrices_back is np_matrices

def test_transpose() -> None:
    """
    Can transpose matrices
    """
