"""
Shared backend test cases
"""

import numpy as np

def transpose(backend) -> None:
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

backend_cases = [
        transpose,
        ]
