"""
LRU can integrate with other features
"""

import numpy as np
import virtlinalg as vla
import virtlinalg.np as vnp

def test_identity_lru() -> None:
    """
    Can do a low-rank update of an identity linear map
    """
    ident = vla.identity(3)
    np_vector = np.array([1, 2, 3])
    vector = vnp.wrap_vectors(np_vector)

    update = vla.low_rank_update(ident, vector, vector.T)

    assert np.array_equal(
            vnp.unwrap(update @ vnp.wrap(np.eye(3))),
            np.eye(3) + np_vector[:, None] @ np_vector[None, :]
            )
