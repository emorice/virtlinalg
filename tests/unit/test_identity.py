"""
Identity linear applications
"""

import numpy as np
import virtlinalg as vla
import virtlinalg.np as vnp

def test_identity() -> None:
    """
    Applying identity changes nothing
    """
    vla_vectors = vnp.wrap_vectors(np.array([1, 2, 3]))
    identity = vla.identity(3)

    vla_result = identity @ vla_vectors

    assert vla_result is vla_vectors
