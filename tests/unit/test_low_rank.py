"""
Low Rank matrices
"""

import numpy as np
import virtlinalg as vla
import virtlinalg.np

def test_product() -> None:
    """
    Can multiply a low-rank and a vector
    """

    vec1 = vla.np.wrap(np.array([[1, 2]]).T)
    vec2 = vla.np.wrap(np.array([[3, 4]]).T)
    vec3 = vla.np.wrap(np.array([[5, 6]]).T)

    lora = vla.low_rank_product(vec1, vec2.T)

    result = lora @ vec3

    # 3*5=15 + 4*6=24 = 39, 39*2=78
    assert np.array_equal(vla.np.unwrap(result), np.array([[39, 78]]).T)
