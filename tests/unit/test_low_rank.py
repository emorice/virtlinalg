"""
Low Rank matrices
"""

import numpy as np
import virtlinalg as vla
import virtlinalg.np as vnp

def test_low_rank_product() -> None:
    """
    Can multiply a low-rank and a vector
    """

    vec1 = vnp.wrap(np.array([[1, 2]]).T)
    vec2 = vnp.wrap(np.array([[3, 4]]).T)
    vec3 = vnp.wrap(np.array([[5, 6]]).T)

    lora = vla.low_rank_product(vec1, vec2.T)

    result = lora @ vec3

    # 3*5=15 + 4*6=24 = 39, 39*2=78
    assert np.array_equal(vnp.unwrap(result), np.array([[39, 78]]).T)

def test_low_rank_update() -> None:
    """
    Can multiply a low-rank update and a vector
    """

    vec1 = vnp.wrap(np.array([[1, 2]]).T)

    mat1 = vnp.wrap(np.array([[3, 4], [5, 6]]))
    vec2 = vnp.wrap(np.array([[7, 8]]).T)
    mat2 = vnp.wrap(np.array([[9]]))
    vec3 = vnp.wrap(np.array([[10, 11]]).T)

    update = vla.low_rank_update(mat1, vec2, mat2, vec3.T)

    result = update @ vec1

    # base: [1*3+2*4 = 11, 1*5+2*6 = 17]
    # update: 10+2*11 = 32, *9 = 288, *[7, 8] = [2016, 2304]
    # total: [2027, 2321]
    assert np.array_equal(vnp.unwrap(result), np.array([[2027, 2321]]).T)
