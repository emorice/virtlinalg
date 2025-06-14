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

    update = vla.low_rank_update(mat1, vec2, vec3.T, mat2)

    result = update @ vec1

    # base: [1*3+2*4 = 11, 1*5+2*6 = 17]
    # update: 10+2*11 = 32, *9 = 288, *[7, 8] = [2016, 2304]
    # total: [2027, 2321]
    assert np.array_equal(vnp.unwrap(result), np.array([[2027, 2321]]).T)

def test_optional_center() -> None:
    """
    Can omit the center of the update
    """
    base = vnp.wrap(np.array([[1, 2], [3, 4]]))
    left = vnp.wrap(np.array([[5, 6]]).T)
    right_t = vnp.wrap(np.array([[7, 8]]).T)

    center = vnp.wrap(np.array([[1]]))

    with_center = vla.low_rank_update(base, left, right_t.T, center)
    wout_center = vla.low_rank_update(base, left, right_t.T)

    eye = vnp.wrap(np.eye(2))

    assert np.array_equal(
            vnp.unwrap(with_center @ eye),
            vnp.unwrap(wout_center @ eye),
            )

    assert np.array_equal(
            vnp.unwrap(vla.inv(with_center) @ eye),
            vnp.unwrap(vla.inv(wout_center) @ eye),
            )

def test_inverse_low_rank_update() -> None:
    """
    Can correctly compute a low-rank update inverse
    """
    base = vnp.wrap(np.array([[3, 4], [5, 6]]))
    left = vnp.wrap(np.array([[7, 8]]).T)
    center = vnp.wrap(np.array([[9]]))
    right_t = vnp.wrap(np.array([[10, 11]]).T)

    update = vla.low_rank_update(base, left, right_t.T, center)
    dense_update = base + left @ center @ right_t.T

    result = vla.inv(update) @ dense_update

    assert np.allclose(vnp.unwrap(result), np.array([[1., 0.], [0., 1.]]))
