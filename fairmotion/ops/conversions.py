# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np

from fairmotion.utils import constants, utils
from scipy.spatial.transform import Rotation

import warnings

"""
Glossary:
p: position (3,)
rad: radians
deg: degrees
A: Axis angle (3,)
E: Euler angle (3,)
Q: Quaternion (4,)
R: Rotation matrix (3,3)
T: Transition matrix (4,4)

Quaternion uses the xyzw order
Rotation matrix matrix is column-wise
"""

"""
TODO:
Euler Angle order correction for Test
"""
###################
# MARK: 
# coded added for input compression
"""
custom "compressed" representation
"""
def R2C(R):
    return batch_auto_reshape(
        R, lambda x: compress_matrices(x), (3, 3), (1, ),
    )

def C2R(R):
    return batch_auto_reshape(
        R, lambda x: decompress_matrices(x), (3, 3), (1, ),
    )

def compress_matrices(matrices):
    """
    Compresses an array of 3x3 matrices into a single 64-bit integer for each matrix.
    Parameters:
    matrices (np.array): Input array of shape (n, 3, 3) containing 3x3 matrices.
    Returns:
    np.array: Array of 64-bit integers representing the compressed matrices.
    """
    # Normalize the values from -1 to 1 to a range of 0 to 127
    normalized_matrices = np.interp(matrices, [-1, 1], [0, 127]).astype(np.uint8)
    # Reshape to have each matrix as a single row of 9 elements
    reshaped_matrices = normalized_matrices.reshape(-1, 9)
    # Convert each element to a 7-bit binary string
    binary_strings = np.vectorize(lambda x: format(x, '07b'))(reshaped_matrices)
    # Concatenate binary strings for each matrix and convert to 64-bit integer
    compressed_matrices = np.array([int(''.join(row), 2) for row in binary_strings])
    return compressed_matrices

def decompress_matrices(compressed_matrices):
    """
    Decompresses an array of 64-bit integers back into an array of 3x3 matrices.
    Parameters:
    compressed_matrices (np.array): Input array of 64-bit integers representing the compressed matrices.
    Returns:
    np.array: Array of 3x3 matrices of floats ranging from -1 to 1.
    """
    # Convert each 64-bit integer to a binary string of length 63 (9 elements of 7 bits each)
    binary_strings = np.vectorize(lambda x: format(x, '063b'))(compressed_matrices)
    # Split each binary string into 9 chunks of 7 bits and convert back to integers
    split_binary = np.array([np.array([int(binary_strings[i][j:j+7], 2) for j in range(0, 63, 7)]) for i in range(len(binary_strings))])
    # Reshape back to 3x3 matrices
    reshaped_matrices = split_binary.reshape(-1, 3, 3)
    # Normalize back to the range of -1 to 1
    decompressed_matrices = np.interp(reshaped_matrices, [0, 127], [-1, 1])
    return decompressed_matrices

# End of code added for input compression
##############

def batch_auto_reshape(x, fn, shape_in, shape_out):
    reshape = x.ndim - len(shape_in) > 1
    xx = x.reshape(-1, *shape_in) if reshape else x
    y = fn(xx)
    return y.reshape(x.shape[: -len(shape_in)] + shape_out) if reshape else y


"""
Angle conversions
"""


def rad2deg(rad):
    """Convert from radians to degrees."""
    return rad * 180.0 / np.pi


def deg2rad(deg):
    """Convert from degrees to radians."""
    return deg * np.pi / 180.0


"""
From A to other representations
"""


def A2A(A):
    """
    The same 3D orientation could be represented by the two different
    axis-angle representatons -- (axis, angle) and (-axis, 2pi - angle) where
    we assume 0 <= angle <= pi. This method forces that the representation of
    orientation strictly uses an angle between 0 and pi.
    """

    def a2a(a):
        angle = np.linalg.norm(a)
        if angle <= constants.EPSILON:
            return a
        if angle > 2 * np.pi:
            angle = angle % 2 * np.pi
            warnings.warn("!!!Angle is larger than 2PI!!!")
        if angle > np.pi:
            return (-a / angle) * (2 * np.pi - angle)
        else:
            return a

    return batch_auto_reshape(
        A, lambda x: utils._apply_fn_agnostic_to_vec_mat(x, a2a), (3,), (3,),
    )


def A2E(A, order="xyz", degrees=False):
    return batch_auto_reshape(
        A,
        lambda x: Rotation.from_rotvec(x).as_euler(order, degrees=degrees),
        (3,),
        (3,),
    )


def A2Q(A):
    return batch_auto_reshape(
        A, lambda x: Rotation.from_rotvec(x).as_quat(), (3,), (4,),
    )


def A2R(A):
    return batch_auto_reshape(
        A, lambda x: Rotation.from_rotvec(x).as_matrix(), (3,), (3, 3),
    )


def A2T(A):
    return batch_auto_reshape(
        A, lambda x: Rp2T(A2R(x), constants.zero_p()), (3,), (4, 4),
    )


def Ax2R(theta):
    """
    Convert (axis) angle along x axis Ax to rotation matrix R
    """
    if isinstance(theta, np.ndarray):
        x = np.zeros((theta.shape + (3,)))
    else:
        x = np.zeros(3)
    x[..., 0] = theta
    return A2R(x)


def Ay2R(theta):
    """
    Convert (axis) angle along y axis Ay to rotation matrix R
    """
    if isinstance(theta, np.ndarray):
        R = np.zeros(theta.shape + (3, 3))
        R[...] = constants.eye_R()
    else:
        R = constants.eye_R()
    c = np.cos(theta)
    s = np.sin(theta)
    R[..., 0, 0] = c
    R[..., 0, 2] = s
    R[..., 2, 0] = -s
    R[..., 2, 2] = c
    return R


def Az2R(theta):
    """
    Convert (axis) angle along z axis Az to rotation matrix R
    """
    if isinstance(theta, np.ndarray):
        R = np.zeros(theta.shape + (3, 3))
        R[...] = constants.eye_R()
    else:
        R = constants.eye_R()
    c = np.cos(theta)
    s = np.sin(theta)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    return R


"""
From R to other representations
"""


def R2A(R):
    return batch_auto_reshape(
        R, lambda x: Rotation.from_matrix(x).as_rotvec(), (3, 3), (3,),
    )


def R2E(R, order="XYZ", degrees=False):
    return batch_auto_reshape(
        R,
        lambda x: Rotation.from_matrix(x).as_euler(order, degrees=degrees),
        (3, 3),
        (3,),
    )


def R2Q(R):
    return batch_auto_reshape(
        R, lambda x: Rotation.from_matrix(x).as_quat(), (3, 3), (4,),
    )


def R2R6D(R):
    return R[..., 0:2]


def R6D2R(R6D):
    R3D = np.cross(R6D[..., 0], R6D[..., 1])
    R = np.concatenate((R6D, np.expand_dims(R3D, axis=-1)), axis=-1)
    return R


def R2R(R):
    """
    This returns valid (corrected) rotation if input
    rotations are invalid. Otherwise returns the same values.
    """
    return batch_auto_reshape(
        R, lambda x: Rotation.from_matrix(x).as_matrix(), (3, 3), (3, 3),
    )


def R2T(R):
    return Rp2T(R, constants.zero_p())


"""
From Q to other representations
"""


def Q2A(Q):
    return batch_auto_reshape(
        Q, lambda x: Rotation.from_quat(x).as_rotvec(), (4,), (3,),
    )


def Q2E(Q, order="xyz", degrees=False):
    return batch_auto_reshape(
        Q,
        lambda x: Rotation.from_quat(x).as_euler(order, degrees=degrees),
        (4,),
        (3,),
    )


def Q2Q(Q, op, xyzw_in=True):
    """
    This returns valid (corrected) rotation if input rotations are invalid.
    Otherwise returns the same values.
    """
    return batch_auto_reshape(
        Q, lambda x: Rotation.from_quat(x).as_quat(), (4,), (4,),
    )


def Q2R(Q):
    return batch_auto_reshape(
        Q, lambda x: Rotation.from_quat(x).as_matrix(), (4,), (3, 3),
    )


def Q2T(Q):
    return batch_auto_reshape(
        Q, lambda x: Rp2T(Q2R(x), constants.zero_p()), (4,), (4, 4),
    )


"""
From T to other representations
"""


def T2p(T):
    _, p = T2Rp(T)
    return p


def T2R(T):
    R, _ = T2Rp(T)
    return R


def T2Rp(T):
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    return R, p


def T2Qp(T):
    R, p = T2Rp(T)
    Q = R2Q(R)
    return Q, p


def Ap2T(A, p):
    return Rp2T(A2R(A), p)


def E2R(theta):
    return Rotation.from_euler("xyz", theta).as_matrix()


def Ep2T(E, p, order="xyz", degrees=False):
    return Rp2T(E2R(E, order, degrees), p)


"""
From some representations to T
"""


def Qp2T(Q, p):
    R = Q2R(Q)
    return Rp2T(R, p)


def Rp2T(R, p):
    input_shape = R.shape[:-2] if R.ndim > 2 else p.shape[:-1]
    R_flat = R.reshape((-1, 3, 3))
    p_flat = p.reshape((-1, 3))
    T = np.zeros((int(np.prod(input_shape)), 4, 4))
    T[...] = constants.eye_T()
    T[..., :3, :3] = R_flat
    T[..., :3, 3] = p_flat
    return T.reshape(list(input_shape) + [4, 4])


def p2T(p):
    return Rp2T(constants.eye_R(), np.array(p))
