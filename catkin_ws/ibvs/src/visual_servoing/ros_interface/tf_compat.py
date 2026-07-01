"""
Drop-in replacement for tf.transformations using scipy.

Provides the same API as ROS1's tf.transformations for quaternion/matrix
operations, but uses scipy.spatial.transform.Rotation internally.

Quaternion convention: [x, y, z, w] (same as ROS).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_matrix(quaternion):
    """Convert quaternion [x, y, z, w] to 4x4 homogeneous rotation matrix.

    Args:
        quaternion: [x, y, z, w] quaternion (list, tuple, or array)

    Returns:
        4x4 numpy array (homogeneous rotation matrix)
    """
    q = np.array(quaternion, dtype=np.float64)
    r = Rotation.from_quat(q)  # scipy uses [x, y, z, w] same as ROS
    mat = np.eye(4)
    mat[:3, :3] = r.as_matrix()
    return mat


def quaternion_from_matrix(matrix):
    """Convert 4x4 rotation matrix to quaternion [x, y, z, w].

    Args:
        matrix: 4x4 numpy array (homogeneous rotation matrix)

    Returns:
        [x, y, z, w] quaternion as numpy array
    """
    r = Rotation.from_matrix(np.array(matrix[:3, :3], dtype=np.float64))
    return r.as_quat()  # returns [x, y, z, w]


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Convert quaternion [x, y, z, w] to Euler angles.

    Args:
        quaternion: [x, y, z, w] quaternion
        axes: Euler angle convention (default 'sxyz' for static XYZ)

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    q = np.array(quaternion, dtype=np.float64)
    r = Rotation.from_quat(q)
    # scipy uses intrinsic rotations with uppercase, extrinsic with lowercase
    # ROS 'sxyz' = static/extrinsic XYZ = scipy 'xyz' (lowercase)
    euler = r.as_euler('xyz', degrees=False)
    return tuple(euler)


def quaternion_from_euler(roll, pitch, yaw, axes='sxyz'):
    """Convert Euler angles to quaternion [x, y, z, w].

    Args:
        roll: Rotation about X axis (radians)
        pitch: Rotation about Y axis (radians)
        yaw: Rotation about Z axis (radians)
        axes: Euler angle convention (default 'sxyz')

    Returns:
        [x, y, z, w] quaternion as numpy array
    """
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_quat()


def quaternion_multiply(q1, q0):
    """Multiply two quaternions [x, y, z, w].

    Args:
        q1: First quaternion [x, y, z, w]
        q0: Second quaternion [x, y, z, w]

    Returns:
        Product quaternion [x, y, z, w] as numpy array
    """
    r1 = Rotation.from_quat(np.array(q1, dtype=np.float64))
    r0 = Rotation.from_quat(np.array(q0, dtype=np.float64))
    return (r1 * r0).as_quat()


def quaternion_inverse(quaternion):
    """Compute inverse of quaternion [x, y, z, w].

    Args:
        quaternion: [x, y, z, w] quaternion

    Returns:
        Inverse quaternion [x, y, z, w] as numpy array
    """
    r = Rotation.from_quat(np.array(quaternion, dtype=np.float64))
    return r.inv().as_quat()
