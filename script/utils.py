import inspect

import cv2
import numpy as np
import tf
from sympy.matrices import Matrix
from tf.transformations import quaternion_from_euler


def isdebugging():
    """
        Returns true if code is being debugged
    Returns:
        Returns true if code is being debugged
    """
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def jpeg2np(image, size=None):
    """
        Converts a jpeg image in a 3d numpy array of RGB pixels and resizes it to the given size (if provided).
      Args:
        image: a compressed BGR jpeg image.
        size: a tuple containing width and height, or None for no resizing.

      Returns:
        img: the raw, resized image as a 3d numpy array of RGB pixels.
    """
    compressed = np.fromstring(image, np.uint8)
    raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)

    return img


def time_conversion_to_nano(sec, nano):
    """
        convert time from ros timestamp to nanosecond timestamp
    Args:
        sec: seconds timestamp
        nano: nanoseconds remainder timestamp

    Returns:
        sum of nanoseconds
    """
    return (sec * 1000 * 1000 * 1000) + nano


def find_nearest(array, value):
    """
        find nearest value in array
    Args:
        array: array of values
        value: reference value

    Returns:
        min index of nearest array's element to value
    """
    return (np.abs(array - value)).argmin()


def rospose2homogmat(p, q):
    """
        Convert rospose Pose to homogeneus matrix
    Args:
        p: position array
        q: rotation quaternion array

    Returns:
        w_t_o: Homogeneous roto-translation matrix
            World
                T
                  object
    """
    w_r_o = np.array(quat2mat(q)).astype(np.float64)
    tempmat = np.hstack((w_r_o, np.expand_dims(p, axis=1)))
    w_t_o = np.vstack((tempmat, [0, 0, 0, 1]))
    return w_t_o


def quat2mat(quat):
    """ Symbolic conversion from quaternion to rotation matrix

    For a unit quaternion

    From: http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    """
    x, y, z, w = quat
    return Matrix([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def quat_to_eul(q):
    """
        Convert quaternion orientation to euler orientation
    Args:
        q: quaternion array

    Returns:
        euler: array of 3-D rotation   [roll, pitch, yaw]
    """
    euler = tf.transformations.euler_from_quaternion(q)  #
    return euler


def change_frame_reference_twist(new_frame_reference, twist):
    """
         Change frame of reference of twist information from World to bebop.

         Args:
             new_frame_reference: pose of the bebop
             twist: ros twist of the drone

         Returns:
             new frame of reference for twist:
                 bebop
                     T
                      twist
     """
    quaternion_new_frame_ref = new_frame_reference[['b_rot_x', 'b_rot_y', 'b_rot_z', 'b_rot_w']].values
    _, _, yaw = quat_to_eul(quaternion_new_frame_ref)
    quaternion_new_frame_ref = quaternion_from_euler(0, 0, yaw)
    w_t_nfr = rospose2homogmat([0, 0, 0], quaternion_new_frame_ref)
    w_t_tw = rospose2homogmat([twist['t_x'], twist['t_y'], 0], [0, 0, 0, 1])
    nfr_t_w = np.linalg.inv(w_t_nfr)
    nfr_t_tw = np.matmul(nfr_t_w, w_t_tw)

    return nfr_t_tw


def change_frame_reference(pose_bebop, pose_head):
    """
        Change frame of reference of pose head from World to bebop.

        Args:
            pose_bebop: pose of the bebop
            pose_head: pose of the head

        Returns:
            the new pose for head:
                bebop
                    T
                     head
    """
    position_bebop = pose_bebop[['b_pos_x', 'b_pos_y', 'b_pos_z']].values
    quaternion_bebop = pose_bebop[['b_rot_x', 'b_rot_y', 'b_rot_z', 'b_rot_w']].values
    _, _, yaw = quat_to_eul(quaternion_bebop)
    quaternion_bebop = quaternion_from_euler(0, 0, yaw)
    position_head = pose_head[['h_pos_x', 'h_pos_y', 'h_pos_z']].values
    quaternion_head = pose_head[['h_rot_x', 'h_rot_y', 'h_rot_z', 'h_rot_w']].values
    w_t_b = rospose2homogmat(position_bebop, quaternion_bebop)
    w_t_h = rospose2homogmat(position_head, quaternion_head)
    b_t_w = np.linalg.inv(w_t_b)
    b_t_h = np.matmul(b_t_w, w_t_h)

    return b_t_h