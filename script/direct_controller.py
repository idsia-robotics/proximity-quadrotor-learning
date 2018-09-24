#!/usr/bin/env python

from __future__ import division

import math

import numpy as np
from tf.transformations import (quaternion_conjugate,
                                quaternion_multiply)

F = 2.83


def rotate(v, q, inverse=False):
    """rotate vector v from world to body frame (with quaternion q)"""
    cq = quaternion_conjugate(q)
    if inverse:
        z = quaternion_multiply(q, v)
        return quaternion_multiply(z, cq)
    else:
        z = quaternion_multiply(v, q)
        return quaternion_multiply(cq, z)


def cmd_from_acc(acc, q):
    t = [acc[0] / F, acc[1] / F, 0, 0]
    return rotate(t, q)[:2]


def cmd_from_angular_speed(omega):
    return min(max(omega / 1.75, -1), 1)


def clamp(xs, bss):
    return [max(min(x, bs[1]), bs[0]) for x, bs in zip(xs, bss)]


def target_yaw_to_observe(observer_point, target_point):
    d = np.array(target_point) - np.array(observer_point)
    return np.arctan2(d[1], d[0])


class Controller(object):
    """docstring for Controller"""

    def new_controller(self, position_head, yaw_head, velocity_drone, velocity_head=[0, 0, 0], distance=1.5, delta_altitude=0.0, delay=0.1,
                       tau=0.5, eta=1.0, rotation_tau=0.5, max_speed=1.5, max_ang_speed=2.0, max_acc=1.0, F=2.83):
        position_target = np.array(position_head) - np.array([np.cos(yaw_head) * distance, np.sin(yaw_head) * distance, delta_altitude])
        des_velocity_drone = (np.array(position_target) - delay * np.array(velocity_drone)) / eta + np.array(velocity_head)
        cmd_linear_z = des_velocity_drone[2]
        s_des = np.linalg.norm(des_velocity_drone[:2])
        if s_des > max_speed:
            des_velocity_drone = des_velocity_drone / s_des * max_speed
        des_horizontal_acceleration_drone = (des_velocity_drone[:2] - np.array(velocity_drone)[:2]) / tau
        des_horizontal_acceleration_drone = np.array(clamp(des_horizontal_acceleration_drone, ((-max_acc, max_acc), (-max_acc, max_acc))))
        cmd_linear_x, cmd_linear_y = des_horizontal_acceleration_drone / F
        target_yaw = np.arctan2(position_head[1], position_head[0])
        if target_yaw > math.pi:
            target_yaw = target_yaw - 2 * math.pi
        if target_yaw < -math.pi:
            target_yaw = target_yaw + 2 * math.pi
        v_yaw = target_yaw / rotation_tau
        if abs(v_yaw) > max_ang_speed:
            v_yaw = v_yaw / abs(v_yaw) * max_ang_speed

        return cmd_linear_x, cmd_linear_y, cmd_linear_z, cmd_from_angular_speed(v_yaw)
