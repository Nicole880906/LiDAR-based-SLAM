import gtsam
import numpy as np


def encoder2vel(encoder_counts, time):
    meter_per_tic = 0.0022
    r_dis = (encoder_counts[0] + encoder_counts[2]) * meter_per_tic / 2
    l_dis = (encoder_counts[1] + encoder_counts[3]) * meter_per_tic / 2
    time_diff = np.append(np.array([1]), np.diff(time))
    r_vel = r_dis / time_diff
    l_vel = l_dis / time_diff
    vt = (r_vel + l_vel) / 2
    return vt


def yaw2theta(yaw, time):
    dt = np.diff(time)
    dyaw_angle = yaw[:-1] * dt
    yaw_angle = np.zeros_like(yaw)
    yaw_angle[1:] = np.cumsum(dyaw_angle)
    return yaw_angle


def angle2xy(lidar, angle_min, angle_increment):
    angles = (angle_min + np.arange(1081) * angle_increment).flatten()
    mask = (lidar > 0.1) & (lidar < 30.0)
    lidar_x = lidar * np.cos(angles)[:, None]
    lidar_y = lidar * np.sin(angles)[:, None]

    result = []
    for i in range(lidar.shape[1]):
        lidar_col = mask[:, i]
        x = lidar_x[lidar_col, i] + 0.18
        y = lidar_y[lidar_col, i] + 0.005
        result.append(np.column_stack((x, y)))
    return result


def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def mat2pose2(T):
    dx = T[0, 2]
    dy = T[1, 2]
    dtheta = np.arctan2(T[1, 0], T[0, 0])
    return gtsam.Pose2(dx, dy, dtheta)
