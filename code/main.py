from pathlib import Path
import sys

from icp import icp
from load_data import load_data
from util import (
    angle2xy,
    calculate_distance,
    draw_maps,
    encoder2vel,
    mat2pose2,
    yaw2theta,
)

import gtsam
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
from env_dirs import latest_env_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SIM_ROOT = PROJECT_ROOT / "data_sim"
RESULT_DIR = PROJECT_ROOT / "result"

# Load Data — pick an env folder (None = latest data_sim/envN)
ENV_NAME = None
_env = (DATA_SIM_ROOT / ENV_NAME) if ENV_NAME else latest_env_dir(DATA_SIM_ROOT)
if _env is None or not _env.is_dir():
    raise FileNotFoundError(
        f"No data_sim/env* folder found under {DATA_SIM_ROOT}. "
        "Run: cd code/sim && python3 generate_dataset.py"
    )
DATA_DIR = str(_env)
print(f"Loading sim data from {DATA_DIR}")
encoder, lidar, imu = load_data(DATA_DIR)

encoder_data = encoder["counts"]
encoder_time = encoder["stamps"]
lidar_data = lidar["ranges"]
lidar_time = lidar["stamps"]
imu_data = imu["angular_velocity"][2]
imu_time = imu["stamps"]
# Calculate velocity and yaw angle
vt = encoder2vel(encoder_data, encoder_time)
yaw_angle = yaw2theta(imu_data, imu_time)

# ----------------------------- 1. Encoder and IMU odometry -----------------------------------
x, y, theta = 0.0, 0.0, 0.0
imu_trajectory = [[x, y, theta]]

# Interpolation: align encoder/imu timestamps
unwrapped_theta = np.unwrap(yaw_angle)
yaw_angle_interp = np.interp(encoder_time, imu_time, unwrapped_theta)

for i in range(1, len(encoder_time)):

    dt = encoder_time[i] - encoder_time[i - 1]
    vx = vt[i] * np.cos(yaw_angle_interp[i])
    vy = vt[i] * np.sin(yaw_angle_interp[i])
    x += vx * dt
    y += vy * dt

    imu_trajectory.append([x, y, yaw_angle_interp[i]])
imu_trajectory = np.array(imu_trajectory)  # {World}

# -------------------------------------- 2. ICP -----------------------------------------------
# Convert lidar angles to x,y
lidar_xy = angle2xy(lidar_data, lidar["angle_min"], lidar["angle_increment"])

# Interpolation: align encoder/imu timestamps
unwrapped_theta = np.unwrap(imu_trajectory[:, 2])
robot_x = np.interp(lidar_time, encoder_time, imu_trajectory[:, 0])
robot_y = np.interp(lidar_time, encoder_time, imu_trajectory[:, 1])
robot_theta = np.interp(lidar_time, encoder_time, unwrapped_theta)
robot_pose = np.column_stack((robot_x, robot_y, robot_theta))

# Use ICP to refine trajectory
lidar_trajectory = [robot_pose[0]]

# Transformation matrix. Maps {B} to {W}
init_theta = robot_pose[0][2]
Twb = np.eye(3)
Twb[:2, :2] = np.array(
    [[np.cos(init_theta), -np.sin(init_theta)], [np.sin(init_theta), np.cos(init_theta)]]
)

# ICP refine relative transformation matrix between two steps
relative_T = []
for i in range(len(lidar_time) - 1):
    x1, y1, th1 = robot_pose[i]
    x2, y2, th2 = robot_pose[i + 1]

    dx_world = x2 - x1
    dy_world = y2 - y1
    dx_body = dx_world * np.cos(th1) + dy_world * np.sin(th1)
    dy_body = -dx_world * np.sin(th1) + dy_world * np.cos(th1)
    dth = th2 - th1

    T_org = np.array(
        [[np.cos(dth), -np.sin(dth), dx_body], [np.sin(dth), np.cos(dth), dy_body], [0, 0, 1]]
    )

    updated_T, _ = icp(lidar_xy[i + 1], lidar_xy[i], T_org)
    relative_T.append(updated_T)

    # Update new trajectory
    Twb = Twb @ updated_T
    new_x = Twb[0, 2]
    new_y = Twb[1, 2]
    new_th = np.arctan2(Twb[1, 0], Twb[0, 0])
    lidar_trajectory.append([new_x, new_y, new_th])
lidar_trajectory = np.array(lidar_trajectory)  # {World}

mapinfo = draw_maps(lidar_trajectory, lidar_xy, RESULT_DIR, name=_env.name)

# ------------------------------ 3. Factor graph optimization ---------------------------------
factor_graph = gtsam.NonlinearFactorGraph()
est_pose = gtsam.Values()
odometry_noise = gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.01])
loop_noise = gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.02])
prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.001])

FITNESS_SCORE = 0.1

# Build the factor graph
for i in range(len(lidar_trajectory)):
    print(f"{i} / {len(lidar_trajectory)}")
    key = gtsam.symbol("x", i)
    pose = lidar_trajectory[i]
    est_pose.insert(key, gtsam.Pose2(pose[0], pose[1], pose[2]))

    if i == 0:
        # Prior Factor
        factor_graph.add(
            gtsam.PriorFactorPose2(
                key, gtsam.Pose2(pose[0], pose[1], pose[2]), prior_noise
            )
        )
    else:
        # Odometry Factors
        rel_T = relative_T[i - 1]
        factor_graph.add(
            gtsam.BetweenFactorPose2(
                gtsam.symbol("x", i - 1), key, mat2pose2(rel_T), odometry_noise
            )
        )

    # Fixed-interval loop closure
    if i > 0 and i % 10 == 0:
        prev_idx = i - 10
        update_T_fixed, err = icp(lidar_xy[i], lidar_xy[prev_idx], np.eye(3))

        if err < FITNESS_SCORE:
            factor_graph.add(
                gtsam.BetweenFactorPose2(
                    gtsam.symbol("x", prev_idx), key, mat2pose2(update_T_fixed), loop_noise
                )
            )

    # Proximity-based loop closure
    for j in range(0, i - 150, 10):
        pose_before = lidar_trajectory[j]

        if calculate_distance(pose, pose_before) < 2.0:
            dx_world = pose[0] - pose_before[0]
            dy_world = pose[1] - pose_before[1]
            dx_local = dx_world * np.cos(pose_before[2]) + dy_world * np.sin(pose_before[2])
            dy_local = -dx_world * np.sin(pose_before[2]) + dy_world * np.cos(pose_before[2])
            dtheta = pose[2] - pose_before[2]

            T_guess = np.array(
                [
                    [np.cos(dtheta), -np.sin(dtheta), dx_local],
                    [np.sin(dtheta), np.cos(dtheta), dy_local],
                    [0, 0, 1],
                ]
            )

            update_T_loop, err = icp(lidar_xy[i], lidar_xy[j], T_guess)

            if err < FITNESS_SCORE:
                factor_graph.add(
                    gtsam.BetweenFactorPose2(
                        gtsam.symbol("x", j), key, mat2pose2(update_T_loop), loop_noise
                    )
                )
                break

# Optimize and Reconstruct
optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, est_pose)
result = optimizer.optimize()

pgo_trajectory = []
for i in range(len(lidar_trajectory)):
    p = result.atPose2(gtsam.symbol("x", i))
    pgo_trajectory.append([p.x(), p.y(), p.theta()])
pgo_trajectory = np.array(pgo_trajectory)

mapinfo = draw_maps(pgo_trajectory, lidar_xy, RESULT_DIR, type="After", name=_env.name)
