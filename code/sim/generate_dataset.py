#!/usr/bin/env python3
"""Generate a course-compatible simulated LiDAR SLAM dataset (Phase 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from draw_path import draw_robot_path
from env_dirs import next_env_dir
from export_npz import export_dataset
from robot import DiffDriveRobot
from sensors import SensorSuite
from world import make_env, robot_path


def path_length(waypoints: np.ndarray) -> float:
    if len(waypoints) < 2:
        return 0.0
    d = np.diff(waypoints, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def duration_for_path(waypoints: np.ndarray, speed: float = 0.35, margin: float = 1.4) -> float:
    """Estimate sim time needed to finish the path (with slack for turns)."""
    return max(30.0, path_length(waypoints) / max(speed, 1e-3) * margin)


def simulate(
    waypoints: np.ndarray,
    duration: float = 60.0,
    dt: float = 0.01,
    enc_hz: float = 40.0,
    lidar_hz: float = 40.0,
    imu_hz: float = 100.0,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    walls = make_env()
    dx, dy = waypoints[1] - waypoints[0]
    theta0 = float(np.arctan2(dy, dx))
    robot = DiffDriveRobot(x=waypoints[0, 0], y=waypoints[0, 1], theta=theta0)
    sensors = SensorSuite(rng)

    enc_period = 1.0 / enc_hz
    lidar_period = 1.0 / lidar_hz
    imu_period = 1.0 / imu_hz
    next_enc = 0.0
    next_lidar = 0.0
    next_imu = 0.0

    enc_counts, enc_t = [], []
    lidar_ranges, lidar_t = [], []
    imu_w, imu_a, imu_t = [], [], []
    gt_poses, gt_t = [], []

    wp_idx = 1
    t = 0.0
    steps = int(duration / dt)
    finished_hold = 0

    for _ in range(steps):
        target = waypoints[min(wp_idx, len(waypoints) - 1)]
        if np.hypot(target[0] - robot.pose[0], target[1] - robot.pose[1]) < 0.35:
            if wp_idx < len(waypoints) - 1:
                wp_idx += 1
            else:
                finished_hold += 1
                if finished_hold > int(1.0 / dt):
                    break

        robot.track_waypoint(target)
        robot.step(dt)
        v_l, v_r = robot.wheel_speeds()

        if t + 1e-12 >= next_imu:
            ang, lin = sensors.imu_sample(robot.w)
            imu_w.append(ang)
            imu_a.append(lin)
            imu_t.append(t)
            next_imu += imu_period

        if t + 1e-12 >= next_enc:
            enc_counts.append(sensors.encoder_counts(v_l, v_r, enc_period))
            enc_t.append(t)
            next_enc += enc_period

        if t + 1e-12 >= next_lidar:
            lidar_ranges.append(sensors.scan_lidar(robot.pose, walls))
            lidar_t.append(t)
            gt_poses.append(robot.pose.copy())
            gt_t.append(t)
            next_lidar += lidar_period

        t += dt

    return {
        "encoder_counts": np.column_stack(enc_counts) if enc_counts else np.zeros((4, 0), np.int16),
        "encoder_stamps": np.asarray(enc_t, dtype=np.float64),
        "lidar_ranges": np.column_stack(lidar_ranges) if lidar_ranges else np.zeros((1081, 0)),
        "lidar_stamps": np.asarray(lidar_t, dtype=np.float64),
        "imu_angular": np.column_stack(imu_w) if imu_w else np.zeros((3, 0)),
        "imu_linear": np.column_stack(imu_a) if imu_a else np.zeros((3, 0)),
        "imu_stamps": np.asarray(imu_t, dtype=np.float64),
        "gt_poses": np.column_stack([np.asarray(gt_t), np.asarray(gt_poses)])
        if gt_poses
        else np.zeros((0, 4)),
        "walls": walls,
    }


def plot_preview(result: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    walls = result["walls"]
    gt = result["gt_poses"]
    fig, ax = plt.subplots(figsize=(7, 7))
    for x1, y1, x2, y2 in walls:
        ax.plot([x1, x2], [y1, y2], color="k", lw=2)
    if gt.size:
        ax.plot(gt[:, 1], gt[:, 2], color="#EB4C4C", lw=1.5, label="ground truth")
        ax.scatter(gt[0, 1], gt[0, 2], c="green", s=60, zorder=5, label="start")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Simulated loop corridor")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved preview to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 2D LiDAR SLAM data simulator")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="max sim seconds (default: auto from path length)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--default-path",
        action="store_true",
        help="use built-in corridor loop instead of drawing",
    )
    parser.add_argument(
        "--path-file",
        type=Path,
        default=None,
        help="load waypoints from a saved .npz (skips drawing)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    out_dir = next_env_dir(Path(__file__).resolve().parent.parent.parent / "data_sim")
    path_save = out_dir / "path.npz"

    walls = make_env()

    if args.path_file is not None:
        data = np.load(args.path_file)
        waypoints = data["waypoints"]
        print(f"Loaded path from {args.path_file} ({len(waypoints)} waypoints)")
    elif args.default_path:
        waypoints = robot_path()
        print(f"Using default loop path ({len(waypoints)} waypoints)")
    else:
        print("Opening map — draw a path, then press Enter or d when done.")
        waypoints = draw_robot_path(walls, save_path=path_save)
        print(f"Using drawn path ({len(waypoints)} waypoints)")

    length = path_length(waypoints)
    duration = args.duration if args.duration is not None else duration_for_path(waypoints)
    print(
        f"Path length ≈ {length:.1f} m → simulating up to {duration:.1f}s "
        f"(seed={args.seed})"
    )
    result = simulate(waypoints=waypoints, duration=duration, seed=args.seed)

    # Warn if the robot did not finish the drawn path.
    if result["gt_poses"].size:
        end = result["gt_poses"][-1, 1:3]
        goal = waypoints[-1]
        leftover = float(np.hypot(end[0] - goal[0], end[1] - goal[1]))
        if leftover > 0.6:
            print(
                f"Warning: run ended {leftover:.2f} m from the last waypoint. "
                f"Increase --duration (e.g. --duration {duration * 1.5:.0f}) "
                "or draw a clearer free-space path."
            )
    export_dataset(
        out_dir=out_dir,
        encoder_counts=result["encoder_counts"],
        encoder_stamps=result["encoder_stamps"],
        lidar_ranges=result["lidar_ranges"],
        lidar_stamps=result["lidar_stamps"],
        imu_angular=result["imu_angular"],
        imu_linear=result["imu_linear"],
        imu_stamps=result["imu_stamps"],
        gt_poses=result["gt_poses"],
    )
    print(
        "shapes:",
        f"enc={result['encoder_counts'].shape}",
        f"lidar={result['lidar_ranges'].shape}",
        f"imu={result['imu_angular'].shape}",
    )
    if not args.no_plot:
        plot_preview(result, out_dir / "preview.png")

    print(f"All outputs for this run are in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
