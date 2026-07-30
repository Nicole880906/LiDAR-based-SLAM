"""Export simulated streams to NPZ files inside an env folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sensors import (
    LIDAR_ANGLE_INC,
    LIDAR_ANGLE_MAX,
    LIDAR_ANGLE_MIN,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
)


def export_dataset(
    out_dir: Path,
    encoder_counts: np.ndarray,
    encoder_stamps: np.ndarray,
    lidar_ranges: np.ndarray,
    lidar_stamps: np.ndarray,
    imu_angular: np.ndarray,
    imu_linear: np.ndarray,
    imu_stamps: np.ndarray,
    gt_poses: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "Encoders.npz",
        counts=encoder_counts.astype(np.int16),
        time_stamps=encoder_stamps.astype(np.float64),
    )
    np.savez(
        out_dir / "Hokuyo.npz",
        angle_min=np.float64(LIDAR_ANGLE_MIN),
        angle_max=np.float64(LIDAR_ANGLE_MAX),
        angle_increment=np.array([[LIDAR_ANGLE_INC]], dtype=np.float64),
        range_min=np.float64(LIDAR_RANGE_MIN),
        range_max=np.int64(int(LIDAR_RANGE_MAX)),
        ranges=lidar_ranges.astype(np.float64),
        time_stamps=lidar_stamps.astype(np.float64),
    )
    np.savez(
        out_dir / "Imu.npz",
        angular_velocity=imu_angular.astype(np.float64),
        linear_acceleration=imu_linear.astype(np.float64),
        time_stamps=imu_stamps.astype(np.float64),
    )

    if gt_poses is not None:
        np.savez(
            out_dir / "GroundTruth.npz",
            poses=gt_poses.astype(np.float64),
        )

    print(f"Wrote sensor logs to {out_dir.resolve()}")
