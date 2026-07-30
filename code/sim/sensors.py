"""Noisy encoder, IMU, and 2D LiDAR models matching course NPZ fields."""

from __future__ import annotations

import numpy as np

from world import lidar_sim

LIDAR_ANGLE_MIN = -2.356194490192345
LIDAR_ANGLE_MAX = 2.356194490192345
LIDAR_ANGLE_INC = 0.004363323129985824
LIDAR_N_BEAMS = 1081
LIDAR_RANGE_MIN = 0.1
LIDAR_RANGE_MAX = 30.0
LIDAR_BODY_OFFSET = np.array([0.18, 0.005], dtype=np.float64)

METER_PER_TIC = 0.0022


class SensorSuite:
    def __init__(
        self,
        rng: np.random.Generator,
        lidar_noise_std: float = 0.015,
        enc_slip_std: float = 0.03,
        gyro_bias: float = 0.005,
        gyro_noise_std: float = 0.02,
        accel_noise_std: float = 0.05,
    ):
        self.rng = rng
        self.lidar_noise_std = lidar_noise_std
        self.enc_slip_std = enc_slip_std
        self.gyro_bias = gyro_bias
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self._angles = LIDAR_ANGLE_MIN + np.arange(LIDAR_N_BEAMS) * LIDAR_ANGLE_INC

    def scan_lidar(self, pose: np.ndarray, walls: np.ndarray) -> np.ndarray:
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        origin = np.array([x, y]) + R @ LIDAR_BODY_OFFSET

        dir_body = np.column_stack([np.cos(self._angles), np.sin(self._angles)])
        directions = dir_body @ R.T
        d = lidar_sim(origin, directions, walls)
        noise = self.rng.normal(0.0, self.lidar_noise_std, size=LIDAR_N_BEAMS)
        ranges = np.where(np.isfinite(d), d + noise, LIDAR_RANGE_MAX + 1.0)
        return ranges

    def encoder_counts(self, v_l: float, v_r: float, dt: float) -> np.ndarray:
        """
        Per-interval tick counts for [FR, FL, RR, RL], matching course layout
        used in main.encoder2vel (avg of indices 0&2 = right, 1&3 = left).
        """
        # True arc lengths, then wheel slip scale noise.
        slip_r = 1.0 + self.rng.normal(0.0, self.enc_slip_std)
        slip_l = 1.0 + self.rng.normal(0.0, self.enc_slip_std)
        dist_r = max(0.0, v_r * dt * slip_r)
        dist_l = max(0.0, v_l * dt * slip_l)
        ticks_r = int(np.clip(round(dist_r / METER_PER_TIC), -32768, 32767))
        ticks_l = int(np.clip(round(dist_l / METER_PER_TIC), -32768, 32767))
        # Duplicate front/rear like a 4-encoder differential platform.
        return np.array([ticks_r, ticks_l, ticks_r, ticks_l], dtype=np.int16)

    def imu_sample(self, w: float) -> tuple[np.ndarray, np.ndarray]:
        """Return angular_velocity (3,), linear_acceleration (3,) in body frame."""
        wz = w + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise_std)
        angular = np.array(
            [
                self.rng.normal(0.0, self.gyro_noise_std),
                self.rng.normal(0.0, self.gyro_noise_std),
                wz,
            ]
        )
        # Rough accel noise with gravity on z in "g" units.
        ax = self.rng.normal(0.0, self.accel_noise_std)
        ay = self.rng.normal(0.0, self.accel_noise_std)
        az = 1.0 + self.rng.normal(0.0, self.accel_noise_std)
        linear = np.array([ax, ay, az])
        return angular, linear
