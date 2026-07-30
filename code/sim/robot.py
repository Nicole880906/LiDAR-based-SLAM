"""Differential-drive robot with simple waypoint tracking."""

from __future__ import annotations

import numpy as np


class DiffDriveRobot:
    def __init__(
        self,
        x: float = 3.0,
        y: float = 0.0,
        theta: float = np.pi / 2,
        wheel_base: float = 0.35,
        max_v: float = 0.45,
        max_w: float = 1.0,
    ):
        self.pose = np.array([x, y, theta], dtype=np.float64)
        self.wheel_base = wheel_base
        self.max_v = max_v
        self.max_w = max_w
        self.v = 0.0
        self.w = 0.0

    def track_waypoint(self, target: np.ndarray, kp_ang: float = 2.5) -> None:
        dx = target[0] - self.pose[0]
        dy = target[1] - self.pose[1]
        dist = np.hypot(dx, dy)
        desired = np.arctan2(dy, dx)
        heading_err = np.arctan2(np.sin(desired - self.pose[2]), np.cos(desired - self.pose[2]))

        self.w = float(np.clip(kp_ang * heading_err, -self.max_w, self.max_w))
        # Slow down while turning hard.
        turn_factor = max(0.15, 1.0 - abs(heading_err) / (np.pi / 2))
        self.v = float(np.clip(0.6 * dist, 0.05, self.max_v) * turn_factor)

    def step(self, dt: float) -> None:
        x, y, th = self.pose
        self.pose[0] = x + self.v * np.cos(th) * dt
        self.pose[1] = y + self.v * np.sin(th) * dt
        self.pose[2] = th + self.w * dt

    def wheel_speeds(self) -> tuple[float, float]:
        """Left/right linear wheel speeds (m/s) from body v, w."""
        v_r = self.v + 0.5 * self.w * self.wheel_base
        v_l = self.v - 0.5 * self.w * self.wheel_base
        return v_l, v_r
