"""2D occupancy world built from line-segment walls."""

from __future__ import annotations

import numpy as np


def box_obs(cx: float, cy: float, w: float, h: float) -> list[list[float]]:
    """rectangle obstacle centered at (cx, cy)."""
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    return [
        [x0, y0, x1, y0],
        [x1, y0, x1, y1],
        [x1, y1, x0, y1],
        [x0, y1, x0, y0],
    ]


def lshape_obs(
    x: float,
    y: float,
    arm_x: float,
    arm_y: float,
    thickness: float = 0.12,
) -> list[list[float]]:
    """
    L made of two thin rectangles meeting at (x, y).
    arm_x / arm_y are signed lengths along +x / +y from the corner.
    """
    segs: list[list[float]] = []
    # Horizontal arm
    hx0, hx1 = sorted([x, x + arm_x])
    hy0, hy1 = y - thickness / 2, y + thickness / 2
    segs += [
        [hx0, hy0, hx1, hy0],
        [hx1, hy0, hx1, hy1],
        [hx1, hy1, hx0, hy1],
        [hx0, hy1, hx0, hy0],
    ]
    # Vertical arm
    vx0, vx1 = x - thickness / 2, x + thickness / 2
    vy0, vy1 = sorted([y, y + arm_y])
    segs += [
        [vx0, vy0, vx1, vy0],
        [vx1, vy0, vx1, vy1],
        [vx1, vy1, vx0, vy1],
        [vx0, vy1, vx0, vy0],
    ]
    return segs


def all_obs(o: float, i: float) -> list[list[float]]:
    """
    Add all obstacles in the corridor
    """
    obstacles: list[list[float]] = []

    obstacles += box_obs(o - 0.35, -2.2, 0.55, 0.9)
    obstacles += box_obs(i + 0.28, 0.6, 0.45, 0.55)
    obstacles += box_obs(o - 0.45, 2.4, 0.35, 0.35)
    obstacles += [[3.35, -0.8, 3.85, -0.2]]

    obstacles += lshape_obs(-1.8, o - 0.15, 0.9, -0.55)
    obstacles += box_obs(0.4, i + 0.25, 1.1, 0.35)
    obstacles += box_obs(2.1, o - 0.4, 0.25, 0.65)
    obstacles += box_obs(-2.6, o - 0.35, 0.7, 0.5)

    obstacles += box_obs(-(o - 0.4), 1.5, 0.4, 0.4)
    obstacles += box_obs(-(o - 0.4), 0.7, 0.4, 0.4)
    obstacles += lshape_obs(-(i + 0.1), -1.2, -0.7, 0.65)
    obstacles += [[-3.7, -2.6, -3.2, -2.0]]
    obstacles += box_obs(-(i + 0.3), 2.5, 0.5, 0.3)

    obstacles += box_obs(-1.2, -(o - 0.35), 0.3, 0.55)
    obstacles += box_obs(-0.4, -(o - 0.55), 0.3, 0.75)
    obstacles += box_obs(0.5, -(o - 0.35), 0.3, 0.55)
    obstacles += lshape_obs(1.6, -(i + 0.1), 0.8, -0.5)
    obstacles += box_obs(2.7, -(o - 0.3), 0.6, 0.4)
    obstacles += [[-2.4, -3.55, -1.9, -3.15]]

    return obstacles


def make_env(
    outer: float = 8.0,
    inner: float = 4.0,
    corridor: float = 1.6,
) -> np.ndarray:
    """
    Build a square loop (outer box with an inner hole).
    Returns wall segments as (N, 4) array: [x1, y1, x2, y2].
    """
    o = outer / 2.0
    i = inner / 2.0
    
    assert (o - i) >= corridor / 2.0

    outer_walls = [
        [-o, -o, o, -o],
        [o, -o, o, o],
        [o, o, -o, o],
        [-o, o, -o, -o],
    ]
    inner_walls = [
        [-i, -i, i, -i],
        [i, -i, i, i],
        [i, i, -i, i],
        [-i, i, -i, -i],
    ]
    obstacles = all_obs(o, i)
    return np.asarray(outer_walls + inner_walls + obstacles, dtype=np.float64)


def robot_path(outer: float = 8.0, inner: float = 4.0, n_per_side: int = 8) -> np.ndarray:
    """Centerline waypoints that drive once around the corridor loop."""
    o = outer / 2.0
    i = inner / 2.0
    r = 0.5 * (o + i)
    # Square path at radius r, starting on +x side going CCW.
    sides = [
        np.column_stack([np.full(n_per_side, r), np.linspace(-r, r, n_per_side, endpoint=False)]),
        np.column_stack([np.linspace(r, -r, n_per_side, endpoint=False), np.full(n_per_side, r)]),
        np.column_stack([np.full(n_per_side, -r), np.linspace(r, -r, n_per_side, endpoint=False)]),
        np.column_stack([np.linspace(-r, r, n_per_side, endpoint=False), np.full(n_per_side, -r)]),
    ]
    pts = np.vstack(sides)
    # Close the loop for a second pass (helps proximity loop-closure demos).
    return np.vstack([pts, pts, pts[:1]])


def lidar_sim(
    origin: np.ndarray,
    directions: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """
    Vectorized ray–segment intersections.

    origin: (2,)
    directions: (N, 2) unit vectors
    segments: (M, 4) as x1,y1,x2,y2
    returns: (N,) distances; np.inf if no hit
    """
    ox, oy = origin
    dx = directions[:, 0][:, None]  # (N,1)
    dy = directions[:, 1][:, None]

    x1 = segments[:, 0][None, :]
    y1 = segments[:, 1][None, :]
    sx = (segments[:, 2] - segments[:, 0])[None, :]
    sy = (segments[:, 3] - segments[:, 1])[None, :]

    denom = dx * sy - dy * sx
    safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    qx = x1 - ox
    qy = y1 - oy
    t = (qx * sy - qy * sx) / safe
    u = (qx * dy - qy * dx) / safe

    hits = (np.abs(denom) > 1e-12) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
    t_masked = np.where(hits, t, np.inf)
    return np.min(t_masked, axis=1)