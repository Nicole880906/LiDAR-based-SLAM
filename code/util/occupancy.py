from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bresenham2D(sx, sy, ex, ey):
    """Bresenham's ray tracing algorithm in 2D."""
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx

    if dy == 0:
        q = np.zeros((dx + 1, 1))
    else:
        q = np.append(
            0,
            np.greater_equal(
                np.diff(
                    np.mod(
                        np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy),
                        dx,
                    )
                ),
                0,
            ),
        )
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x, y))


def build_occupancy_grid(trajectory, lidar_scans):
    """
    Ray-cast LiDAR hits into a log-odds occupancy grid using the robot trajectory.
    Returns (log_odds_grid, grid_info).
    """
    grid = {}
    grid["res"] = np.array([0.05, 0.05])
    grid["min"] = np.array([-5.0, -5.0])
    grid["max"] = np.array([10.0, 10.0])
    grid["size"] = np.ceil((grid["max"] - grid["min"]) / grid["res"]).astype(int)
    is_even = grid["size"] % 2 == 0
    grid["size"][is_even] += 1
    log_odds = np.zeros(grid["size"])

    OCCUPIED = 0.8
    FREE = -0.4

    for i in range(0, len(trajectory), 10):
        robot_x, robot_y, robot_theta = trajectory[i]
        lidar_scan = lidar_scans[i]
        lidar_x = robot_x + (
            lidar_scan[:, 0] * np.cos(robot_theta) - lidar_scan[:, 1] * np.sin(robot_theta)
        )
        lidar_y = robot_y + (
            lidar_scan[:, 0] * np.sin(robot_theta) + lidar_scan[:, 1] * np.cos(robot_theta)
        )

        start_grid = np.floor(
            (np.array([robot_x, robot_y]) - grid["min"]) / grid["res"]
        ).astype(int)

        for wx, wy in zip(lidar_x, lidar_y):
            end_grid = np.floor((np.array([wx, wy]) - grid["min"]) / grid["res"]).astype(int)
            ray = bresenham2D(
                start_grid[0], start_grid[1], end_grid[0], end_grid[1]
            ).astype(int)

            free_i = ray[0, :-1]
            free_j = ray[1, :-1]
            valid_free = (
                (free_i >= 0)
                & (free_i < grid["size"][0])
                & (free_j >= 0)
                & (free_j < grid["size"][1])
            )
            log_odds[free_i[valid_free], free_j[valid_free]] += FREE
            if (0 <= end_grid[0] < grid["size"][0]) and (0 <= end_grid[1] < grid["size"][1]):
                log_odds[end_grid[0], end_grid[1]] += OCCUPIED

    return log_odds, grid


def draw_maps(trajectory, lidar, results_dir, type="Before", name=""):
    """Build occupancy grid, save PNG under results_dir, and show the figure."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_odds_map, mapinfo = build_occupancy_grid(trajectory, lidar)

    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [mapinfo["min"][0], mapinfo["max"][0], mapinfo["min"][1], mapinfo["max"][1]]
    ax.imshow(log_odds_map.T > 0, origin="lower", extent=extent, cmap="binary")
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label="SLAM Path",
        color="#EB4C4C",
        linestyle="-",
    )
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        color="green",
        marker="o",
        s=100,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        color="#FF8C00",
        marker="X",
        s=100,
        label="Goal",
        zorder=5,
    )
    ax.set_title(f"Occupancy Grid Map ({type} Optimization)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    prefix = f"{name}_" if name else ""
    out_path = results_dir / f"{prefix}occupancy_{type.lower()}_opt.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()
    plt.close(fig)
    return mapinfo
