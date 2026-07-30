"""Interactive path drawing on the simulated map."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def densify_path(waypoints: np.ndarray, spacing: float = 0.35) -> np.ndarray:
    """Insert points along segments so waypoint tracking stays smooth."""
    if len(waypoints) < 2:
        return waypoints.copy()
    pts = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(1, int(np.ceil(dist / spacing)))
        for k in range(1, n + 1):
            t = k / n
            pts.append((1 - t) * a + t * b)
    return np.asarray(pts, dtype=np.float64)


def draw_robot_path(
    walls: np.ndarray,
    outer: float = 8.0,
    save_path: Path | None = None,
) -> np.ndarray:
    """
    Show the map and let the user click a robot path.

    Controls
    --------
    Left click  : add waypoint
    Right click : undo last waypoint
    Enter / d   : finish (need >= 2 points)
    c           : clear all points
    Escape      : cancel (raises SystemExit)
    """
    half = outer / 2.0
    points: list[list[float]] = []

    fig, ax = plt.subplots(figsize=(8, 8))
    for x1, y1, x2, y2 in walls:
        ax.plot([x1, x2], [y1, y2], color="k", lw=2)

    (path_line,) = ax.plot([], [], "-o", color="#EB4C4C", lw=2, ms=6, label="drawn path")
    start_pt = ax.scatter([], [], c="green", s=80, zorder=5, label="start")

    ax.set_aspect("equal")
    ax.set_xlim(-half - 0.5, half + 0.5)
    ax.set_ylim(-half - 0.5, half + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title(
        "Draw path: left-click add · right-click undo · c clear · Enter/d done"
    )

    def _redraw() -> None:
        if not points:
            path_line.set_data([], [])
            start_pt.set_offsets(np.empty((0, 2)))
        else:
            arr = np.asarray(points)
            path_line.set_data(arr[:, 0], arr[:, 1])
            start_pt.set_offsets(arr[:1])
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            points.append([float(event.xdata), float(event.ydata)])
            _redraw()
        elif event.button == 3 and points:
            points.pop()
            _redraw()

    def on_key(event) -> None:
        key = (event.key or "").lower()
        if key in ("enter", "d"):
            if len(points) < 2:
                title.set_text("Need at least 2 points — keep clicking")
                fig.canvas.draw_idle()
                return
            plt.close(fig)
        elif key == "c":
            points.clear()
            title.set_text(
                "Draw path: left-click add · right-click undo · c clear · Enter/d done"
            )
            _redraw()
        elif key == "escape":
            plt.close(fig)
            raise SystemExit("Path drawing cancelled.")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if len(points) < 2:
        raise SystemExit("Path drawing cancelled or too few points.")

    waypoints = densify_path(np.asarray(points, dtype=np.float64))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, waypoints=waypoints, clicks=np.asarray(points, dtype=np.float64))
        print(f"Saved drawn path to {save_path.resolve()}")
    return waypoints
