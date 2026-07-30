"""Helpers for data_sim/envN experiment folders."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ENV_RE = re.compile(r"^env(\d+)$")


def list_env_indices(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    idxs: list[int] = []
    for p in root.iterdir():
        m = ENV_RE.match(p.name)
        if p.is_dir() and m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def latest_env_dir(root: Path) -> Path | None:
    idxs = list_env_indices(root)
    if not idxs:
        return None
    return root / f"env{idxs[-1]}"


def _migrate_loose_files(root: Path) -> None:
    """If old flat files exist and no env* yet, move them into env1."""
    if list_env_indices(root):
        return
    loose = [
        p
        for p in root.iterdir()
        if p.is_file()
        and (
            p.suffix in {".npz", ".png"}
            or p.name.startswith(("Encoders", "Hokuyo", "Imu", "GroundTruth", "path", "preview"))
        )
    ]
    if not loose:
        return
    env1 = root / "env1"
    env1.mkdir(parents=True, exist_ok=True)
    for p in loose:
        shutil.move(str(p), str(env1 / p.name))
    print(f"Moved {len(loose)} existing files into {env1}")


def next_env_dir(root: Path) -> Path:
    """
    Create and return the next data_sim/envN folder.
    Inspects existing env* names and uses max+1 (or env1).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _migrate_loose_files(root)
    idxs = list_env_indices(root)
    n = (max(idxs) + 1) if idxs else 1
    env = root / f"env{n}"
    env.mkdir(parents=True, exist_ok=False)
    print(f"Created experiment folder: {env.resolve()}")
    return env
