from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np


__all__ = ["make_side_lane", "make_csv_paths"]


def make_side_lane(center_path: np.ndarray, lane_width: float = 6.5) -> Tuple[np.ndarray, np.ndarray]:
    right_lane = []
    left_lane = []
    half_width = lane_width / 2.0

    for (x, y, theta) in center_path:
        dx = -half_width * np.sin(theta)
        dy = half_width * np.cos(theta)
        right_lane.append([x + dx, y + dy])
        left_lane.append([x - dx, y - dy])

    return np.array(right_lane), np.array(left_lane)


def make_csv_paths(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: list[list[float]] = []
    header: list[str] | None = None

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                header = [value.strip().lower() for value in row]

    if not rows:
        raise ValueError(f"No numeric path rows found in {csv_path}")

    numeric = np.asarray(rows, dtype=np.float64)
    path_x = numeric[:, 0]
    path_y = numeric[:, 1]

    if header is not None and any(token in {"theta", "yaw", "heading"} for token in header):
        theta_idx = next(i for i, name in enumerate(header) if name in {"theta", "yaw", "heading"})
        theta = numeric[:, theta_idx]
    elif numeric.shape[1] == 3:
        theta = numeric[:, 2]
    else:
        dx = np.diff(path_x, append=path_x[-1])
        dy = np.diff(path_y, append=path_y[-1])
        if len(dx) > 1:
            dx[-1] = dx[-2]
            dy[-1] = dy[-2]
        theta = np.arctan2(dy, dx)

    path = np.column_stack([path_x, path_y, theta])
    return path, path_x, path_y
