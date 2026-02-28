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

    path = []
    path_x = []
    path_y = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            x, y, theta = map(float, row)
            path.append([x, y, theta])
            path_x.append(x)
            path_y.append(y)

    return np.array(path), np.array(path_x), np.array(path_y)
