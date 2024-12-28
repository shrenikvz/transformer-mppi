import numpy as np
import csv
from typing import Tuple, List

__all__ = ["make_side_lane", "make_csv_paths"]

def make_side_lane(
    center_path: np.ndarray, 
    lane_width: float = 6.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate left and right lane boundaries from a center path.

    Args:
        center_path: Nx3 array of [x, y, heading] coordinates where heading is in radians
        lane_width: Width of the lane in meters. Default is 6.5m

    Returns:
        Tuple containing:
            - right_lane: Nx2 array of [x, y] coordinates for right boundary
            - left_lane: Nx2 array of [x, y] coordinates for left boundary

    Example:
        >>> center = np.array([[0, 0, 0], [1, 0, np.pi/4]])
        >>> right, left = make_side_lane(center, lane_width=4.0)
    """
    right_lane = []
    left_lane = []
    half_width = lane_width / 2.0

    for (x, y, theta) in center_path:
        # Calculate normal vectors to the path direction
        # Rotate the heading vector by 90 degrees to get the normal
        dx = -half_width * np.sin(theta)  # x-component of normal
        dy = half_width * np.cos(theta)   # y-component of normal

        # Offset the center point by the normal vector
        right_lane.append([x + dx, y + dy])
        left_lane.append([x - dx, y - dy])

    return np.array(right_lane), np.array(left_lane)

def make_csv_paths(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load racing circuit path from a CSV file.

    Args:
        csv_path: Path to the CSV file containing track data.
                 Expected format: each row contains [x, y, heading]
                 where heading is in radians

    Returns:
        Tuple containing:
            - path: Nx3 array of [x, y, heading] coordinates
            - path_x: N-length array of x coordinates
            - path_y: N-length array of y coordinates

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file format is invalid

    Example:
        >>> path, xs, ys = make_csv_paths('track_data.csv')
    """
    path: List[List[float]] = []
    path_x: List[float] = []
    path_y: List[float] = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    x, y, theta = map(float, row)
                    path.append([x, y, theta])
                    path_x.append(x)
                    path_y.append(y)
                except ValueError as e:
                    raise ValueError(f"Invalid row format in CSV: {row}") from e

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return np.array(path), np.array(path_x), np.array(path_y)

