import numpy as np

from src.environment.maps import ObstacleMap

__all__ = ["generate_random_obstacles"]

def generate_random_obstacles(
    obstacle_map: ObstacleMap,
    random_x_range: tuple[float, float],
    random_y_range: tuple[float, float],
    num_circle_obs: int,
    radius_range: tuple[float, float],
    max_iteration: int = 1000,
    seed: int = 42,
):
    """
    Generate and add random circular obstacles to the obstacle map.

    This function attempts to create a specified number of non-overlapping circular obstacles
    within the defined x and y ranges. Each obstacle has a radius within the specified range.
    If the function fails to place an obstacle without overlapping after the maximum number of
    iterations, it raises a RuntimeError.

    Parameters:
        obstacle_map (ObstacleMap): The map to which obstacles are added.
        random_x_range (tuple[float, float]): The (min, max) range for the x-coordinate of obstacle centers.
        random_y_range (tuple[float, float]): The (min, max) range for the y-coordinate of obstacle centers.
        num_circle_obs (int): The number of circular obstacles to generate.
        radius_range (tuple[float, float]): The (min, max) range for obstacle radii.
        max_iteration (int, optional): Maximum attempts to place each obstacle. Defaults to 1000.
        seed (int, optional): Seed for random number generator to ensure reproducibility. Defaults to 42.

    Raises:
        RuntimeError: If unable to place an obstacle without overlap after max_iteration attempts.
    """
    rng = np.random.default_rng(seed)

    for _ in range(num_circle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            # Generate random center coordinates within the specified ranges
            center_x = rng.uniform(*random_x_range)
            center_y = rng.uniform(*random_y_range)
            center = np.array([center_x, center_y])

            # Generate a random radius within the specified range
            radius = rng.uniform(*radius_range)

            # Check for overlap with existing obstacles
            is_overlap = any(
                np.linalg.norm(circle_obs.center - center) <= (circle_obs.radius + radius)
                for circle_obs in obstacle_map.circle_obs_list
            )

            if not is_overlap:
                obstacle_map.add_circle_obstacle(center, radius)
                break  # Successfully placed the obstacle
            num_trial += 1

        if num_trial == max_iteration:
            raise RuntimeError("Cannot generate random obstacles (max iteration reached).")
