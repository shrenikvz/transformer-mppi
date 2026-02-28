from __future__ import annotations

import numpy as np

from transformer_mppi.environment.maps import ObstacleMap


__all__ = ["generate_random_obstacles"]


def generate_random_obstacles(
    obstacle_map: ObstacleMap,
    random_x_range: tuple[float, float],
    random_y_range: tuple[float, float],
    num_circle_obs: int,
    radius_range: tuple[float, float],
    max_iteration: int = 1000,
    seed: int = 42,
    dynamic_obstacles: int = 0,
    dynamic_speed_range: tuple[float, float] = (0.1, 0.5),
) -> None:
    rng = np.random.default_rng(seed)

    obstacle_map.clear_obstacles()

    for idx in range(num_circle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center = np.array(
                [
                    rng.uniform(*random_x_range),
                    rng.uniform(*random_y_range),
                ]
            )
            radius = float(rng.uniform(*radius_range))

            is_overlap = any(
                np.linalg.norm(circle_obs.center - center) <= (circle_obs.radius + radius)
                for circle_obs in obstacle_map.circle_obs_list
            )
            if not is_overlap:
                dynamic = idx < dynamic_obstacles
                velocity = None
                if dynamic:
                    speed = float(rng.uniform(*dynamic_speed_range))
                    heading = float(rng.uniform(0.0, 2.0 * np.pi))
                    velocity = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float64)
                obstacle_map.add_circle_obstacle(center=center, radius=radius, velocity=velocity, dynamic=dynamic)
                break
            num_trial += 1

        if num_trial == max_iteration:
            raise RuntimeError("Cannot generate random obstacles (max iteration reached).")

    obstacle_map.convert_to_torch()
