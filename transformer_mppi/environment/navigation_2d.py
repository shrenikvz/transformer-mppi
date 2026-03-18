from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from transformer_mppi.environment.maps import ObstacleMap
from transformer_mppi.environment.obstacles import generate_random_obstacles
from transformer_mppi.utils import Array, angle_normalize, as_array, as_dtype, resolve_device, to_numpy


class Navigation2DEnv:
    def __init__(
        self,
        num_obstacles: int = 15,
        obstacle_radius: float = 1.0,
        dynamic_obstacles: int = 0,
        dynamic_speed_range: tuple[float, float] = (0.1, 0.5),
        map_size: tuple[int, int] = (20, 20),
        cell_size: float = 0.1,
        start_pos: tuple[float, float] = (-9.0, -9.0),
        goal_pos: tuple[float, float] = (9.0, 9.0),
        device: jax.Device | str | None = None,
        dtype: Any = jnp.float32,
        seed: int = 42,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype = as_dtype(dtype)
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.dynamic_obstacles = dynamic_obstacles
        self.dynamic_speed_range = dynamic_speed_range
        self.map_size = map_size
        self.cell_size = cell_size

        self.u_min = as_array([0.0, -1.0], dtype=self.dtype, device=self.device)
        self.u_max = as_array([2.0, 1.0], dtype=self.dtype, device=self.device)

        self._obstacle_map = ObstacleMap(map_size=map_size, cell_size=cell_size, device=self.device, dtype=self.dtype)
        self._start_pos = as_array(start_pos, dtype=self.dtype, device=self.device)
        self._goal_pos = as_array(goal_pos, dtype=self.dtype, device=self.device)
        self._x_lim = as_array(self._obstacle_map.x_lim, dtype=self.dtype, device=self.device)
        self._y_lim = as_array(self._obstacle_map.y_lim, dtype=self.dtype, device=self.device)

        self._robot_state = as_array(jnp.zeros(3, dtype=self.dtype), device=self.device)
        self.regenerate_map(seed=seed, dynamic_obstacles=dynamic_obstacles)
        self.reset()

    def regenerate_map(self, seed: int | None = None, dynamic_obstacles: int | None = None) -> None:
        if dynamic_obstacles is not None:
            self.dynamic_obstacles = dynamic_obstacles
        use_seed = self.seed if seed is None else seed

        generate_random_obstacles(
            obstacle_map=self._obstacle_map,
            random_x_range=(-7.5, 7.5),
            random_y_range=(-7.5, 7.5),
            num_circle_obs=self.num_obstacles,
            radius_range=(self.obstacle_radius, self.obstacle_radius),
            max_iteration=2000,
            seed=use_seed,
            dynamic_obstacles=self.dynamic_obstacles,
            dynamic_speed_range=self.dynamic_speed_range,
        )

    def reset(self) -> Array:
        heading = angle_normalize(
            jnp.arctan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )
        self._robot_state = self._robot_state.at[:2].set(self._start_pos)
        self._robot_state = self._robot_state.at[2].set(heading)
        return self._robot_state.copy()

    def dynamics(self, state: Array, action: Array, delta_t: float = 0.1) -> Array:
        state = jnp.asarray(state, dtype=self.dtype)
        action = jnp.asarray(action, dtype=self.dtype)

        x = state[:, 0:1]
        y = state[:, 1:2]
        theta = state[:, 2:3]

        v = jnp.clip(action[:, 0:1], self.u_min[0], self.u_max[0])
        omega = jnp.clip(action[:, 1:2], self.u_min[1], self.u_max[1])

        theta = angle_normalize(theta)
        new_x = x + v * jnp.cos(theta) * delta_t
        new_y = y + v * jnp.sin(theta) * delta_t
        new_theta = angle_normalize(theta + omega * delta_t)

        clamped_x = jnp.clip(new_x, self._x_lim[0], self._x_lim[1])
        clamped_y = jnp.clip(new_y, self._y_lim[0], self._y_lim[1])
        return jnp.concatenate([clamped_x, clamped_y, new_theta], axis=1)

    def cost_function(self, state: Array, action: Array, info: dict | None = None) -> Array:
        del action, info
        state = jnp.asarray(state, dtype=self.dtype)
        goal_cost = jnp.linalg.norm(state[:, :2] - self._goal_pos, axis=1)
        pos_batch = state[:, :2][:, None, :]
        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        return goal_cost + 10000.0 * obstacle_cost

    def step(self, action: Array, update_dynamic_obstacles: bool = True) -> tuple[Array, bool, bool]:
        action = as_array(action, dtype=self.dtype, device=self.device)
        action = jnp.clip(action, self.u_min, self.u_max)
        next_state = self.dynamics(self._robot_state[None, :], action[None, :]).squeeze(0)
        self._robot_state = next_state

        if update_dynamic_obstacles and self.dynamic_obstacles > 0:
            self._obstacle_map.update_dynamic_obstacles(dt=0.1)

        goal_reached = bool(np.asarray(jnp.linalg.norm(self._robot_state[:2] - self._goal_pos) < 0.5))
        collision_cost = self._obstacle_map.compute_cost(self._robot_state[:2][None, None, :]).squeeze()
        collision = bool(np.asarray(collision_cost >= 1.0))
        return self._robot_state.copy(), goal_reached, collision

    def get_obstacle_centers(self, max_obstacles: int | None = None) -> np.ndarray:
        centers = np.array([obs.center for obs in self._obstacle_map.circle_obs_list], dtype=np.float64)
        if centers.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        if max_obstacles is None:
            return centers
        if centers.shape[0] >= max_obstacles:
            return centers[:max_obstacles]
        pad = np.zeros((max_obstacles - centers.shape[0], 2), dtype=np.float64)
        return np.vstack([centers, pad])

    def get_context(self, max_obstacles: int | None = None) -> np.ndarray:
        return self.get_obstacle_centers(max_obstacles=max_obstacles).reshape(-1)

    def render(
        self,
        trajectory: list[np.ndarray] | None = None,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        self._obstacle_map.render(ax)
        if trajectory:
            traj = np.asarray(trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color="purple", linewidth=2.0)
            ax.scatter(traj[0, 0], traj[0, 1], color="blue", label="start")
            ax.scatter(traj[-1, 0], traj[-1, 1], color="red", label="end")
        goal_pos = to_numpy(self._goal_pos)
        ax.scatter(goal_pos[0], goal_pos[1], color="green", marker="*", s=120, label="goal")
        ax.grid(True, linestyle=":", alpha=0.5)
        if title:
            ax.set_title(title)
        ax.legend(loc="upper right")
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


__all__ = ["Navigation2DEnv"]
