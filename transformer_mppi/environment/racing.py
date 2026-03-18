from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from transformer_mppi.environment.maps import LaneMap, ObstacleMap
from transformer_mppi.environment.obstacles import generate_random_obstacles
from transformer_mppi.utils import (
    Array,
    angle_normalize,
    as_array,
    as_dtype,
    make_csv_paths,
    make_side_lane,
    resolve_device,
    to_numpy,
)


class RacingEnv:
    def __init__(
        self,
        circuit_csv: str | Path = "circuit.csv",
        num_obstacles: int = 50,
        obstacle_radius: float = 0.8,
        dynamic_obstacles: int = 0,
        dynamic_speed_range: tuple[float, float] = (0.1, 0.5),
        map_size: tuple[int, int] = (80, 80),
        cell_size: float = 0.1,
        line_width: float = 6.5,
        device: jax.Device | str | None = None,
        dtype: Any = jnp.float32,
        seed: int = 42,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype = as_dtype(dtype)
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        self.u_min = as_array([-2.0, -0.25], dtype=self.dtype, device=self.device)
        self.u_max = as_array([2.0, 0.25], dtype=self.dtype, device=self.device)

        self.L = as_array(1.0, dtype=self.dtype, device=self.device)
        self.V_MAX = as_array(8.0, dtype=self.dtype, device=self.device)

        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.dynamic_obstacles = dynamic_obstacles
        self.dynamic_speed_range = dynamic_speed_range
        self.map_size = map_size
        self.cell_size = cell_size
        self.line_width = line_width

        path, _, _ = make_csv_paths(Path(circuit_csv))
        self._racing_center_path_np = path
        self.racing_center_path = as_array(path, dtype=self.dtype, device=self.device)
        self.right_lane, self.left_lane = make_side_lane(path, lane_width=self.line_width)

        self._lane_map = LaneMap(
            lane=path,
            lane_width=self.line_width * 0.8,
            map_size=self.map_size,
            cell_size=self.cell_size,
            device=self.device,
            dtype=self.dtype,
        )
        self._obstacle_map = ObstacleMap(
            map_size=self.map_size,
            cell_size=self.cell_size,
            device=self.device,
            dtype=self.dtype,
        )

        self._start_pos = as_array(path[0, :2], dtype=self.dtype, device=self.device)
        self._goal_pos = as_array(path[-1, :2], dtype=self.dtype, device=self.device)
        self._x_lim = as_array(self._obstacle_map.x_lim, dtype=self.dtype, device=self.device)
        self._y_lim = as_array(self._obstacle_map.y_lim, dtype=self.dtype, device=self.device)

        self._robot_state = as_array(jnp.zeros(4, dtype=self.dtype), device=self.device)
        self.regenerate_map(seed=seed, dynamic_obstacles=dynamic_obstacles)
        self.reset()

    def regenerate_map(self, seed: int | None = None, dynamic_obstacles: int | None = None) -> None:
        if dynamic_obstacles is not None:
            self.dynamic_obstacles = dynamic_obstacles
        use_seed = self.seed + 1000 if seed is None else seed + 1000
        generate_random_obstacles(
            obstacle_map=self._obstacle_map,
            random_x_range=(-35, 35),
            random_y_range=(-35, 35),
            num_circle_obs=self.num_obstacles,
            radius_range=(self.obstacle_radius, self.obstacle_radius),
            max_iteration=4000,
            seed=use_seed,
            dynamic_obstacles=self.dynamic_obstacles,
            dynamic_speed_range=self.dynamic_speed_range,
        )

    def reset(self) -> Array:
        heading = angle_normalize(
            jnp.arctan2(
                self.racing_center_path[1, 1] - self._start_pos[1],
                self.racing_center_path[1, 0] - self._start_pos[0],
            )
        )
        self._robot_state = self._robot_state.at[:2].set(self._start_pos)
        self._robot_state = self._robot_state.at[2].set(heading)
        self._robot_state = self._robot_state.at[3].set(0.0)
        return self._robot_state.copy()

    def dynamics(self, state: Array, action: Array, delta_t: float = 0.1) -> Array:
        state = jnp.asarray(state, dtype=self.dtype)
        action = jnp.asarray(action, dtype=self.dtype)

        x = state[:, 0:1]
        y = state[:, 1:2]
        theta = state[:, 2:3]
        v = state[:, 3:4]

        accel = jnp.clip(action[:, 0:1], self.u_min[0], self.u_max[0])
        steer = jnp.clip(action[:, 1:2], self.u_min[1], self.u_max[1])

        theta = angle_normalize(theta)
        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dv = accel
        dtheta = v * jnp.tan(steer) / self.L

        new_x = x + dx * delta_t
        new_y = y + dy * delta_t
        new_theta = angle_normalize(theta + dtheta * delta_t)
        new_v = v + dv * delta_t

        clamped_x = jnp.clip(new_x, self._x_lim[0], self._x_lim[1])
        clamped_y = jnp.clip(new_y, self._y_lim[0], self._y_lim[1])
        clamped_v = jnp.clip(new_v, -self.V_MAX, self.V_MAX)
        return jnp.concatenate([clamped_x, clamped_y, new_theta, clamped_v], axis=1)

    def step(self, action: Array, update_dynamic_obstacles: bool = True) -> tuple[Array, bool, bool]:
        action = as_array(action, dtype=self.dtype, device=self.device)
        action = jnp.clip(action, self.u_min, self.u_max)
        self._robot_state = self.dynamics(self._robot_state[None, :], action[None, :]).squeeze(0)

        if update_dynamic_obstacles and self.dynamic_obstacles > 0:
            self._obstacle_map.update_dynamic_obstacles(dt=0.1)

        goal_reached = bool(np.asarray(jnp.linalg.norm(self._robot_state[:2] - self._goal_pos) < 1.5))
        collision_cost = self.collision_check(self._robot_state[:2][None, None, :]).squeeze()
        collision = bool(np.asarray(collision_cost >= 1.0))
        return self._robot_state.copy(), goal_reached, collision

    def collision_check(self, pos_batch: Array) -> Array:
        collisions = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        collisions = collisions + self._lane_map.compute_cost(pos_batch).squeeze(1)
        return collisions

    def calc_reference_trajectory(
        self,
        state: Array,
        cind: int,
        horizon: int,
        dl: float = 0.1,
        lookahead_distance: float = 3.0,
        reference_path_interval: float = 0.85,
    ) -> tuple[Array, int]:
        state = jnp.asarray(state, dtype=self.dtype)
        deltas = self.racing_center_path[:, :2] - state[:2]
        closest_idx = jnp.argmin(jnp.sum(deltas * deltas, axis=1))
        ind = jnp.maximum(jnp.asarray(cind, dtype=jnp.int32), closest_idx.astype(jnp.int32))

        travel = lookahead_distance + reference_path_interval * jnp.arange(1, horizon + 2, dtype=self.dtype)
        dind = jnp.rint(travel / dl).astype(jnp.int32)
        path_indices = ind + dind
        clipped_indices = jnp.clip(path_indices, 0, self.racing_center_path.shape[0] - 1)

        xref = jnp.zeros((horizon + 1, 4), dtype=self.dtype)
        xref = xref.at[:, :3].set(self.racing_center_path[clipped_indices, :3])
        velocities = jnp.where(path_indices < self.racing_center_path.shape[0], self.V_MAX, 0.0)
        xref = xref.at[:, 3].set(velocities)
        return xref, int(np.asarray(ind))

    def racing_cost_function(self, state: Array, action: Array, info: dict) -> Array:
        del action
        state = jnp.asarray(state, dtype=self.dtype)
        ref_path = jnp.asarray(info["ref_path"], dtype=self.dtype)
        t_idx = jnp.asarray(info.get("t", 0), dtype=jnp.int32)

        ec = jnp.sin(ref_path[t_idx, 2]) * (state[:, 0] - ref_path[t_idx, 0]) - jnp.cos(ref_path[t_idx, 2]) * (
            state[:, 1] - ref_path[t_idx, 1]
        )
        el = -jnp.cos(ref_path[t_idx, 2]) * (state[:, 0] - ref_path[t_idx, 0]) - jnp.sin(ref_path[t_idx, 2]) * (
            state[:, 1] - ref_path[t_idx, 1]
        )

        path_cost = 2.0 * jnp.square(ec) + 3.0 * jnp.square(el)
        velocity_cost = 2.0 * jnp.square(state[:, 3] - ref_path[t_idx, 3])

        pos_batch = state[:, :2][:, None, :]
        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        obstacle_cost = obstacle_cost + self._lane_map.compute_cost(pos_batch).squeeze(1)
        return path_cost + velocity_cost + 10000.0 * obstacle_cost

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

    def get_lane_waypoints(self, state: np.ndarray, n_waypoints: int) -> np.ndarray:
        deltas = self._racing_center_path_np[:, :2] - state[:2]
        closest_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        waypoints = self._racing_center_path_np[closest_idx : closest_idx + n_waypoints, :3]
        if waypoints.shape[0] < n_waypoints:
            pad_length = n_waypoints - waypoints.shape[0]
            pad = np.tile(waypoints[-1:], (pad_length, 1))
            waypoints = np.vstack((waypoints, pad))
        return waypoints

    def get_context(self, state: np.ndarray, max_obstacles: int | None = None, n_waypoints: int = 10) -> np.ndarray:
        obstacle_info = self.get_obstacle_centers(max_obstacles=max_obstacles).reshape(-1)
        lane_info = self.get_lane_waypoints(state=state, n_waypoints=n_waypoints).reshape(-1)
        return np.concatenate([obstacle_info, lane_info])

    def render(
        self,
        trajectory: list[np.ndarray] | None = None,
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.left_lane[:, 0], self.left_lane[:, 1], "g--", linewidth=1.5)
        center_path = to_numpy(self.racing_center_path)
        ax.plot(center_path[:, 0], center_path[:, 1], "k--", linewidth=1.0)
        ax.plot(self.right_lane[:, 0], self.right_lane[:, 1], "g--", linewidth=1.5)
        self._obstacle_map.render(ax, zorder=1)

        if trajectory:
            traj = np.asarray(trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color="purple", linewidth=2.0, label="trajectory")
            ax.scatter(traj[0, 0], traj[0, 1], color="blue", label="start")
            ax.scatter(traj[-1, 0], traj[-1, 1], color="red", label="end")

        if title:
            ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.axis("equal")
        ax.legend(loc="upper right")
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


__all__ = ["RacingEnv"]
