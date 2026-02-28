from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_mppi.environment.maps import LaneMap, ObstacleMap
from transformer_mppi.environment.obstacles import generate_random_obstacles
from transformer_mppi.utils import angle_normalize, make_csv_paths, make_side_lane


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
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.u_min = torch.tensor([-2.0, -0.25], device=self.device, dtype=self.dtype)
        self.u_max = torch.tensor([2.0, 0.25], device=self.device, dtype=self.dtype)

        self.L = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        self.V_MAX = torch.tensor(8.0, device=self.device, dtype=self.dtype)

        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.dynamic_obstacles = dynamic_obstacles
        self.dynamic_speed_range = dynamic_speed_range
        self.map_size = map_size
        self.cell_size = cell_size
        self.line_width = line_width

        path, _, _ = make_csv_paths(Path(circuit_csv))
        self.racing_center_path = torch.tensor(path, device=self.device, dtype=self.dtype)
        self.right_lane, self.left_lane = make_side_lane(path, lane_width=self.line_width)

        self._lane_map = LaneMap(
            lane=self.racing_center_path.cpu().numpy(),
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

        self._start_pos = torch.tensor(
            [self.racing_center_path[0][0], self.racing_center_path[0][1]],
            device=self.device,
            dtype=self.dtype,
        )
        self._goal_pos = torch.tensor(
            [self.racing_center_path[-1][0], self.racing_center_path[-1][1]],
            device=self.device,
            dtype=self.dtype,
        )

        self._robot_state = torch.zeros(4, device=self.device, dtype=self.dtype)
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

    def reset(self) -> torch.Tensor:
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self.racing_center_path[1][1] - self._start_pos[1],
                self.racing_center_path[1][0] - self._start_pos[0],
            )
        )
        self._robot_state[3] = 0.0
        return self._robot_state.clone()

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1) -> torch.Tensor:
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = state[:, 3].view(-1, 1)

        accel = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        steer = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])

        theta = angle_normalize(theta)
        dx = v * torch.cos(theta)
        dy = v * torch.sin(theta)
        dv = accel
        dtheta = v * torch.tan(steer) / self.L

        new_x = x + dx * delta_t
        new_y = y + dy * delta_t
        new_theta = angle_normalize(theta + dtheta * delta_t)
        new_v = v + dv * delta_t

        x_lim = torch.tensor(self._obstacle_map.x_lim, device=self.device, dtype=self.dtype)
        y_lim = torch.tensor(self._obstacle_map.y_lim, device=self.device, dtype=self.dtype)
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])
        clamped_v = torch.clamp(new_v, -self.V_MAX, self.V_MAX)

        return torch.cat([clamped_x, clamped_y, new_theta, clamped_v], dim=1)

    def step(self, action: torch.Tensor, update_dynamic_obstacles: bool = True) -> tuple[torch.Tensor, bool, bool]:
        action = torch.clamp(action, self.u_min, self.u_max)
        self._robot_state = self.dynamics(self._robot_state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

        if update_dynamic_obstacles and self.dynamic_obstacles > 0:
            self._obstacle_map.update_dynamic_obstacles(dt=0.1)

        goal_reached = bool(torch.norm(self._robot_state[:2] - self._goal_pos) < 1.5)
        collision_cost = self.collision_check(self._robot_state[:2].view(1, 1, 2)).squeeze().item()
        collision = bool(collision_cost >= 1.0)
        return self._robot_state.clone(), goal_reached, collision

    def collision_check(self, pos_batch: torch.Tensor) -> torch.Tensor:
        collisions = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        collisions += self._lane_map.compute_cost(pos_batch).squeeze(1)
        return collisions

    def calc_reference_trajectory(
        self,
        state: torch.Tensor,
        cind: int,
        horizon: int,
        dl: float = 0.1,
        lookahead_distance: float = 3.0,
        reference_path_interval: float = 0.85,
    ) -> tuple[torch.Tensor, int]:
        xref = torch.zeros((horizon + 1, 4), dtype=state.dtype, device=state.device)
        path = self.racing_center_path
        ncourse = len(path)

        ind = min(
            range(ncourse),
            key=lambda i: np.hypot(
                path[i, 0].item() - state[0].item(),
                path[i, 1].item() - state[1].item(),
            ),
        )
        ind = max(cind, ind)

        travel = lookahead_distance
        for i in range(horizon + 1):
            travel += reference_path_interval
            dind = int(round(travel / dl))
            if (ind + dind) < ncourse:
                xref[i, :3] = path[ind + dind]
                xref[i, 3] = self.V_MAX
            else:
                xref[i, :3] = path[-1]
                xref[i, 3] = 0.0

        return xref, ind

    def racing_cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict) -> torch.Tensor:
        t_idx = info.get("t", 0)
        ref_path = info["ref_path"]

        ec = torch.sin(ref_path[t_idx, 2]) * (state[:, 0] - ref_path[t_idx, 0]) - torch.cos(ref_path[t_idx, 2]) * (
            state[:, 1] - ref_path[t_idx, 1]
        )
        el = -torch.cos(ref_path[t_idx, 2]) * (state[:, 0] - ref_path[t_idx, 0]) - torch.sin(ref_path[t_idx, 2]) * (
            state[:, 1] - ref_path[t_idx, 1]
        )

        path_cost = 2.0 * ec.pow(2) + 3.0 * el.pow(2)
        velocity_cost = 2.0 * (state[:, 3] - ref_path[t_idx, 3]).pow(2)

        pos_batch = state[:, :2].unsqueeze(1)
        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        obstacle_cost += self._lane_map.compute_cost(pos_batch).squeeze(1)

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
        closest_idx = min(
            range(len(self.racing_center_path)),
            key=lambda i: np.hypot(
                self.racing_center_path[i, 0].item() - state[0],
                self.racing_center_path[i, 1].item() - state[1],
            ),
        )
        waypoints = self.racing_center_path[closest_idx : closest_idx + n_waypoints, :3].cpu().numpy()
        if waypoints.shape[0] < n_waypoints:
            pad_length = n_waypoints - waypoints.shape[0]
            pad = np.tile(waypoints[-1:], (pad_length, 1))
            waypoints = np.vstack((waypoints, pad))
        return waypoints

    def get_context(self, state: np.ndarray, max_obstacles: int | None = None, n_waypoints: int = 10) -> np.ndarray:
        obstacle_info = self.get_obstacle_centers(max_obstacles=max_obstacles).reshape(-1)
        lane_info = self.get_lane_waypoints(state=state, n_waypoints=n_waypoints).reshape(-1)
        return np.concatenate([obstacle_info, lane_info])

    def render(self, trajectory: list[np.ndarray] | None = None, title: str | None = None, save_path: str | Path | None = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.left_lane[:, 0], self.left_lane[:, 1], "g--", linewidth=1.5)
        ax.plot(self.racing_center_path[:, 0].cpu().numpy(), self.racing_center_path[:, 1].cpu().numpy(), "k--", linewidth=1.0)
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
