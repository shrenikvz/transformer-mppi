from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_mppi.environment.maps import ObstacleMap
from transformer_mppi.environment.obstacles import generate_random_obstacles
from transformer_mppi.utils import angle_normalize


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

        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.dynamic_obstacles = dynamic_obstacles
        self.dynamic_speed_range = dynamic_speed_range
        self.map_size = map_size
        self.cell_size = cell_size

        self.u_min = torch.tensor([0.0, -1.0], device=self.device, dtype=self.dtype)
        self.u_max = torch.tensor([2.0, 1.0], device=self.device, dtype=self.dtype)

        self._obstacle_map = ObstacleMap(map_size=map_size, cell_size=cell_size, device=self.device, dtype=self.dtype)
        self._start_pos = torch.tensor(start_pos, device=self.device, dtype=self.dtype)
        self._goal_pos = torch.tensor(goal_pos, device=self.device, dtype=self.dtype)

        self._robot_state = torch.zeros(3, device=self.device, dtype=self.dtype)
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

    def reset(self) -> torch.Tensor:
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )
        return self._robot_state.clone()

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1) -> torch.Tensor:
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)

        v = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        omega = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])

        theta = angle_normalize(theta)
        new_x = x + v * torch.cos(theta) * delta_t
        new_y = y + v * torch.sin(theta) * delta_t
        new_theta = angle_normalize(theta + omega * delta_t)

        x_lim = torch.tensor(self._obstacle_map.x_lim, device=self.device, dtype=self.dtype)
        y_lim = torch.tensor(self._obstacle_map.y_lim, device=self.device, dtype=self.dtype)

        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])

        return torch.cat([clamped_x, clamped_y, new_theta], dim=1)

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, info: dict | None = None) -> torch.Tensor:
        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)
        pos_batch = state[:, :2].unsqueeze(1)
        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        return goal_cost + 10000.0 * obstacle_cost

    def step(self, action: torch.Tensor, update_dynamic_obstacles: bool = True) -> tuple[torch.Tensor, bool, bool]:
        action = torch.clamp(action, self.u_min, self.u_max)
        next_state = self.dynamics(self._robot_state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
        self._robot_state = next_state

        if update_dynamic_obstacles and self.dynamic_obstacles > 0:
            self._obstacle_map.update_dynamic_obstacles(dt=0.1)

        goal_reached = bool(torch.norm(self._robot_state[:2] - self._goal_pos) < 0.5)
        collision = bool(self._obstacle_map.compute_cost(self._robot_state[:2].view(1, 1, 2)).item() >= 1.0)
        return self._robot_state.clone(), goal_reached, collision

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

    def render(self, trajectory: list[np.ndarray] | None = None, title: str | None = None, save_path: str | Path | None = None) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        self._obstacle_map.render(ax)
        if trajectory:
            traj = np.asarray(trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color="purple", linewidth=2.0)
            ax.scatter(traj[0, 0], traj[0, 1], color="blue", label="start")
            ax.scatter(traj[-1, 0], traj[-1, 1], color="red", label="end")
        ax.scatter(self._goal_pos[0].item(), self._goal_pos[1].item(), color="green", marker="*", s=120, label="goal")
        ax.grid(True, linestyle=":", alpha=0.5)
        if title:
            ax.set_title(title)
        ax.legend(loc="upper right")
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


__all__ = ["Navigation2DEnv"]
