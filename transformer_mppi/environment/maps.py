from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


@dataclass
class CircleObstacle:
    center: np.ndarray
    radius: float
    velocity: np.ndarray | None = None
    dynamic: bool = False


class ObstacleMap:
    def __init__(
        self,
        map_size: tuple[int, int],
        cell_size: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.map_size = map_size
        self.cell_size = cell_size

        cell_map_dim = [int(map_size[0] / cell_size), int(map_size[1] / cell_size)]
        self._map = np.zeros(cell_map_dim)
        self._cell_map_origin = np.array([cell_map_dim[0] / 2, cell_map_dim[1] / 2]).astype(int)

        x_range = self.cell_size * self._map.shape[0]
        y_range = self.cell_size * self._map.shape[1]
        self.x_lim = [-x_range / 2, x_range / 2]
        self.y_lim = [-y_range / 2, y_range / 2]

        self._map_torch: torch.Tensor | None = None
        self._torch_cell_map_origin: torch.Tensor | None = None
        self.circle_obs_list: list[CircleObstacle] = []

    def clear_obstacles(self) -> None:
        self.circle_obs_list = []
        self._map.fill(0)

    def add_circle_obstacle(
        self,
        center: np.ndarray,
        radius: float,
        velocity: np.ndarray | None = None,
        dynamic: bool = False,
    ) -> None:
        self.circle_obs_list.append(
            CircleObstacle(center=np.asarray(center, dtype=np.float64), radius=float(radius), velocity=velocity, dynamic=dynamic)
        )
        self._rasterize_circle(self.circle_obs_list[-1])

    def _rasterize_circle(self, obstacle: CircleObstacle) -> None:
        center_occ = (obstacle.center / self.cell_size) + self._cell_map_origin
        center_occ = np.round(center_occ).astype(int)
        radius_occ = int(obstacle.radius / self.cell_size)

        for i in range(-radius_occ, radius_occ + 1):
            for j in range(-radius_occ, radius_occ + 1):
                if i**2 + j**2 <= radius_occ**2:
                    i_bounded = np.clip(center_occ[0] + i, 0, self._map.shape[0] - 1)
                    j_bounded = np.clip(center_occ[1] + j, 0, self._map.shape[1] - 1)
                    self._map[i_bounded, j_bounded] = 1

    def rebuild_occupancy(self) -> None:
        self._map.fill(0)
        for obstacle in self.circle_obs_list:
            self._rasterize_circle(obstacle)
        self.convert_to_torch()

    def convert_to_torch(self) -> torch.Tensor:
        self._map_torch = torch.from_numpy(self._map).to(self.device, self.dtype)
        self._torch_cell_map_origin = torch.from_numpy(self._cell_map_origin).to(self.device, self.dtype)
        return self._map_torch

    def update_dynamic_obstacles(self, dt: float = 0.1) -> None:
        changed = False
        for obstacle in self.circle_obs_list:
            if not obstacle.dynamic or obstacle.velocity is None:
                continue
            changed = True
            next_center = obstacle.center + obstacle.velocity * dt

            if next_center[0] < self.x_lim[0] or next_center[0] > self.x_lim[1]:
                obstacle.velocity[0] *= -1
            if next_center[1] < self.y_lim[0] or next_center[1] > self.y_lim[1]:
                obstacle.velocity[1] *= -1

            obstacle.center = np.array(
                [
                    np.clip(obstacle.center[0] + obstacle.velocity[0] * dt, self.x_lim[0], self.x_lim[1]),
                    np.clip(obstacle.center[1] + obstacle.velocity[1] * dt, self.y_lim[0], self.y_lim[1]),
                ],
                dtype=np.float64,
            )

        if changed:
            self.rebuild_occupancy()

    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        if self._map_torch is None or self._torch_cell_map_origin is None:
            raise ValueError("Obstacle map not prepared. Call convert_to_torch() first.")

        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, self.dtype)

        x_occ = (x / self.cell_size) + self._torch_cell_map_origin
        x_occ = torch.round(x_occ).long().to(self.device)

        is_out_of_bound = torch.logical_or(
            torch.logical_or(x_occ[..., 0] < 0, x_occ[..., 0] >= self._map_torch.shape[0]),
            torch.logical_or(x_occ[..., 1] < 0, x_occ[..., 1] >= self._map_torch.shape[1]),
        )
        x_occ[..., 0] = torch.clamp(x_occ[..., 0], 0, self._map_torch.shape[0] - 1)
        x_occ[..., 1] = torch.clamp(x_occ[..., 1], 0, self._map_torch.shape[1] - 1)

        collisions = self._map_torch[x_occ[..., 0], x_occ[..., 1]]
        collisions[is_out_of_bound] = 1.0
        return collisions

    def render(self, ax, zorder: int = 0) -> None:
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect("equal")

        for circle_obs in self.circle_obs_list:
            color = "tab:orange" if circle_obs.dynamic else "gray"
            ax.add_patch(plt.Circle(circle_obs.center, circle_obs.radius, color=color, alpha=0.7, zorder=zorder))


class LaneMap:
    def __init__(
        self,
        lane: np.ndarray,
        lane_width: float,
        map_size: tuple[int, int],
        cell_size: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if lane_width <= 0:
            raise ValueError("lane_width must be positive")
        if len(lane.shape) != 2 or lane.shape[1] != 3:
            raise ValueError("lane must have shape (N,3)")

        self.device = torch.device(device)
        self.dtype = dtype

        cell_map_dim = [int(map_size[0] / cell_size), int(map_size[1] / cell_size)]
        self._map = np.ones(cell_map_dim)
        self._cell_size = cell_size
        self._cell_map_origin = np.array([cell_map_dim[0] // 2, cell_map_dim[1] // 2])
        self._torch_cell_map_origin = torch.from_numpy(self._cell_map_origin).to(self.device, self.dtype)

        self.x_lim = [-map_size[0] / 2, map_size[0] / 2]
        self.y_lim = [-map_size[1] / 2, map_size[1] / 2]

        self._populate_map(lane, lane_width)
        self._map_torch = torch.tensor(self._map, device=self.device, dtype=self.dtype)

    def _populate_map(self, lane: np.ndarray, lane_width: float) -> None:
        for point in lane:
            x, y, _ = point
            cell_x = int(round(x / self._cell_size)) + self._cell_map_origin[0]
            cell_y = int(round(y / self._cell_size)) + self._cell_map_origin[1]
            if 0 <= cell_x < self._map.shape[0] and 0 <= cell_y < self._map.shape[1]:
                self._map[cell_x, cell_y] = 0

        distance_map = distance_transform_edt(self._map)
        max_distance = (lane_width / 2) / self._cell_size
        self._map = np.where(distance_map <= max_distance, 0, 1)

    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(self.device, self.dtype)

        x_occ = (x / self._cell_size) + self._torch_cell_map_origin
        x_occ = torch.round(x_occ).long().to(self.device)

        is_out_of_bound = torch.logical_or(
            torch.logical_or(x_occ[..., 0] < 0, x_occ[..., 0] >= self._map_torch.shape[0]),
            torch.logical_or(x_occ[..., 1] < 0, x_occ[..., 1] >= self._map_torch.shape[1]),
        )
        x_occ[..., 0] = torch.clamp(x_occ[..., 0], 0, self._map_torch.shape[0] - 1)
        x_occ[..., 1] = torch.clamp(x_occ[..., 1], 0, self._map_torch.shape[1] - 1)

        collisions = self._map_torch[x_occ[..., 0], x_occ[..., 1]]
        collisions[is_out_of_bound] = 1.0
        return collisions

    def render_occupancy(self, ax) -> None:
        ax.imshow(self._map.T, origin="lower", extent=[*self.x_lim, *self.y_lim], cmap="gray_r", alpha=0.3)


__all__ = ["CircleObstacle", "ObstacleMap", "LaneMap"]
