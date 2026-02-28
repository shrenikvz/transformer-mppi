from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TaskName = Literal["navigation2d", "racing"]
ProfileName = Literal["quick", "paper"]


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int = 256
    num_layers: int = 3
    nhead: int = 8
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    train_envs: int
    max_steps: int
    mppi_samples: int
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float
    val_fraction: float = 0.1


@dataclass(frozen=True)
class BenchmarkConfig:
    episodes: int
    sample_sizes: tuple[int, ...]
    dynamic_obstacle_counts: tuple[int, ...]


@dataclass(frozen=True)
class TaskConfig:
    task: TaskName
    profile: ProfileName
    seed: int
    horizon: int
    k_history: int
    num_obstacles: int
    obstacle_radius: float
    transformer: TransformerConfig
    training: TrainingConfig
    benchmark: BenchmarkConfig
    control_lambda: float
    control_sigmas: tuple[float, float]
    exploration: float
    use_sg_filter: bool
    output_dir: Path

    # Navigation settings
    navigation_map_size: tuple[int, int] = (20, 20)
    navigation_start: tuple[float, float] = (-9.0, -9.0)
    navigation_goal: tuple[float, float] = (9.0, 9.0)

    # Racing settings
    racing_map_size: tuple[int, int] = (80, 80)
    racing_line_width: float = 6.5
    racing_n_waypoints: int = 10
    racing_lookahead_distance: float = 3.0
    racing_reference_path_interval: float = 0.85


def _navigation_quick(output_dir: Path) -> TaskConfig:
    return TaskConfig(
        task="navigation2d",
        profile="quick",
        seed=42,
        horizon=20,
        k_history=5,
        num_obstacles=15,
        obstacle_radius=1.0,
        transformer=TransformerConfig(hidden_size=128, num_layers=2, nhead=8, dropout=0.1),
        training=TrainingConfig(
            train_envs=40,
            max_steps=150,
            mppi_samples=300,
            batch_size=128,
            epochs=80,
            patience=15,
            learning_rate=5e-4,
        ),
        benchmark=BenchmarkConfig(
            episodes=10,
            sample_sizes=(50, 100, 200, 300, 400, 500),
            dynamic_obstacle_counts=(0, 3, 6, 9, 12, 15),
        ),
        control_lambda=1.0,
        control_sigmas=(1.0, 1.0),
        exploration=0.2,
        use_sg_filter=False,
        output_dir=output_dir,
    )


def _navigation_paper(output_dir: Path) -> TaskConfig:
    return TaskConfig(
        task="navigation2d",
        profile="paper",
        seed=42,
        horizon=20,
        k_history=5,
        num_obstacles=15,
        obstacle_radius=1.0,
        transformer=TransformerConfig(hidden_size=256, num_layers=3, nhead=8, dropout=0.1),
        training=TrainingConfig(
            train_envs=1000,
            max_steps=150,
            mppi_samples=1000,
            batch_size=256,
            epochs=2000,
            patience=50,
            learning_rate=5e-4,
        ),
        benchmark=BenchmarkConfig(
            episodes=10,
            sample_sizes=(50, 100, 200, 300, 400, 500),
            dynamic_obstacle_counts=(0, 3, 6, 9, 12, 15),
        ),
        control_lambda=1.0,
        control_sigmas=(1.0, 1.0),
        exploration=0.2,
        use_sg_filter=False,
        output_dir=output_dir,
    )


def _racing_quick(output_dir: Path) -> TaskConfig:
    return TaskConfig(
        task="racing",
        profile="quick",
        seed=42,
        horizon=25,
        k_history=5,
        num_obstacles=50,
        obstacle_radius=0.8,
        transformer=TransformerConfig(hidden_size=128, num_layers=2, nhead=8, dropout=0.1),
        training=TrainingConfig(
            train_envs=30,
            max_steps=500,
            mppi_samples=2500,
            batch_size=64,
            epochs=80,
            patience=15,
            learning_rate=5e-4,
        ),
        benchmark=BenchmarkConfig(
            episodes=10,
            sample_sizes=(5000, 6000, 7000, 8000, 9000, 10000),
            dynamic_obstacle_counts=(0, 10, 20, 30, 40, 50),
        ),
        control_lambda=1.0,
        control_sigmas=(1.0, 1.0),
        exploration=0.2,
        use_sg_filter=False,
        output_dir=output_dir,
    )


def _racing_paper(output_dir: Path) -> TaskConfig:
    return TaskConfig(
        task="racing",
        profile="paper",
        seed=42,
        horizon=25,
        k_history=5,
        num_obstacles=50,
        obstacle_radius=0.8,
        transformer=TransformerConfig(hidden_size=256, num_layers=3, nhead=8, dropout=0.1),
        training=TrainingConfig(
            train_envs=300,
            max_steps=500,
            mppi_samples=10000,
            batch_size=256,
            epochs=2000,
            patience=50,
            learning_rate=5e-4,
        ),
        benchmark=BenchmarkConfig(
            episodes=10,
            sample_sizes=(5000, 6000, 7000, 8000, 9000, 10000),
            dynamic_obstacle_counts=(0, 10, 20, 30, 40, 50),
        ),
        control_lambda=1.0,
        control_sigmas=(1.0, 1.0),
        exploration=0.2,
        use_sg_filter=False,
        output_dir=output_dir,
    )


def get_task_config(task: TaskName, profile: ProfileName, output_dir: str | Path) -> TaskConfig:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if task == "navigation2d" and profile == "quick":
        return _navigation_quick(out)
    if task == "navigation2d" and profile == "paper":
        return _navigation_paper(out)
    if task == "racing" and profile == "quick":
        return _racing_quick(out)
    if task == "racing" and profile == "paper":
        return _racing_paper(out)
    raise ValueError(f"Unsupported task/profile: {task}/{profile}")
