from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformer_mppi.config import get_task_config
from transformer_mppi.controllers import MPPI, TransformerMPPIController
from transformer_mppi.environment import Navigation2DEnv, RacingEnv
from transformer_mppi.training import TransformerArtifacts, train_transformer_model
from transformer_mppi.utils import set_global_seed


@dataclass
class EpisodeMetrics:
    total_cost: float
    steps: int
    success: bool
    collision: bool


def _build_navigation_env(cfg, seed: int, dynamic_obstacles: int = 0) -> Navigation2DEnv:
    return Navigation2DEnv(
        num_obstacles=cfg.num_obstacles,
        obstacle_radius=cfg.obstacle_radius,
        dynamic_obstacles=dynamic_obstacles,
        map_size=cfg.navigation_map_size,
        start_pos=cfg.navigation_start,
        goal_pos=cfg.navigation_goal,
        seed=seed,
    )


def _build_racing_env(cfg, seed: int, circuit_csv: str | Path, dynamic_obstacles: int = 0) -> RacingEnv:
    return RacingEnv(
        circuit_csv=circuit_csv,
        num_obstacles=cfg.num_obstacles,
        obstacle_radius=cfg.obstacle_radius,
        dynamic_obstacles=dynamic_obstacles,
        map_size=cfg.racing_map_size,
        line_width=cfg.racing_line_width,
        seed=seed,
    )


def _init_mppi_navigation(cfg, env: Navigation2DEnv, num_samples: int, seed: int) -> MPPI:
    sigmas = torch.tensor(cfg.control_sigmas, device=env.device, dtype=env.dtype)
    return MPPI(
        horizon=cfg.horizon,
        num_samples=num_samples,
        dim_state=3,
        dim_control=2,
        dynamics=env.dynamics,
        cost_func=env.cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=sigmas,
        lambda_=cfg.control_lambda,
        exploration=cfg.exploration,
        use_sg_filter=cfg.use_sg_filter,
        device=env.device,
        dtype=env.dtype,
        seed=seed,
    )


def _init_mppi_racing(cfg, env: RacingEnv, num_samples: int, seed: int) -> MPPI:
    sigmas = torch.tensor(cfg.control_sigmas, device=env.device, dtype=env.dtype)
    return MPPI(
        horizon=cfg.horizon,
        num_samples=num_samples,
        dim_state=4,
        dim_control=2,
        dynamics=env.dynamics,
        cost_func=env.racing_cost_function,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=sigmas,
        lambda_=cfg.control_lambda,
        exploration=cfg.exploration,
        use_sg_filter=cfg.use_sg_filter,
        device=env.device,
        dtype=env.dtype,
        seed=seed,
    )


def collect_training_data_navigation(cfg) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for env_idx in range(cfg.training.train_envs):
        env_seed = cfg.seed + env_idx
        env = _build_navigation_env(cfg, seed=env_seed, dynamic_obstacles=0)
        state = env.reset()

        mppi = _init_mppi_navigation(cfg, env, num_samples=cfg.training.mppi_samples, seed=env_seed)
        mppi.reset()

        context = env.get_context(max_obstacles=cfg.num_obstacles)
        state_np = state.cpu().numpy()
        history = [np.concatenate([state_np, context]) for _ in range(cfg.k_history)]

        for _ in range(cfg.training.max_steps):
            inputs.append(np.stack(history, axis=0))
            action_seq, _ = mppi.forward(state=state)
            targets.append(action_seq.detach().cpu().numpy())

            next_state, goal_reached, collision = env.step(action_seq[0], update_dynamic_obstacles=False)
            state = next_state
            if collision or goal_reached:
                break

            state_np = state.cpu().numpy()
            context = env.get_context(max_obstacles=cfg.num_obstacles)
            history.pop(0)
            history.append(np.concatenate([state_np, context]))

    return np.asarray(inputs), np.asarray(targets)


def collect_training_data_racing(cfg, circuit_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for env_idx in range(cfg.training.train_envs):
        env_seed = cfg.seed + env_idx
        env = _build_racing_env(cfg, seed=env_seed, circuit_csv=circuit_csv, dynamic_obstacles=0)
        state = env.reset()

        mppi = _init_mppi_racing(cfg, env, num_samples=cfg.training.mppi_samples, seed=env_seed)
        mppi.reset()

        current_state_np = state.cpu().numpy()
        context = env.get_context(state=current_state_np, max_obstacles=cfg.num_obstacles, n_waypoints=cfg.racing_n_waypoints)
        history = [np.concatenate([current_state_np, context]) for _ in range(cfg.k_history)]
        current_path_index = 0

        for step in range(cfg.training.max_steps):
            reference_path, current_path_index = env.calc_reference_trajectory(
                state=state,
                cind=current_path_index,
                horizon=cfg.horizon,
                lookahead_distance=cfg.racing_lookahead_distance,
                reference_path_interval=cfg.racing_reference_path_interval,
            )
            info = {"ref_path": reference_path}

            inputs.append(np.stack(history, axis=0))
            action_seq, _ = mppi.forward(state=state, info=info)
            targets.append(action_seq.detach().cpu().numpy())

            next_state, goal_reached, collision = env.step(action_seq[0], update_dynamic_obstacles=False)
            state = next_state

            if step < 20:
                goal_reached = False
            if collision or goal_reached:
                break

            current_state_np = state.cpu().numpy()
            context = env.get_context(
                state=current_state_np,
                max_obstacles=cfg.num_obstacles,
                n_waypoints=cfg.racing_n_waypoints,
            )
            history.pop(0)
            history.append(np.concatenate([current_state_np, context]))

    return np.asarray(inputs), np.asarray(targets)


def _run_navigation_episode_mppi(cfg, sample_size: int, seed: int, dynamic_obstacles: int) -> EpisodeMetrics:
    env = _build_navigation_env(cfg, seed=seed, dynamic_obstacles=dynamic_obstacles)
    state = env.reset()
    mppi = _init_mppi_navigation(cfg, env, num_samples=sample_size, seed=seed)
    mppi.reset()

    total_cost = 0.0
    success = False
    collision = False

    for step in range(cfg.training.max_steps):
        action_seq, _ = mppi.forward(state=state)
        action = action_seq[0]
        next_state, goal_reached, collided = env.step(action, update_dynamic_obstacles=True)

        total_cost += env.cost_function(next_state.unsqueeze(0), action.unsqueeze(0), {}).item()
        state = next_state
        collision = collided

        if goal_reached:
            success = True
            break
        if collided:
            break

    return EpisodeMetrics(total_cost=total_cost, steps=step + 1, success=success, collision=collision)


def _run_navigation_episode_transformer(
    cfg,
    artifacts: TransformerArtifacts,
    sample_size: int,
    seed: int,
    dynamic_obstacles: int,
) -> EpisodeMetrics:
    env = _build_navigation_env(cfg, seed=seed, dynamic_obstacles=dynamic_obstacles)
    state = env.reset()
    mppi = _init_mppi_navigation(cfg, env, num_samples=sample_size, seed=seed)
    controller = TransformerMPPIController(mppi=mppi, artifacts=artifacts)
    controller.reset()

    state_np = state.cpu().numpy()
    context = env.get_context(max_obstacles=cfg.num_obstacles)
    controller.update_history(state=state_np, context=context)

    total_cost = 0.0
    success = False
    collision = False

    for step in range(cfg.training.max_steps):
        action, _, _ = controller.act(state=state)
        next_state, goal_reached, collided = env.step(action, update_dynamic_obstacles=True)

        total_cost += env.cost_function(next_state.unsqueeze(0), action.unsqueeze(0), {}).item()
        state = next_state
        collision = collided

        if goal_reached:
            success = True
            break
        if collided:
            break

        state_np = state.cpu().numpy()
        context = env.get_context(max_obstacles=cfg.num_obstacles)
        controller.update_history(state=state_np, context=context)

    return EpisodeMetrics(total_cost=total_cost, steps=step + 1, success=success, collision=collision)


def _run_racing_episode_mppi(
    cfg,
    sample_size: int,
    seed: int,
    circuit_csv: str | Path,
    dynamic_obstacles: int,
) -> EpisodeMetrics:
    env = _build_racing_env(cfg, seed=seed, circuit_csv=circuit_csv, dynamic_obstacles=dynamic_obstacles)
    state = env.reset()
    mppi = _init_mppi_racing(cfg, env, num_samples=sample_size, seed=seed)
    mppi.reset()

    total_cost = 0.0
    success = False
    collision = False
    current_path_index = 0

    for step in range(cfg.training.max_steps):
        reference_path, current_path_index = env.calc_reference_trajectory(
            state=state,
            cind=current_path_index,
            horizon=cfg.horizon,
            lookahead_distance=cfg.racing_lookahead_distance,
            reference_path_interval=cfg.racing_reference_path_interval,
        )
        info = {"ref_path": reference_path}

        action_seq, _ = mppi.forward(state=state, info=info)
        action = action_seq[0]
        next_state, goal_reached, collided = env.step(action, update_dynamic_obstacles=True)

        total_cost += env.racing_cost_function(next_state.unsqueeze(0), action.unsqueeze(0), info).item()
        state = next_state
        collision = collided

        if step < 20:
            goal_reached = False

        if goal_reached:
            success = True
            break
        if collided:
            break

    return EpisodeMetrics(total_cost=total_cost, steps=step + 1, success=success, collision=collision)


def _run_racing_episode_transformer(
    cfg,
    artifacts: TransformerArtifacts,
    sample_size: int,
    seed: int,
    circuit_csv: str | Path,
    dynamic_obstacles: int,
) -> EpisodeMetrics:
    env = _build_racing_env(cfg, seed=seed, circuit_csv=circuit_csv, dynamic_obstacles=dynamic_obstacles)
    state = env.reset()
    mppi = _init_mppi_racing(cfg, env, num_samples=sample_size, seed=seed)
    controller = TransformerMPPIController(mppi=mppi, artifacts=artifacts)
    controller.reset()

    current_state_np = state.cpu().numpy()
    context = env.get_context(
        state=current_state_np,
        max_obstacles=cfg.num_obstacles,
        n_waypoints=cfg.racing_n_waypoints,
    )
    controller.update_history(state=current_state_np, context=context)

    total_cost = 0.0
    success = False
    collision = False
    current_path_index = 0

    for step in range(cfg.training.max_steps):
        reference_path, current_path_index = env.calc_reference_trajectory(
            state=state,
            cind=current_path_index,
            horizon=cfg.horizon,
            lookahead_distance=cfg.racing_lookahead_distance,
            reference_path_interval=cfg.racing_reference_path_interval,
        )
        info = {"ref_path": reference_path}

        action, _, _ = controller.act(state=state, info=info)
        next_state, goal_reached, collided = env.step(action, update_dynamic_obstacles=True)

        total_cost += env.racing_cost_function(next_state.unsqueeze(0), action.unsqueeze(0), info).item()
        state = next_state
        collision = collided

        if step < 20:
            goal_reached = False

        if goal_reached:
            success = True
            break
        if collided:
            break

        current_state_np = state.cpu().numpy()
        context = env.get_context(
            state=current_state_np,
            max_obstacles=cfg.num_obstacles,
            n_waypoints=cfg.racing_n_waypoints,
        )
        controller.update_history(state=current_state_np, context=context)

    return EpisodeMetrics(total_cost=total_cost, steps=step + 1, success=success, collision=collision)


def _aggregate_metrics(metrics: list[EpisodeMetrics]) -> tuple[float, float, float]:
    success_rate = 100.0 * (sum(m.success for m in metrics) / len(metrics))
    successful = [m for m in metrics if m.success]

    if successful:
        avg_cost = float(np.mean([m.total_cost for m in successful]))
        avg_steps = float(np.mean([m.steps for m in successful]))
    else:
        avg_cost = float("nan")
        avg_steps = float("nan")

    return avg_cost, avg_steps, success_rate


def _run_sample_sweep_navigation(cfg, artifacts: TransformerArtifacts, out_csv_dir: Path) -> None:
    cost_rows = []
    steps_rows = []
    success_rows = []

    first_sample = cfg.benchmark.sample_sizes[0]
    per_episode_rows_cost = []
    per_episode_rows_steps = []
    per_episode_rows_collision = []

    for sample_size in cfg.benchmark.sample_sizes:
        mppi_metrics = []
        trans_metrics = []

        for ep in range(cfg.benchmark.episodes):
            seed = cfg.seed + 10_000 + ep
            mppi_ep = _run_navigation_episode_mppi(cfg, sample_size=sample_size, seed=seed, dynamic_obstacles=0)
            trans_ep = _run_navigation_episode_transformer(
                cfg,
                artifacts=artifacts,
                sample_size=sample_size,
                seed=seed,
                dynamic_obstacles=0,
            )
            mppi_metrics.append(mppi_ep)
            trans_metrics.append(trans_ep)

            if sample_size == first_sample:
                per_episode_rows_cost.append(
                    {"episode": ep + 1, "original_mppi": mppi_ep.total_cost, "transformer_mppi": trans_ep.total_cost}
                )
                per_episode_rows_steps.append(
                    {"episode": ep + 1, "original_mppi": mppi_ep.steps, "transformer_mppi": trans_ep.steps}
                )
                per_episode_rows_collision.append(
                    {"episode": ep + 1, "original_mppi": int(mppi_ep.collision), "transformer_mppi": int(trans_ep.collision)}
                )

        mppi_avg_cost, mppi_avg_steps, mppi_success = _aggregate_metrics(mppi_metrics)
        trans_avg_cost, trans_avg_steps, trans_success = _aggregate_metrics(trans_metrics)

        cost_rows.append(
            {
                "num_samples": sample_size,
                "original_mppi": mppi_avg_cost,
                "transformer_mppi": trans_avg_cost,
            }
        )
        steps_rows.append(
            {
                "num_samples": sample_size,
                "original_mppi": mppi_avg_steps,
                "transformer_mppi": trans_avg_steps,
            }
        )
        success_rows.append(
            {
                "num_samples": sample_size,
                "original_mppi": mppi_success,
                "transformer_mppi": trans_success,
            }
        )

    pd.DataFrame(cost_rows).to_csv(out_csv_dir / "average_cost_vs_num_samples_obstacle.csv", index=False)
    pd.DataFrame(steps_rows).to_csv(out_csv_dir / "average_steps_vs_num_samples_obstacle.csv", index=False)
    pd.DataFrame(success_rows).to_csv(out_csv_dir / "success_rate_vs_num_samples_obstacle.csv", index=False)

    pd.DataFrame(per_episode_rows_cost).to_csv(out_csv_dir / "cost_per_episode_num_samples_50_obstacle.csv", index=False)
    pd.DataFrame(per_episode_rows_steps).to_csv(out_csv_dir / "steps_per_episode_num_samples_50_obstacle.csv", index=False)
    pd.DataFrame(per_episode_rows_collision).to_csv(out_csv_dir / "collision_vs_episode_num_samples_50_obstacle.csv", index=False)


def _run_dynamic_sweep_navigation(cfg, artifacts: TransformerArtifacts, out_csv_dir: Path) -> None:
    sample_size = cfg.benchmark.sample_sizes[0]
    cost_rows = []
    success_rows = []

    for n_dyn in cfg.benchmark.dynamic_obstacle_counts:
        mppi_metrics = []
        trans_metrics = []

        for ep in range(cfg.benchmark.episodes):
            seed = cfg.seed + 20_000 + ep
            mppi_ep = _run_navigation_episode_mppi(cfg, sample_size=sample_size, seed=seed, dynamic_obstacles=n_dyn)
            trans_ep = _run_navigation_episode_transformer(
                cfg,
                artifacts=artifacts,
                sample_size=sample_size,
                seed=seed,
                dynamic_obstacles=n_dyn,
            )
            mppi_metrics.append(mppi_ep)
            trans_metrics.append(trans_ep)

        mppi_avg_cost, _, mppi_success = _aggregate_metrics(mppi_metrics)
        trans_avg_cost, _, trans_success = _aggregate_metrics(trans_metrics)

        cost_rows.append(
            {
                "num_obstacles": n_dyn,
                "original_mppi": mppi_avg_cost,
                "transformer_mppi": trans_avg_cost,
            }
        )
        success_rows.append(
            {
                "num_obstacles": n_dyn,
                "original_mppi": mppi_success,
                "transformer_mppi": trans_success,
            }
        )

    pd.DataFrame(cost_rows).to_csv(out_csv_dir / "average_cost_vs_num_dynamic_obstacle.csv", index=False)
    pd.DataFrame(success_rows).to_csv(out_csv_dir / "success_rate_vs_num_dynamic_obstacles.csv", index=False)


def _run_sample_sweep_racing(cfg, artifacts: TransformerArtifacts, out_csv_dir: Path, circuit_csv: str | Path) -> None:
    cost_rows = []
    steps_rows = []

    first_sample = cfg.benchmark.sample_sizes[0]
    per_episode_rows_cost = []
    per_episode_rows_steps = []

    for sample_size in cfg.benchmark.sample_sizes:
        mppi_metrics = []
        trans_metrics = []

        for ep in range(cfg.benchmark.episodes):
            seed = cfg.seed + 30_000 + ep
            mppi_ep = _run_racing_episode_mppi(
                cfg,
                sample_size=sample_size,
                seed=seed,
                circuit_csv=circuit_csv,
                dynamic_obstacles=0,
            )
            trans_ep = _run_racing_episode_transformer(
                cfg,
                artifacts=artifacts,
                sample_size=sample_size,
                seed=seed,
                circuit_csv=circuit_csv,
                dynamic_obstacles=0,
            )
            mppi_metrics.append(mppi_ep)
            trans_metrics.append(trans_ep)

            if sample_size == first_sample:
                per_episode_rows_cost.append(
                    {"episode": ep + 1, "original_mppi": mppi_ep.total_cost, "transformer_mppi": trans_ep.total_cost}
                )
                per_episode_rows_steps.append(
                    {"episode": ep + 1, "original_mppi": mppi_ep.steps, "transformer_mppi": trans_ep.steps}
                )

        mppi_avg_cost, mppi_avg_steps, _ = _aggregate_metrics(mppi_metrics)
        trans_avg_cost, trans_avg_steps, _ = _aggregate_metrics(trans_metrics)

        cost_rows.append(
            {
                "num_samples": sample_size,
                "original_mppi": mppi_avg_cost,
                "transformer_mppi": trans_avg_cost,
            }
        )
        steps_rows.append(
            {
                "num_samples": sample_size,
                "original_mppi": mppi_avg_steps,
                "transformer_mppi": trans_avg_steps,
            }
        )

    pd.DataFrame(cost_rows).to_csv(out_csv_dir / "average_cost_vs_num_samples_racing.csv", index=False)
    pd.DataFrame(steps_rows).to_csv(out_csv_dir / "average_steps_vs_num_samples_racing.csv", index=False)

    # Keep filename compatible with existing manuscript assets.
    pd.DataFrame(per_episode_rows_cost).to_csv(out_csv_dir / "cost_per_episode_num_samples_50_racing.csv", index=False)
    pd.DataFrame(per_episode_rows_steps).to_csv(out_csv_dir / "steps_per_episode_num_samples_50_racing.csv", index=False)


def _run_dynamic_sweep_racing(cfg, artifacts: TransformerArtifacts, out_csv_dir: Path, circuit_csv: str | Path) -> None:
    sample_size = cfg.benchmark.sample_sizes[0]
    cost_rows = []

    for n_dyn in cfg.benchmark.dynamic_obstacle_counts:
        mppi_metrics = []
        trans_metrics = []

        for ep in range(cfg.benchmark.episodes):
            seed = cfg.seed + 40_000 + ep
            mppi_ep = _run_racing_episode_mppi(
                cfg,
                sample_size=sample_size,
                seed=seed,
                circuit_csv=circuit_csv,
                dynamic_obstacles=n_dyn,
            )
            trans_ep = _run_racing_episode_transformer(
                cfg,
                artifacts=artifacts,
                sample_size=sample_size,
                seed=seed,
                circuit_csv=circuit_csv,
                dynamic_obstacles=n_dyn,
            )
            mppi_metrics.append(mppi_ep)
            trans_metrics.append(trans_ep)

        mppi_avg_cost, _, _ = _aggregate_metrics(mppi_metrics)
        trans_avg_cost, _, _ = _aggregate_metrics(trans_metrics)

        cost_rows.append(
            {
                "num_obstacles": n_dyn,
                "original_mppi": mppi_avg_cost,
                "transformer_mppi": trans_avg_cost,
            }
        )

    pd.DataFrame(cost_rows).to_csv(out_csv_dir / "average_cost_vs_num_dynamic_racing.csv", index=False)


def _export_prediction_csvs(task: str, artifacts: TransformerArtifacts, input_sequences: np.ndarray, target_sequences: np.ndarray, out_csv_dir: Path) -> None:
    if len(input_sequences) == 0:
        return

    idx = min(5, len(input_sequences) - 1)
    src = input_sequences[idx]
    gt = target_sequences[idx]
    pred = artifacts.predict_action_sequence(src)

    # Truncate in case horizon mismatch due to future changes.
    horizon = min(gt.shape[0], pred.shape[0])
    gt = gt[:horizon]
    pred = pred[:horizon]

    if task == "navigation2d":
        prefix = "transformer_predictions_2d_obstacle"
    else:
        prefix = "transformer_predictions_racing"

    df_u1 = pd.DataFrame({"horizon": np.arange(1, horizon + 1), "original": gt[:, 0], "transformer": pred[:, 0]})
    df_u2 = pd.DataFrame({"horizon": np.arange(1, horizon + 1), "original": gt[:, 1], "transformer": pred[:, 1]})

    df_u1.to_csv(out_csv_dir / f"{prefix}_u1.csv", index=False)
    df_u2.to_csv(out_csv_dir / f"{prefix}_u2.csv", index=False)


def run_reproduction_task(
    task: str,
    profile: str,
    output_dir: str | Path,
    circuit_csv: str | Path = "circuit.csv",
) -> Path:
    if task not in ("navigation2d", "racing"):
        raise ValueError("task must be one of: navigation2d, racing")

    cfg = get_task_config(task=task, profile=profile, output_dir=Path(output_dir) / task)
    set_global_seed(cfg.seed)

    task_dir = Path(cfg.output_dir)
    csv_dir = task_dir / "csv"
    ckpt_dir = task_dir / "checkpoints" / f"{task}_{profile}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if task == "navigation2d":
        x, y = collect_training_data_navigation(cfg)
    else:
        x, y = collect_training_data_racing(cfg, circuit_csv=circuit_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts, history = train_transformer_model(
        input_sequences=x,
        target_sequences=y,
        horizon=cfg.horizon,
        k_history=cfg.k_history,
        hidden_size=cfg.transformer.hidden_size,
        num_layers=cfg.transformer.num_layers,
        nhead=cfg.transformer.nhead,
        dropout=cfg.transformer.dropout,
        batch_size=cfg.training.batch_size,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        learning_rate=cfg.training.learning_rate,
        val_fraction=cfg.training.val_fraction,
        seed=cfg.seed,
        device=device,
    )
    artifacts.save(ckpt_dir)

    pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.train_losses) + 1),
            "train_loss": history.train_losses,
            "val_loss": history.val_losses,
        }
    ).to_csv(task_dir / "training_history.csv", index=False)

    _export_prediction_csvs(task=task, artifacts=artifacts, input_sequences=x, target_sequences=y, out_csv_dir=csv_dir)

    if task == "navigation2d":
        _run_sample_sweep_navigation(cfg, artifacts=artifacts, out_csv_dir=csv_dir)
        _run_dynamic_sweep_navigation(cfg, artifacts=artifacts, out_csv_dir=csv_dir)
    else:
        _run_sample_sweep_racing(cfg, artifacts=artifacts, out_csv_dir=csv_dir, circuit_csv=circuit_csv)
        _run_dynamic_sweep_racing(cfg, artifacts=artifacts, out_csv_dir=csv_dir, circuit_csv=circuit_csv)

    return task_dir


def run_reproduction(
    task: str,
    profile: str,
    output_dir: str | Path,
    circuit_csv: str | Path = "circuit.csv",
) -> list[Path]:
    if task == "both":
        tasks = ["navigation2d", "racing"]
    else:
        tasks = [task]

    results = []
    for task_name in tasks:
        results.append(run_reproduction_task(task=task_name, profile=profile, output_dir=output_dir, circuit_csv=circuit_csv))
    return results
