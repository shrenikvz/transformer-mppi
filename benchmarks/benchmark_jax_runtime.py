from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from transformer_mppi.config import get_task_config
from transformer_mppi.controllers import MPPI, TransformerModel
from transformer_mppi.environment import Navigation2DEnv, RacingEnv
from transformer_mppi.utils import as_array, resolve_device


def _block(value):
    return jax.block_until_ready(value)


def _benchmark(fn, iterations: int) -> tuple[float, float]:
    start = time.perf_counter()
    result = fn()
    if isinstance(result, tuple):
        _block(result[0])
    else:
        _block(result)
    compile_plus_first = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
        if isinstance(result, tuple):
            _block(result[0])
        else:
            _block(result)
    steady_state = (time.perf_counter() - start) / iterations
    return compile_plus_first, steady_state


def benchmark_navigation(device: jax.Device, iterations: int) -> None:
    cfg = get_task_config("navigation2d", "quick", output_dir=Path("/tmp/transformer_mppi_bench"))
    env = Navigation2DEnv(
        num_obstacles=cfg.num_obstacles,
        obstacle_radius=cfg.obstacle_radius,
        dynamic_obstacles=0,
        map_size=cfg.navigation_map_size,
        start_pos=cfg.navigation_start,
        goal_pos=cfg.navigation_goal,
        device=device,
    )
    sigmas = as_array(cfg.control_sigmas, dtype=env.dtype, device=device)
    mppi = MPPI(
        horizon=cfg.horizon,
        num_samples=cfg.training.mppi_samples,
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
        device=device,
        dtype=env.dtype,
        seed=cfg.seed,
    )
    state = env.reset()

    mppi_compile, mppi_steady = _benchmark(lambda: mppi.forward(state), iterations)
    print(f"navigation_mppi_compile_plus_first_s={mppi_compile:.6f}")
    print(f"navigation_mppi_steady_state_s={mppi_steady:.6f}")

    input_size = 3 + cfg.num_obstacles * 2
    model = TransformerModel(
        input_size=input_size,
        output_size=2,
        hidden_size=cfg.transformer.hidden_size,
        num_layers=cfg.transformer.num_layers,
        nhead=cfg.transformer.nhead,
        dropout=cfg.transformer.dropout,
        device=device,
    )
    src = as_array(jnp.zeros((cfg.k_history, 1, input_size), dtype=jnp.float32), device=device)
    tgt = as_array(jnp.zeros((cfg.horizon, 1, 2), dtype=jnp.float32), device=device)
    params = model.init_params(jax.random.PRNGKey(cfg.seed), src, tgt)

    model_compile, model_steady = _benchmark(
        lambda: model.predict_autoregressive(params=params, src=src, horizon=cfg.horizon),
        iterations,
    )
    print(f"navigation_transformer_compile_plus_first_s={model_compile:.6f}")
    print(f"navigation_transformer_steady_state_s={model_steady:.6f}")


def benchmark_racing(device: jax.Device, iterations: int, circuit_csv: str | Path) -> None:
    cfg = get_task_config("racing", "quick", output_dir=Path("/tmp/transformer_mppi_bench"))
    env = RacingEnv(
        circuit_csv=circuit_csv,
        num_obstacles=cfg.num_obstacles,
        obstacle_radius=cfg.obstacle_radius,
        dynamic_obstacles=0,
        map_size=cfg.racing_map_size,
        line_width=cfg.racing_line_width,
        device=device,
    )
    sigmas = as_array(cfg.control_sigmas, dtype=env.dtype, device=device)
    mppi = MPPI(
        horizon=cfg.horizon,
        num_samples=cfg.training.mppi_samples,
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
        device=device,
        dtype=env.dtype,
        seed=cfg.seed,
    )
    state = env.reset()
    reference_path, _ = env.calc_reference_trajectory(
        state=state,
        cind=0,
        horizon=cfg.horizon,
        lookahead_distance=cfg.racing_lookahead_distance,
        reference_path_interval=cfg.racing_reference_path_interval,
    )
    info = {"ref_path": reference_path}

    mppi_compile, mppi_steady = _benchmark(lambda: mppi.forward(state, info=info), iterations)
    print(f"racing_mppi_compile_plus_first_s={mppi_compile:.6f}")
    print(f"racing_mppi_steady_state_s={mppi_steady:.6f}")

    input_size = 4 + cfg.num_obstacles * 2 + cfg.racing_n_waypoints * 3
    model = TransformerModel(
        input_size=input_size,
        output_size=2,
        hidden_size=cfg.transformer.hidden_size,
        num_layers=cfg.transformer.num_layers,
        nhead=cfg.transformer.nhead,
        dropout=cfg.transformer.dropout,
        device=device,
    )
    src = as_array(jnp.zeros((cfg.k_history, 1, input_size), dtype=jnp.float32), device=device)
    tgt = as_array(jnp.zeros((cfg.horizon, 1, 2), dtype=jnp.float32), device=device)
    params = model.init_params(jax.random.PRNGKey(cfg.seed), src, tgt)

    model_compile, model_steady = _benchmark(
        lambda: model.predict_autoregressive(params=params, src=src, horizon=cfg.horizon),
        iterations,
    )
    print(f"racing_transformer_compile_plus_first_s={model_compile:.6f}")
    print(f"racing_transformer_steady_state_s={model_steady:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JAX MPPI and transformer runtime.")
    parser.add_argument("--task", choices=["navigation2d", "racing", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default=None, help="Optional JAX device spec, e.g. gpu:0 or cpu:0")
    parser.add_argument("--circuit-csv", default="circuit.csv")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device}")

    if args.task in {"navigation2d", "both"}:
        benchmark_navigation(device=device, iterations=args.iterations)
    if args.task in {"racing", "both"}:
        benchmark_racing(device=device, iterations=args.iterations, circuit_csv=args.circuit_csv)


if __name__ == "__main__":
    main()
