from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from transformer_mppi.utils import Array, as_array, as_dtype, resolve_device


@dataclass
class _AdamState:
    step: Array
    m: Array
    v: Array


def _make_info(info: dict, t: Array) -> dict:
    step_info = dict(info)
    step_info["t"] = t
    return step_info


def _sanitize_info(info: dict | None, dtype: jnp.dtype) -> dict:
    if not info:
        return {}

    sanitized = {}
    for key, value in info.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_info(value, dtype)
        elif isinstance(value, (jax.Array, np.ndarray, list, tuple, int, float, bool, np.number)):
            sanitized[key] = jnp.asarray(value, dtype=dtype)
        else:
            sanitized[key] = value
    return sanitized


def _rollout_states(
    dynamics: Callable[[Array, Array], Array],
    state: Array,
    action_seqs: Array,
) -> Array:
    batch_size = action_seqs.shape[0]
    initial_state = jnp.broadcast_to(state, (batch_size, state.shape[0]))

    def step_fn(carry: Array, action_t: Array) -> tuple[Array, Array]:
        next_state = dynamics(carry, action_t)
        return next_state, carry

    final_state, state_hist = jax.lax.scan(step_fn, initial_state, jnp.swapaxes(action_seqs, 0, 1))
    state_seq = jnp.concatenate([state_hist, final_state[None, :, :]], axis=0)
    return jnp.swapaxes(state_seq, 0, 1)


def _rollout_costs(
    dynamics: Callable[[Array, Array], Array],
    cost_func: Callable[[Array, Array, dict], Array],
    state: Array,
    action_seqs: Array,
    info: dict,
    dim_control: int,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    batch_size = action_seqs.shape[0]
    horizon = action_seqs.shape[1]
    initial_state = jnp.broadcast_to(state, (batch_size, state.shape[0]))

    def step_fn(carry: Array, xs: tuple[Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        action_t, t = xs
        step_info = _make_info(info, t)
        cost_t = cost_func(carry, action_t, step_info)
        next_state = dynamics(carry, action_t)
        return next_state, (carry, cost_t)

    times = jnp.arange(horizon, dtype=jnp.int32)
    final_state, (state_hist, cost_hist) = jax.lax.scan(step_fn, initial_state, (jnp.swapaxes(action_seqs, 0, 1), times))
    terminal_t = times[-1] if horizon > 0 else jnp.asarray(0, dtype=jnp.int32)
    terminal_info = _make_info(info, terminal_t)
    terminal_action = jnp.zeros((batch_size, dim_control), dtype=dtype)
    terminal_cost = cost_func(final_state, terminal_action, terminal_info)

    state_seq = jnp.concatenate([state_hist, final_state[None, :, :]], axis=0)
    total_cost = jnp.sum(cost_hist, axis=0) + terminal_cost
    return jnp.swapaxes(state_seq, 0, 1), total_cost


def _savitzky_golay_coeffs(window_size: int, poly_order: int, dtype: jnp.dtype) -> Array:
    if window_size % 2 == 0 or window_size <= poly_order:
        raise ValueError("window_size must be odd and > poly_order")

    half_window = (window_size - 1) // 2
    indices = jnp.arange(-half_window, half_window + 1, dtype=dtype)
    a_mat = jnp.vander(indices, N=poly_order + 1, increasing=True)
    pseudo_inverse = jnp.linalg.pinv(a_mat)
    return pseudo_inverse[0]


def _apply_savitzky_golay(y: Array, coeffs: Array) -> Array:
    pad_size = coeffs.shape[0] // 2
    y_padded = jnp.concatenate([jnp.flip(y[:pad_size]), y, jnp.flip(y[-pad_size:])], axis=0)
    return jnp.convolve(y_padded, jnp.flip(coeffs), mode="valid")


def _adam_init(param: Array) -> _AdamState:
    return _AdamState(
        step=jnp.asarray(0, dtype=jnp.int32),
        m=jnp.zeros_like(param),
        v=jnp.zeros_like(param),
    )


def _adam_update(param: Array, grad: Array, state: _AdamState, learning_rate: float) -> tuple[Array, _AdamState]:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    step = state.step + 1
    m = beta1 * state.m + (1.0 - beta1) * grad
    v = beta2 * state.v + (1.0 - beta2) * jnp.square(grad)
    m_hat = m / (1.0 - beta1**step)
    v_hat = v / (1.0 - beta2**step)
    new_param = param - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    return new_param, _AdamState(step=step, m=m, v=v)


class MPPI:
    """Model Predictive Path Integral controller."""

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[Array, Array], Array],
        cost_func: Callable[[Array, Array, dict], Array],
        u_min: Array,
        u_max: Array,
        sigmas: Array,
        lambda_: float,
        auto_lambda: bool = False,
        exploration: float = 0.2,
        use_sg_filter: bool = False,
        sg_window_size: int = 5,
        sg_poly_order: int = 3,
        device: jax.Device | str | None = None,
        dtype: jnp.dtype = jnp.float32,
        seed: int = 42,
    ) -> None:
        self._device = resolve_device(device)
        self._dtype = as_dtype(dtype)
        self._rng = jax.random.PRNGKey(seed)

        self._horizon = horizon
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._cost_func = cost_func
        self._u_min = as_array(u_min, dtype=self._dtype, device=self._device)
        self._u_max = as_array(u_max, dtype=self._dtype, device=self._device)
        self._sigmas = as_array(sigmas, dtype=self._dtype, device=self._device)
        self._lambda = float(lambda_)
        self._auto_lambda = auto_lambda
        self._exploration = exploration
        self._use_sg_filter = use_sg_filter
        self._sg_window_size = sg_window_size
        self._sg_poly_order = sg_poly_order

        self._weights = as_array(jnp.zeros(self._num_samples, dtype=self._dtype), device=self._device)
        self._previous_action_seq = as_array(
            jnp.zeros((self._horizon, self._dim_control), dtype=self._dtype),
            device=self._device,
        )

        self._coeffs = as_array(
            _savitzky_golay_coeffs(self._sg_window_size, self._sg_poly_order, self._dtype),
            device=self._device,
        )
        self._actions_history_for_sg = as_array(
            jnp.zeros((self._horizon - 1, self._dim_control), dtype=self._dtype),
            device=self._device,
        )

        self._log_temperature = None
        self._temperature_state = None
        if self._auto_lambda:
            self._log_temperature = jnp.log(jnp.asarray([self._lambda], dtype=self._dtype))
            self._temperature_state = _adam_init(self._log_temperature)

        self._build_forward_fn()

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def dim_control(self) -> int:
        return self._dim_control

    @property
    def device(self) -> jax.Device:
        return self._device

    def _build_forward_fn(self) -> None:
        threshold = int(self._num_samples * (1.0 - self._exploration))

        def forward_fn(
            rng: Array,
            state: Array,
            info: dict,
            mean_action_seq: Array,
            actions_history_for_sg: Array,
            lambda_value: Array,
        ) -> tuple[Array, Array, Array, Array]:
            rng, sample_key = jax.random.split(rng)
            noises = jax.random.normal(
                sample_key,
                (self._num_samples, self._horizon, self._dim_control),
                dtype=self._dtype,
            ) * self._sigmas

            inherited_samples = mean_action_seq[None, :, :] + noises[:threshold]
            random_samples = noises[threshold:]
            perturbed_action_seqs = jnp.concatenate([inherited_samples, random_samples], axis=0)
            perturbed_action_seqs = jnp.clip(perturbed_action_seqs, self._u_min, self._u_max)

            state_seq_batch, costs = _rollout_costs(
                dynamics=self._dynamics,
                cost_func=self._cost_func,
                state=state,
                action_seqs=perturbed_action_seqs,
                info=info,
                dim_control=self._dim_control,
                dtype=self._dtype,
            )

            weights = jax.nn.softmax(-costs / lambda_value, axis=0)
            optimal_action_seq = jnp.sum(weights[:, None, None] * perturbed_action_seqs, axis=0)

            if self._use_sg_filter:
                prolonged_action_seq = jnp.concatenate([actions_history_for_sg, optimal_action_seq], axis=0)
                filtered_action_seq = jax.vmap(_apply_savitzky_golay, in_axes=(1, None), out_axes=1)(
                    prolonged_action_seq,
                    self._coeffs,
                )
                optimal_action_seq = filtered_action_seq[-self._horizon :]

            optimal_state_seq = _rollout_states(self._dynamics, state, optimal_action_seq[None, :, :])
            first_action = optimal_action_seq[0]
            next_actions_history = jnp.concatenate([actions_history_for_sg[1:], first_action[None, :]], axis=0)
            return optimal_action_seq, optimal_state_seq, weights, costs, rng, next_actions_history

        self._forward_fn = jax.jit(forward_fn)

    def set_cost_function(self, cost_func: Callable[[Array, Array, dict], Array]) -> None:
        self._cost_func = cost_func
        self._build_forward_fn()

    def set_num_samples(self, num_samples: int) -> None:
        self._num_samples = num_samples
        self._weights = as_array(jnp.zeros(self._num_samples, dtype=self._dtype), device=self._device)
        self._build_forward_fn()

    def set_previous_action_seq(self, action_seq: Array) -> None:
        action_seq = as_array(action_seq, dtype=self._dtype, device=self._device)
        if action_seq.shape != (self._horizon, self._dim_control):
            raise ValueError(
                f"Expected shape {(self._horizon, self._dim_control)} for previous action sequence, got {tuple(action_seq.shape)}"
            )
        self._previous_action_seq = jnp.clip(action_seq, self._u_min, self._u_max)

    def forward(
        self,
        state: Array,
        info: dict | None = None,
        mean_action_seq: Array | None = None,
    ) -> tuple[Array, Array]:
        info = _sanitize_info(info, self._dtype)
        state = as_array(state, dtype=self._dtype, device=self._device)

        if mean_action_seq is None:
            mean_action_seq = self._previous_action_seq
        else:
            mean_action_seq = as_array(mean_action_seq, dtype=self._dtype, device=self._device)
            if mean_action_seq.shape != (self._horizon, self._dim_control):
                raise ValueError(
                    f"Expected mean_action_seq shape {(self._horizon, self._dim_control)}, got {tuple(mean_action_seq.shape)}"
                )

        (
            optimal_action_seq,
            optimal_state_seq,
            self._weights,
            costs,
            self._rng,
            self._actions_history_for_sg,
        ) = self._forward_fn(
            self._rng,
            state,
            info,
            mean_action_seq,
            self._actions_history_for_sg,
            jnp.asarray(self._lambda, dtype=self._dtype),
        )

        if self._auto_lambda and self._log_temperature is not None and self._temperature_state is not None:
            def temperature_loss(log_temperature: Array) -> Array:
                temperature = jax.nn.softplus(log_temperature)
                cost_logsumexp = jax.scipy.special.logsumexp(-costs / temperature, axis=0)
                epsilon = 0.1
                return jnp.sum(temperature * (epsilon + jnp.mean(cost_logsumexp)))

            grads = jax.grad(temperature_loss)(self._log_temperature)
            self._log_temperature, self._temperature_state = _adam_update(
                self._log_temperature,
                grads,
                self._temperature_state,
                learning_rate=1e-2,
            )
            self._lambda = float(np.asarray(jnp.exp(self._log_temperature)[0]))

        self._previous_action_seq = optimal_action_seq
        return optimal_action_seq, optimal_state_seq

    def reset(self) -> None:
        self._previous_action_seq = as_array(
            jnp.zeros((self._horizon, self._dim_control), dtype=self._dtype),
            device=self._device,
        )
        self._actions_history_for_sg = as_array(
            jnp.zeros((self._horizon - 1, self._dim_control), dtype=self._dtype),
            device=self._device,
        )


__all__ = ["MPPI"]
