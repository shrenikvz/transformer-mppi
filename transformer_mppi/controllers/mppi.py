from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class MPPI(nn.Module):
    """Model Predictive Path Integral controller."""

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_func: Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float,
        auto_lambda: bool = False,
        exploration: float = 0.2,
        use_sg_filter: bool = False,
        sg_window_size: int = 5,
        sg_poly_order: int = 3,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = device
        self._dtype = dtype
        torch.manual_seed(seed)

        self._horizon = horizon
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._cost_func = cost_func
        self._u_min = u_min.to(self._device, self._dtype)
        self._u_max = u_max.to(self._device, self._dtype)
        self._sigmas = sigmas.to(self._device, self._dtype)
        self._lambda = lambda_
        self._auto_lambda = auto_lambda
        self._exploration = exploration
        self._use_sg_filter = use_sg_filter
        self._sg_window_size = sg_window_size
        self._sg_poly_order = sg_poly_order

        self._covariance = torch.diag(self._sigmas**2).to(self._device, self._dtype)
        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        self._noise_distribution = MultivariateNormal(zero_mean, self._covariance)

        self._state_seq_batch = torch.zeros(
            self._num_samples,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        self._weights = torch.zeros(self._num_samples, device=self._device, dtype=self._dtype)

        if self._auto_lambda:
            self.log_temperature = torch.nn.Parameter(
                torch.log(torch.tensor([self._lambda], device=self._device, dtype=self._dtype))
            )
            self.optimizer = torch.optim.Adam([self.log_temperature], lr=1e-2)

        self._previous_action_seq = torch.zeros(
            self._horizon,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )

        self._coeffs = self._savitzky_golay_coeffs(self._sg_window_size, self._sg_poly_order)
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def dim_control(self) -> int:
        return self._dim_control

    @property
    def device(self) -> torch.device:
        return self._device

    def set_cost_function(self, cost_func: Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]) -> None:
        self._cost_func = cost_func

    def set_previous_action_seq(self, action_seq: torch.Tensor) -> None:
        action_seq = action_seq.to(self._device, self._dtype)
        if action_seq.shape != (self._horizon, self._dim_control):
            raise ValueError(
                f"Expected shape {(self._horizon, self._dim_control)} for previous action sequence, got {tuple(action_seq.shape)}"
            )
        self._previous_action_seq = torch.clamp(action_seq, self._u_min, self._u_max)

    def forward(
        self,
        state: torch.Tensor,
        info: dict | None = None,
        mean_action_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if info is None:
            info = {}

        if state.device != self._device or state.dtype != self._dtype:
            state = state.to(self._device, self._dtype)

        if mean_action_seq is None:
            mean_action_seq = self._previous_action_seq.clone().detach()
        else:
            mean_action_seq = mean_action_seq.to(self._device, self._dtype)
            if mean_action_seq.shape != (self._horizon, self._dim_control):
                raise ValueError(
                    f"Expected mean_action_seq shape {(self._horizon, self._dim_control)}, got {tuple(mean_action_seq.shape)}"
                )

        noises = self._noise_distribution.rsample(sample_shape=torch.Size([self._num_samples, self._horizon]))

        threshold = int(self._num_samples * (1 - self._exploration))
        inherited_samples = mean_action_seq + noises[:threshold]
        random_samples = noises[threshold:]
        perturbed_action_seqs = torch.cat([inherited_samples, random_samples], dim=0)
        perturbed_action_seqs = torch.clamp(perturbed_action_seqs, self._u_min, self._u_max)

        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)
        for t in range(self._horizon):
            self._state_seq_batch[:, t + 1, :] = self._dynamics(
                self._state_seq_batch[:, t, :],
                perturbed_action_seqs[:, t, :],
            )

        horizon_costs = torch.zeros(self._num_samples, self._horizon, device=self._device, dtype=self._dtype)
        for t in range(self._horizon):
            info["t"] = t
            horizon_costs[:, t] = self._cost_func(
                self._state_seq_batch[:, t, :],
                perturbed_action_seqs[:, t, :],
                info,
            )

        zero_action = torch.zeros(self._num_samples, self._dim_control, device=self._device, dtype=self._dtype)
        terminal_costs = self._cost_func(self._state_seq_batch[:, -1, :], zero_action, info)
        costs = torch.sum(horizon_costs, dim=1) + terminal_costs

        self._weights = torch.softmax(-costs / self._lambda, dim=0)
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * perturbed_action_seqs,
            dim=0,
        )

        if self._auto_lambda:
            self.optimizer.zero_grad()
            temperature = torch.nn.functional.softplus(self.log_temperature)
            cost_logsumexp = torch.logsumexp(-costs / temperature, dim=0)
            epsilon = 0.1
            loss = temperature * (epsilon + torch.mean(cost_logsumexp))
            loss.backward()
            self.optimizer.step()
            self._lambda = torch.exp(self.log_temperature).item()

        if self._use_sg_filter:
            prolonged_action_seq = torch.cat([self._actions_history_for_sg, optimal_action_seq], dim=0)
            filtered_action_seq = torch.zeros_like(prolonged_action_seq)
            for i in range(self._dim_control):
                filtered_action_seq[:, i] = self._apply_savitzky_golay(prolonged_action_seq[:, i], self._coeffs)
            optimal_action_seq = filtered_action_seq[-self._horizon :]

        optimal_state_seq = self._states_prediction(state, optimal_action_seq.unsqueeze(0))

        self._previous_action_seq = optimal_action_seq
        first_action = optimal_action_seq[0]
        self._actions_history_for_sg = torch.cat([self._actions_history_for_sg[1:], first_action.view(1, -1)])

        return optimal_action_seq, optimal_state_seq

    def reset(self) -> None:
        self._previous_action_seq = torch.zeros(
            self._horizon,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )

    def _states_prediction(self, state: torch.Tensor, action_seqs: torch.Tensor) -> torch.Tensor:
        batch_size = action_seqs.shape[0]
        state_seqs = torch.zeros(
            batch_size,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        state_seqs[:, 0, :] = state
        for t in range(self._horizon):
            state_seqs[:, t + 1, :] = self._dynamics(state_seqs[:, t, :], action_seqs[:, t, :])
        return state_seqs

    def _savitzky_golay_coeffs(self, window_size: int, poly_order: int) -> torch.Tensor:
        if window_size % 2 == 0 or window_size <= poly_order:
            raise ValueError("window_size must be odd and > poly_order")

        half_window = (window_size - 1) // 2
        indices = torch.arange(-half_window, half_window + 1, dtype=self._dtype, device=self._device)
        a_mat = torch.vander(indices, n=poly_order + 1, increasing=True)
        pseudo_inverse = torch.linalg.pinv(a_mat)
        return pseudo_inverse[0]

    def _apply_savitzky_golay(self, y: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        pad_size = len(coeffs) // 2
        y_padded = torch.cat([y[:pad_size].flip(0), y, y[-pad_size:].flip(0)])
        y_filtered = torch.conv1d(y_padded.view(1, 1, -1), coeffs.view(1, 1, -1), padding="valid")
        return y_filtered.view(-1)


__all__ = ["MPPI"]
