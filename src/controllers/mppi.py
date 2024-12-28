import torch
import torch.nn as nn
import math
import numpy as np
from typing import Callable, Dict, Tuple
from torch.distributions.multivariate_normal import MultivariateNormal

class MPPI(nn.Module):
    """
    Model Predictive Path Integral Control (Williams et al., T-RO, 2017).
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_func: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float,
        auto_lambda: bool = False,
        exploration: float = 0.2,
        use_sg_filter: bool = False,
        sg_window_size: int = 5,
        sg_poly_order: int = 3,
        device=None,
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        super().__init__()

        # Set device to GPU if available, else CPU
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = device
        self._dtype = dtype
        torch.manual_seed(seed)

        # Initialize control parameters and constraints
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

        # Define covariance matrix for noise distribution
        self._covariance = torch.diag(self._sigmas**2).to(self._device, self._dtype)
        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        self._noise_distribution = MultivariateNormal(zero_mean, self._covariance)

        # Initialize tensors to store state sequences and weights for samples
        self._state_seq_batch = torch.zeros(
            self._num_samples,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype
        )
        self._weights = torch.zeros(self._num_samples, device=self._device, dtype=self._dtype)

        # Setup for automatic lambda tuning if enabled
        if self._auto_lambda:
            self.log_temperature = torch.nn.Parameter(
                torch.log(torch.tensor([self._lambda], device=self._device, dtype=self._dtype))
            )
            self.optimizer = torch.optim.Adam([self.log_temperature], lr=1e-2)

        # Initialize previous action sequence with zeros
        self._previous_action_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )

        # Initialize Savitzky-Golay filter coefficients and history
        self._coeffs = self._savitzky_golay_coeffs(self._sg_window_size, self._sg_poly_order)
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )

    def forward(self, state: torch.Tensor, info: Dict = {}) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the optimal control problem, returning (optimal_action_seq, optimal_state_seq).
        """
        # Move state tensor to the correct device and dtype if necessary
        if state.device != self._device or state.dtype != self._dtype:
            state = state.to(self._device, self._dtype)

        # Clone previous action sequence to use as the mean for sampling
        mean_action_seq = self._previous_action_seq.clone().detach()

        # Sample noise for perturbing action sequences
        # Shape: (num_samples, horizon, dim_control)
        noises = self._noise_distribution.rsample(sample_shape=torch.Size([self._num_samples, self._horizon]))

        # Implement exploration strategy by combining inherited and random samples
        threshold = int(self._num_samples * (1 - self._exploration))
        inherited_samples = mean_action_seq + noises[:threshold]
        random_samples = noises[threshold:]
        perturbed_action_seqs = torch.cat([inherited_samples, random_samples], dim=0)

        # Clamp action sequences within predefined limits
        perturbed_action_seqs = torch.clamp(perturbed_action_seqs, self._u_min, self._u_max)

        # Initialize the state sequence batch with the current state repeated for all samples
        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)
        for t in range(self._horizon):
            # Propagate states using the dynamics function
            self._state_seq_batch[:, t+1, :] = self._dynamics(
                self._state_seq_batch[:, t, :],
                perturbed_action_seqs[:, t, :]
            )

        # Initialize tensors to store costs
        costs = torch.zeros(self._num_samples, device=self._device, dtype=self._dtype)
        horizon_costs = torch.zeros(self._num_samples, self._horizon, device=self._device, dtype=self._dtype)

        initial_state = self._state_seq_batch[:, 0, :]
        for t in range(self._horizon):
            # Update info dictionary with current timestep
            info["t"] = t
            # Compute cost for each state-action pair at timestep t
            horizon_costs[:, t] = self._cost_func(
                self._state_seq_batch[:, t, :],
                perturbed_action_seqs[:, t, :],
                info
            )

        # Compute terminal costs with zero actions
        zero_action = torch.zeros(self._num_samples, self._dim_control, device=self._device, dtype=self._dtype)
        terminal_costs = self._cost_func(self._state_seq_batch[:, -1, :], zero_action, info)

        # Total cost is the sum of horizon costs and terminal costs
        costs = torch.sum(horizon_costs, dim=1) + terminal_costs

        # Calculate weights using softmax over negative costs scaled by lambda
        self._weights = torch.softmax(-costs / self._lambda, dim=0)

        # Compute the weighted average of perturbed action sequences to get the optimal action sequence
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * perturbed_action_seqs,
            dim=0
        )

        # Automatically tune lambda if enabled
        if self._auto_lambda:
            for _ in range(1):
                self.optimizer.zero_grad()
                temperature = torch.nn.functional.softplus(self.log_temperature)
                cost_logsumexp = torch.logsumexp(-costs / temperature, dim=0)
                epsilon = 0.1
                loss = temperature * (epsilon + torch.mean(cost_logsumexp))
                loss.backward()
                self.optimizer.step()
            self._lambda = torch.exp(self.log_temperature).item()

        # Apply Savitzky-Golay filter to smooth the optimal action sequence if enabled
        if self._use_sg_filter:
            prolonged_action_seq = torch.cat([self._actions_history_for_sg, optimal_action_seq], dim=0)
            filtered_action_seq = torch.zeros_like(prolonged_action_seq)
            for i in range(self._dim_control):
                filtered_action_seq[:, i] = self._apply_savitzky_golay(
                    prolonged_action_seq[:, i], self._coeffs
                )
            optimal_action_seq = filtered_action_seq[-self._horizon:]

        # Predict the final optimal state sequence using the dynamics model
        expanded_optimal_action_seq = optimal_action_seq.unsqueeze(0)
        optimal_state_seq = self._states_prediction(state, expanded_optimal_action_seq)

        # Update the previous action sequence for the next forward pass
        self._previous_action_seq = optimal_action_seq

        # Update actions history for Savitzky-Golay filtering
        first_action = optimal_action_seq[0]
        self._actions_history_for_sg = torch.cat([
            self._actions_history_for_sg[1:], first_action.view(1, -1)
        ])

        return optimal_action_seq, optimal_state_seq

    def reset(self):
        """
        Reset previous actions & filter states.
        """
        # Reset the previous action sequence and actions history used for filtering
        self._previous_action_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )

    def _states_prediction(self, state: torch.Tensor, action_seqs: torch.Tensor):
        """
        Predict state sequences based on the given action sequences using the dynamics model.
        """
        batch_size = action_seqs.shape[0]
        state_dim = self._dim_state
        state_seqs = torch.zeros(
            batch_size, self._horizon + 1, state_dim, device=self._device, dtype=self._dtype
        )
        state_seqs[:, 0, :] = state
        for t in range(self._horizon):
            state_seqs[:, t+1, :] = self._dynamics(state_seqs[:, t, :], action_seqs[:, t, :])
        return state_seqs

    def _savitzky_golay_coeffs(self, window_size: int, poly_order: int) -> torch.Tensor:
        """
        Compute Savitzky-Golay filter coefficients for smoothing.
        
        Args:
            window_size (int): Size of the filter window (must be odd and > poly_order).
            poly_order (int): Order of the polynomial used to fit the samples.
        
        Returns:
            torch.Tensor: Coefficients for the filter.
        """
        if window_size % 2 == 0 or window_size <= poly_order:
            raise ValueError("window_size must be odd and > poly_order.")
        half_window = (window_size - 1) // 2
        indices = torch.arange(-half_window, half_window + 1, dtype=self._dtype, device=self._device)
        A = torch.vander(indices, N=poly_order + 1, increasing=True)
        pseudo_inverse = torch.linalg.pinv(A)
        coeffs = pseudo_inverse[0]
        return coeffs

    def _apply_savitzky_golay(self, y: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply Savitzky-Golay filter to smooth the input tensor.
        
        Args:
            y (torch.Tensor): Input tensor to be smoothed.
            coeffs (torch.Tensor): Savitzky-Golay filter coefficients.
        
        Returns:
            torch.Tensor: Smoothed tensor.
        """
        pad_size = len(coeffs) // 2
        # Reflect padding to handle border effects
        y_padded = torch.cat([y[:pad_size].flip(0), y, y[-pad_size:].flip(0)])
        # Apply convolution with the filter coefficients
        y_filtered = torch.conv1d(
            y_padded.view(1, 1, -1), coeffs.view(1, 1, -1), padding="valid"
        )
        return y_filtered.view(-1)
    

