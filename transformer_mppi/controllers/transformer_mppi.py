from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from transformer_mppi.controllers.mppi import MPPI
from transformer_mppi.training.artifacts import TransformerArtifacts
from transformer_mppi.utils import Array, as_array


class TransformerMPPIController:
    """MPPI controller initialized by a learned transformer mean control sequence."""

    def __init__(self, mppi: MPPI, artifacts: TransformerArtifacts) -> None:
        self.mppi = mppi
        self.artifacts = artifacts
        self._history: deque[np.ndarray] = deque(maxlen=self.artifacts.k_history)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[Array, Array], Array],
        cost_func: Callable[[Array, Array, dict], Array],
        u_min: Array,
        u_max: Array,
        sigmas: Array,
        lambda_: float = 1.0,
        exploration: float = 0.2,
        use_sg_filter: bool = False,
        device: jax.Device | str | None = None,
        dtype: Any = jnp.float32,
    ) -> "TransformerMPPIController":
        artifacts = TransformerArtifacts.load(checkpoint_dir, device=device)
        mppi = MPPI(
            horizon=artifacts.horizon,
            num_samples=100,
            dim_state=dim_state,
            dim_control=dim_control,
            dynamics=dynamics,
            cost_func=cost_func,
            u_min=u_min,
            u_max=u_max,
            sigmas=sigmas,
            lambda_=lambda_,
            exploration=exploration,
            use_sg_filter=use_sg_filter,
            device=device,
            dtype=dtype,
        )
        return cls(mppi=mppi, artifacts=artifacts)

    def reset(self) -> None:
        self._history.clear()
        self.mppi.reset()

    def set_num_samples(self, num_samples: int) -> None:
        self.mppi.set_num_samples(num_samples)

    def update_history(self, state: np.ndarray, context: np.ndarray) -> None:
        feature = np.concatenate([state, context]).astype(np.float64)
        if len(self._history) == 0:
            for _ in range(self.artifacts.k_history):
                self._history.append(feature.copy())
        else:
            self._history.append(feature)

    def _get_src_sequence(self) -> np.ndarray:
        if len(self._history) != self.artifacts.k_history:
            raise RuntimeError("Controller history is not initialized. Call update_history first.")
        src = np.stack(list(self._history), axis=0)
        if src.shape[1] != self.artifacts.input_size:
            raise ValueError(
                f"Input feature size mismatch. expected {self.artifacts.input_size}, got {src.shape[1]}"
            )
        return src

    def predict_mean_action_seq(self) -> Array:
        src = self._get_src_sequence()
        mean_seq_np = self.artifacts.predict_action_sequence(src_sequence=src, device=self.mppi.device)
        return as_array(mean_seq_np, device=self.mppi.device, dtype=self.mppi._dtype)

    def act(self, state: Array, info: dict | None = None) -> tuple[Array, Array, Array]:
        mean_action_seq = self.predict_mean_action_seq()
        action_seq, state_seq = self.mppi.forward(state=state, info=info or {}, mean_action_seq=mean_action_seq)
        action = action_seq[0, :]
        return action, action_seq, state_seq


__all__ = ["TransformerMPPIController"]
