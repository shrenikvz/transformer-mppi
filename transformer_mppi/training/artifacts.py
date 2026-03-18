from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from transformer_mppi.controllers.transformer import TransformerModel
from transformer_mppi.utils import as_array, resolve_device, to_numpy


@dataclass
class TransformerArtifacts:
    model: TransformerModel
    params: Any
    input_scaler: object
    output_scaler: object
    horizon: int
    k_history: int
    input_size: int
    output_size: int
    model_config: dict

    def predict_action_sequence(self, src_sequence: np.ndarray, device: jax.Device | str | None = None) -> np.ndarray:
        if src_sequence.shape != (self.k_history, self.input_size):
            raise ValueError(
                f"Expected input shape {(self.k_history, self.input_size)}, got {tuple(src_sequence.shape)}"
            )

        use_device = resolve_device(device)
        src_scaled = self.input_scaler.transform(src_sequence)
        src_tensor = as_array(src_scaled, dtype=jnp.float32, device=use_device)[:, None, :]

        out_scaled = self.model.predict_autoregressive(self.params, src_tensor, horizon=self.horizon)
        out_scaled_np = to_numpy(out_scaled.squeeze(1))
        out = self.output_scaler.inverse_transform(out_scaled_np)
        return out

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / "model.pt"
        metadata_path = checkpoint_dir / "metadata.json"
        input_scaler_path = checkpoint_dir / "input_scaler.pkl"
        output_scaler_path = checkpoint_dir / "output_scaler.pkl"

        model_path.write_bytes(flax.serialization.to_bytes(self.params))
        metadata = {
            "horizon": self.horizon,
            "k_history": self.k_history,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "model_config": self.model_config,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        with input_scaler_path.open("wb") as f:
            pickle.dump(self.input_scaler, f)
        with output_scaler_path.open("wb") as f:
            pickle.dump(self.output_scaler, f)

    @classmethod
    def load(cls, checkpoint_dir: str | Path, device: jax.Device | str | None = None) -> "TransformerArtifacts":
        checkpoint_dir = Path(checkpoint_dir)

        metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
        with (checkpoint_dir / "input_scaler.pkl").open("rb") as f:
            input_scaler = pickle.load(f)
        with (checkpoint_dir / "output_scaler.pkl").open("rb") as f:
            output_scaler = pickle.load(f)

        use_device = resolve_device(device)
        model = TransformerModel(
            input_size=metadata["input_size"],
            output_size=metadata["output_size"],
            hidden_size=metadata["model_config"]["hidden_size"],
            num_layers=metadata["model_config"]["num_layers"],
            nhead=metadata["model_config"]["nhead"],
            dropout=metadata["model_config"]["dropout"],
            device=use_device,
        )

        dummy_src = as_array(
            jnp.zeros((metadata["k_history"], 1, metadata["input_size"]), dtype=jnp.float32),
            device=use_device,
        )
        dummy_tgt = as_array(
            jnp.zeros((metadata["horizon"], 1, metadata["output_size"]), dtype=jnp.float32),
            device=use_device,
        )
        params = model.init_params(jax.random.PRNGKey(0), dummy_src, dummy_tgt)
        params = flax.serialization.from_bytes(params, (checkpoint_dir / "model.pt").read_bytes())
        params = jax.device_put(params, use_device)

        return cls(
            model=model,
            params=params,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            horizon=metadata["horizon"],
            k_history=metadata["k_history"],
            input_size=metadata["input_size"],
            output_size=metadata["output_size"],
            model_config=metadata["model_config"],
        )
