from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from transformer_mppi.controllers.transformer import TransformerModel
from transformer_mppi.training.artifacts import TransformerArtifacts
from transformer_mppi.utils import as_array, resolve_device


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]


def _prepare_batch(
    src_batch: np.ndarray,
    target_batch: np.ndarray,
    device: jax.Device,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    src = as_array(src_batch, dtype=jnp.float32, device=device)
    target = as_array(target_batch, dtype=jnp.float32, device=device)
    tgt_input = jnp.concatenate(
        [
            jnp.zeros((target.shape[0], 1, target.shape[2]), dtype=target.dtype),
            target[:, :-1, :],
        ],
        axis=1,
    )

    return (
        jnp.swapaxes(src, 0, 1),
        jnp.swapaxes(tgt_input, 0, 1),
        jnp.swapaxes(target, 0, 1),
    )


def _batch_indices(num_samples: int, batch_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + batch_size, num_samples)) for start in range(0, num_samples, batch_size)]


def train_transformer_model(
    input_sequences: np.ndarray,
    target_sequences: np.ndarray,
    horizon: int,
    k_history: int,
    hidden_size: int,
    num_layers: int,
    nhead: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    val_fraction: float,
    seed: int,
    device: jax.Device | str | None = None,
) -> tuple[TransformerArtifacts, TrainingHistory]:
    if input_sequences.ndim != 3:
        raise ValueError("input_sequences must have shape (N, k, input_size)")
    if target_sequences.ndim != 3:
        raise ValueError("target_sequences must have shape (N, horizon, output_size)")

    use_device = resolve_device(device)
    input_size = input_sequences.shape[-1]
    output_size = target_sequences.shape[-1]

    x_flat = input_sequences.reshape(-1, input_size)
    y_flat = target_sequences.reshape(-1, output_size)

    n_quantiles_x = min(1000, x_flat.shape[0])
    n_quantiles_y = min(1000, y_flat.shape[0])
    input_scaler = QuantileTransformer(n_quantiles=n_quantiles_x, output_distribution="uniform", random_state=seed)
    output_scaler = QuantileTransformer(n_quantiles=n_quantiles_y, output_distribution="uniform", random_state=seed)

    input_scaler.fit(x_flat)
    output_scaler.fit(y_flat)

    x_scaled = input_scaler.transform(x_flat).reshape(input_sequences.shape)
    y_scaled = output_scaler.transform(y_flat).reshape(target_sequences.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        x_scaled,
        y_scaled,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    model = TransformerModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nhead=nhead,
        dropout=dropout,
        device=use_device,
    )

    init_src = as_array(jnp.zeros((k_history, 1, input_size), dtype=jnp.float32), device=use_device)
    init_tgt = as_array(jnp.zeros((horizon, 1, output_size), dtype=jnp.float32), device=use_device)
    params = model.init_params(jax.random.PRNGKey(seed), init_src, init_tgt)

    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply_model,
        params=params,
        tx=tx,
    )

    def loss_fn(params: Any, src: jax.Array, tgt_input: jax.Array, target: jax.Array, dropout_rng: jax.Array) -> jax.Array:
        pred = model.apply_model(
            params=params,
            src=src,
            tgt=tgt_input,
            deterministic=False,
            dropout_rng=dropout_rng,
        )
        return jnp.mean(optax.huber_loss(pred, target))

    @jax.jit
    def train_step(
        state: train_state.TrainState,
        src: jax.Array,
        tgt_input: jax.Array,
        target: jax.Array,
        dropout_rng: jax.Array,
    ) -> tuple[train_state.TrainState, jax.Array]:
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, src, tgt_input, target, dropout_rng)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_step(
        params: Any,
        src: jax.Array,
        tgt_input: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        pred = model.apply_model(params=params, src=src, tgt=tgt_input, deterministic=True)
        return jnp.mean(optax.huber_loss(pred, target))

    best_val = float("inf")
    best_params = state.params
    epochs_without_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    rng = np.random.default_rng(seed)
    train_batch_ids = _batch_indices(len(x_train), batch_size)
    val_batch_ids = _batch_indices(len(x_val), batch_size)
    dropout_key = jax.random.PRNGKey(seed + 1)

    for _ in range(epochs):
        shuffled_indices = rng.permutation(len(x_train))
        train_loss_sum = 0.0

        for batch_ids in train_batch_ids:
            idx = shuffled_indices[batch_ids]
            src_batch, tgt_input_batch, target_batch = _prepare_batch(x_train[idx], y_train[idx], use_device)
            dropout_key, step_key = jax.random.split(dropout_key)
            state, loss = train_step(state, src_batch, tgt_input_batch, target_batch, step_key)
            train_loss_sum += float(np.asarray(loss)) * len(idx)

        train_loss = train_loss_sum / len(x_train)
        train_losses.append(train_loss)

        val_loss_sum = 0.0
        for batch_ids in val_batch_ids:
            src_batch, tgt_input_batch, target_batch = _prepare_batch(x_val[batch_ids], y_val[batch_ids], use_device)
            loss = eval_step(state.params, src_batch, tgt_input_batch, target_batch)
            val_loss_sum += float(np.asarray(loss)) * len(batch_ids)

        val_loss = val_loss_sum / len(x_val)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_params = state.params
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    artifacts = TransformerArtifacts(
        model=model,
        params=best_params,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        horizon=horizon,
        k_history=k_history,
        input_size=input_size,
        output_size=output_size,
        model_config={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "nhead": nhead,
            "dropout": dropout,
        },
    )
    history = TrainingHistory(train_losses=train_losses, val_losses=val_losses)
    return artifacts, history
