from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_mppi.utils import Array


def _build_positional_encoding(max_len: int, d_model: int, dtype: jnp.dtype) -> Array:
    position = jnp.arange(max_len, dtype=dtype)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=dtype) * (-math.log(10000.0) / d_model))

    pe = jnp.zeros((max_len, d_model), dtype=dtype)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    if d_model % 2 == 1:
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[:-1]))
    else:
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe[None, :, :]


def _build_decoder_mask(batch_size: int, seq_len: int, valid_length: Array | int | None = None) -> Array:
    causal_mask = jnp.asarray(
        nn.make_causal_mask(jnp.ones((batch_size, seq_len), dtype=jnp.bool_)),
        dtype=jnp.bool_,
    )
    if valid_length is None:
        return causal_mask

    valid_length = jnp.asarray(valid_length, dtype=jnp.int32)
    if valid_length.ndim == 0:
        valid_length = jnp.broadcast_to(valid_length, (batch_size,))
    valid_positions = jnp.arange(seq_len)[None, :] < valid_length[:, None]
    query_mask = valid_positions[:, None, :, None]
    key_mask = valid_positions[:, None, None, :]
    return causal_mask & query_mask & key_mask


class PositionalEncoding(nn.Module):
    d_model: int
    dropout: float = 0.1
    max_len: int = 5000

    @nn.compact
    def __call__(self, x: Array, deterministic: bool = True) -> Array:
        pe = _build_positional_encoding(self.max_len, self.d_model, x.dtype)
        x = x + pe[:, : x.shape[1], :]
        return nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)


class _EncoderLayer(nn.Module):
    hidden_size: int
    nhead: int
    dropout: float

    @nn.compact
    def __call__(self, x: Array, deterministic: bool) -> Array:
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout,
            name="self_attention",
        )(x, deterministic=deterministic)
        x = nn.LayerNorm(name="self_attention_norm")(x + nn.Dropout(rate=self.dropout)(attn_out, deterministic=deterministic))

        ff = nn.Dense(self.hidden_size * 2, name="ff_in")(x)
        ff = nn.relu(ff)
        ff = nn.Dropout(rate=self.dropout)(ff, deterministic=deterministic)
        ff = nn.Dense(self.hidden_size, name="ff_out")(ff)
        x = nn.LayerNorm(name="ff_norm")(x + nn.Dropout(rate=self.dropout)(ff, deterministic=deterministic))
        return x


class _DecoderLayer(nn.Module):
    hidden_size: int
    nhead: int
    dropout: float

    @nn.compact
    def __call__(self, x: Array, memory: Array, decoder_mask: Array, deterministic: bool) -> Array:
        self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout,
            name="self_attention",
        )(x, mask=decoder_mask, deterministic=deterministic)
        x = nn.LayerNorm(name="self_attention_norm")(x + nn.Dropout(rate=self.dropout)(self_attn, deterministic=deterministic))

        cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout,
            name="cross_attention",
        )(x, memory, deterministic=deterministic)
        x = nn.LayerNorm(name="cross_attention_norm")(x + nn.Dropout(rate=self.dropout)(cross_attn, deterministic=deterministic))

        ff = nn.Dense(self.hidden_size * 2, name="ff_in")(x)
        ff = nn.relu(ff)
        ff = nn.Dropout(rate=self.dropout)(ff, deterministic=deterministic)
        ff = nn.Dense(self.hidden_size, name="ff_out")(ff)
        x = nn.LayerNorm(name="ff_norm")(x + nn.Dropout(rate=self.dropout)(ff, deterministic=deterministic))
        return x


class TransformerModel(nn.Module):
    input_size: int
    output_size: int
    hidden_size: int = 256
    num_layers: int = 3
    nhead: int = 8
    dropout: float = 0.1
    device: Any = None

    @nn.compact
    def __call__(
        self,
        src: Array,
        tgt: Array,
        deterministic: bool = True,
        tgt_valid_length: Array | int | None = None,
    ) -> Array:
        src = jnp.asarray(src, dtype=jnp.float32)
        tgt = jnp.asarray(tgt, dtype=jnp.float32)

        src_batch = jnp.swapaxes(src, 0, 1)
        tgt_batch = jnp.swapaxes(tgt, 0, 1)

        src_batch = nn.Dense(self.hidden_size, name="src_fc")(src_batch) * math.sqrt(self.hidden_size)
        tgt_batch = nn.Dense(self.hidden_size, name="tgt_fc")(tgt_batch) * math.sqrt(self.hidden_size)

        positional = PositionalEncoding(self.hidden_size, self.dropout, name="positional_encoding")
        encoded = positional(src_batch, deterministic=deterministic)
        decoded = positional(tgt_batch, deterministic=deterministic)

        for layer_idx in range(self.num_layers):
            encoded = _EncoderLayer(
                hidden_size=self.hidden_size,
                nhead=self.nhead,
                dropout=self.dropout,
                name=f"encoder_layer_{layer_idx}",
            )(encoded, deterministic=deterministic)

        decoder_mask = _build_decoder_mask(decoded.shape[0], decoded.shape[1], valid_length=tgt_valid_length)
        for layer_idx in range(self.num_layers):
            decoded = _DecoderLayer(
                hidden_size=self.hidden_size,
                nhead=self.nhead,
                dropout=self.dropout,
                name=f"decoder_layer_{layer_idx}",
            )(decoded, encoded, decoder_mask, deterministic=deterministic)

        output = nn.Dense(self.output_size, name="output_fc")(decoded)
        return jnp.swapaxes(output, 0, 1)

    def init_params(self, rng: Array, src: Array, tgt: Array) -> Any:
        variables = self.init(rng, src, tgt, deterministic=True)
        return variables["params"]

    def apply_model(
        self,
        params: Any,
        src: Array,
        tgt: Array,
        deterministic: bool = True,
        tgt_valid_length: Array | int | None = None,
        dropout_rng: Array | None = None,
    ) -> Array:
        variables = {"params": params}
        if dropout_rng is None:
            return self.apply(
                variables,
                src,
                tgt,
                deterministic=deterministic,
                tgt_valid_length=tgt_valid_length,
            )
        return self.apply(
            variables,
            src,
            tgt,
            deterministic=deterministic,
            tgt_valid_length=tgt_valid_length,
            rngs={"dropout": dropout_rng},
        )

    def predict_autoregressive(
        self,
        params: Any,
        src: Array,
        horizon: int,
        start_token: Array | None = None,
    ) -> Array:
        src = jnp.asarray(src, dtype=jnp.float32)
        batch_size = src.shape[1]
        if start_token is None:
            start_token = jnp.zeros((1, batch_size, self.output_size), dtype=src.dtype)
        else:
            start_token = jnp.asarray(start_token, dtype=src.dtype)

        decoder_buffer = jnp.zeros((horizon, batch_size, self.output_size), dtype=src.dtype)
        decoder_buffer = decoder_buffer.at[0].set(start_token[0])

        def step_fn(buffer: Array, t: Array) -> tuple[Array, Array]:
            output = self.apply_model(
                params=params,
                src=src,
                tgt=buffer,
                deterministic=True,
                tgt_valid_length=t + 1,
            )
            next_token = output[t]
            next_buffer = jax.lax.cond(
                t + 1 < horizon,
                lambda buf: buf.at[t + 1].set(next_token),
                lambda buf: buf,
                buffer,
            )
            return next_buffer, next_token

        _, outputs = jax.lax.scan(step_fn, decoder_buffer, jnp.arange(horizon, dtype=jnp.int32))
        return outputs


__all__ = ["TransformerModel", "PositionalEncoding"]
