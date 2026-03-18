from __future__ import annotations

import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


def resolve_device(device: jax.Device | str | None = None) -> jax.Device:
    if isinstance(device, jax.Device):
        return device

    if isinstance(device, str):
        platform, _, index_str = device.partition(":")
        devices = jax.devices(platform.lower())
        if not devices:
            raise ValueError(f"No JAX devices available for platform {platform!r}.")
        index = int(index_str) if index_str else 0
        return devices[index]

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []

    if gpu_devices:
        return gpu_devices[0]
    return jax.devices()[0]


def as_dtype(dtype: Any) -> jnp.dtype:
    return jnp.dtype(dtype)


def as_array(x: Any, dtype: Any | None = None, device: jax.Device | str | None = None) -> Array:
    arr = jnp.asarray(x, dtype=as_dtype(dtype) if dtype is not None else None)
    if device is None:
        return arr
    return jax.device_put(arr, resolve_device(device))


def to_numpy(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

