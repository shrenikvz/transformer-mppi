from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def angle_normalize(x):
    """Normalize angles to [-pi, pi]."""
    x = jnp.asarray(x)
    return ((x + np.pi) % (2 * np.pi)) - np.pi
