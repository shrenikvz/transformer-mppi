from __future__ import annotations

import random

import numpy as np
import torch


def angle_normalize(x: torch.Tensor | np.ndarray | float):
    """Normalize angles to [-pi, pi]."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
