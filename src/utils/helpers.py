import torch
import numpy as np

__all__ = ["angle_normalize"]

def angle_normalize(x):
    """
    Normalize angle(s) to be between -pi and pi.
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi

