"""Shared repeatability controls. Restart kernels after changing these settings."""
import random
import numpy as np

REPRODUCIBLE = True
SEED = 42


def configure_randomness():
    """Seed Python and NumPy once at the start of a notebook, when enabled."""
    if REPRODUCIBLE:
        random.seed(SEED)
        np.random.seed(SEED)


def configure_torch(torch):
    """Seed PyTorch when the topic-modeling lesson imports it."""
    if REPRODUCIBLE:
        torch.manual_seed(SEED)


def random_state(original=None):
    """Use the shared seed, or the original lesson's estimator setting."""
    return SEED if REPRODUCIBLE else original
