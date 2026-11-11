"""
Reproducibility helpers.

Results in this project depend on three independent random streams: the
simulation's own RNG (passed to ElevatorSystemEnv as `seed`), NumPy's global
stream (the agent's epsilon-greedy draw), and PyTorch's (weight initialisation
and minibatch sampling). Seeding only some of them still gives runs that cannot
be reproduced, so every entry point should call set_global_seeds() and also
construct its environment with a seed.
"""
import random

import numpy as np
import torch


def set_global_seeds(seed):
    """Seeds Python, NumPy and PyTorch. Returns the seed, for logging."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
