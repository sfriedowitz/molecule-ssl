import torch
import numpy as np
from scipy.stats import dirichlet
import selfies as sf


def random_simplex(size, *, dim: int) -> torch.Tensor:
    x = dirichlet.rvs(size=size, alpha=np.ones(dim))
    return torch.from_numpy(x).float()
