import os
import sys

import torch
import numpy as np
import scipy


def sascore(mol) -> float:
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

    import sascorer

    return sascorer.calculateScore(mol)


def uniform_simplex(size, *, dim: int) -> torch.Tensor:
    x = scipy.stats.dirichlet.rvs(size=size, alpha=np.ones(dim))
    return torch.from_numpy(x).float()


def slerp(start: torch.Tensor, end: torch.Tensor, t: float):
    omega = torch.arccos(
        torch.clip(
            torch.dot(start / torch.linalg.norm(start), end / torch.linalg.norm(end)), -1, 1
        )
    )
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - t) * start + t * end  # L'Hopital's rule/LERP
    return torch.sin((1.0 - t) * omega) / so * start + torch.sin(t * omega) / so * end
