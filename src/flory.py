from typing import Optional

import torch
from botorch.test_functions.base import BaseTestProblem


class FloryHuggins(BaseTestProblem):
    """A Flory-Huggins free energy model with an additional cubic self-interaction term."""

    def __init__(self, n_components: int, noise: Optional[float] = None):
        self.dim = n_components
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise)

        self.sizes = torch.ones(self.dim)
        self.pairwise = torch.zeros(self.dim, self.dim)
        self.cubic = torch.zeros(self.dim)

    def set_size(self, i: int, size: float) -> None:
        self.sizes[i] = size

    def set_pairwise(self, i: int, j: int, coef: float) -> None:
        self.pairwise[i, j] = coef
        self.pairwise[j, i] = coef

    def set_cubic(self, i: int, coef: float) -> None:
        self.cubic[i] = coef

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        entropy = torch.sum((X / self.sizes) * torch.log(X + 1e-16), dim=-1)
        pairwise = 0.5 * torch.sum((X @ self.pairwise) * X, dim=-1)
        cubic = torch.sum(self.cubic * X**3, dim=-1)
        return entropy + pairwise + cubic
