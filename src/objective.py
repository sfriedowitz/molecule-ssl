import math
from typing import Any, Callable, Optional
import selfies as sf
from rdkit import Chem

import torch
from botorch.test_functions.base import BaseTestProblem

from src import molecules
from src.vae import MolecularVAE
from src.selfies import SelfiesEncoder


class SingleMoleculeProblem(BaseTestProblem):
    __min_objective_value = -100.0

    def __init__(
        self,
        objective: Callable[[Any], float],
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        bounds: torch.Tensor,
        noise: Optional[float] = None,
    ):
        self.dim = vae.latent_size
        self._bounds = [(x[0].item(), x[1].item()) for x in bounds.T]
        super().__init__(noise)

        self.vae = vae
        self.selfies_encoder = selfies_encoder
        self.objective = objective

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        decodings = self.vae.decode(X)
        selfies = [self.selfies_encoder.decode_tensor(x) for x in decodings]
        objective_values = []
        for s in selfies:
            try:
                mol = Chem.MolFromSmiles(sf.decoder(s))
                obj = self.objective(mol)
                if math.isnan(obj):
                    obj = self.__min_objective_value
            except:
                obj = self.__min_objective_value
            objective_values.append(obj)
        return torch.tensor(objective_values).unsqueeze(-1)


class PenalizedLogP(SingleMoleculeProblem):
    """Evaluation of the penalized-LogP metric for a single molecule in latent space."""

    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        bounds: torch.Tensor,
        noise: Optional[float] = None,
    ):
        super().__init__(molecules.penalized_logp, vae, selfies_encoder, bounds, noise)


class PenalizedNP(SingleMoleculeProblem):
    """Evaluation of the penalized-LogP metric for a single molecule in latent space."""

    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        bounds: torch.Tensor,
        noise: Optional[float] = None,
    ):
        super().__init__(molecules.penalized_np, vae, selfies_encoder, bounds, noise)


class MoleculeInMixture(BaseTestProblem):
    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        n_components: int,
        bounds: torch.Tensor,
        noise: Optional[float] = None,
    ):
        self.n_components = n_components
        self.vae = vae
        self.selfies_encoder = selfies_encoder

        self.dim = vae.latent_size + n_components
        self._bounds = [(x[0].item(), x[1].item()) for x in bounds.T]
        super().__init__(noise)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        pass
