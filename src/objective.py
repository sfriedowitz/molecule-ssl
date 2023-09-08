import math
from typing import Any, Callable, Optional
import selfies as sf
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

import torch
import torch.nn.functional as F
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
        domain_size: torch.Tensor,
        noise: Optional[float] = None,
    ):
        self.dim = vae.latent_size
        self._bounds = [(-domain_size, domain_size) for _ in range(self.dim)]
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
                    raise ValueError("Unable to compute objective for molecule")
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
        domain_size: float = 5.0,
        noise: Optional[float] = None,
    ):
        super().__init__(molecules.penalized_logp, vae, selfies_encoder, domain_size, noise)


class PenalizedNP(SingleMoleculeProblem):
    """Evaluation of the penalized-LogP metric for a single molecule in latent space."""

    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        domain_size: float = 5.0,
        noise: Optional[float] = None,
    ):
        super().__init__(molecules.penalized_np, vae, selfies_encoder, domain_size, noise)


class WaterOctanolMixture(BaseTestProblem):
    __min_objective_value = -100.0

    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        domain_size: float = 5.0,
        sas_scale: float = 0.0,
        noise: Optional[float] = None,
    ):
        self.dim = vae.latent_size + 3
        self._bounds = [(-domain_size, domain_size) for _ in range(self.dim)]
        super().__init__(noise)

        self.vae = vae
        self.selfies_encoder = selfies_encoder
        self.sas_scale = sas_scale

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        objective_values = []
        for xi in X:
            x_comp = xi[:3]
            x_comp = F.softmax(x_comp).tolist()
            x_water, x_octanol, x_mol = x_comp

            z_latent = xi[3:]
            x_recon = self.vae.decode(z_latent.view(1, -1))
            selfie = self.selfies_encoder.decode_tensor(x_recon[0])
            try:
                mol = Chem.MolFromSmiles(sf.decoder(selfie))
                logp = Chem.Crippen.MolLogP(mol)
                sas = sascorer.calculateScore(mol)
                if math.isnan(logp) or math.isnan(sas):
                    raise ValueError("Unable to compute objective for molecule")

                K = 10**logp
                if x_water > x_octanol:
                    # Water > octanol, prefer low logP for partition into water phase
                    f_comp = max(x_water - 0.5, 0.0) * x_mol * (1 / K)
                else:
                    # Octanol > water, prefer high logP for partition into octanol phase
                    f_comp = max(x_octanol - 0.5, 0.0) * x_mol * K
                objective = f_comp - self.sas_scale * sas
            except:
                objective = self.__min_objective_value

            objective_values.append(objective)

        return torch.tensor(objective_values).unsqueeze(-1)
