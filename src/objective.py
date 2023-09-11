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


class VAETestProblem(BaseTestProblem):
    min_objective_value = -100.0

    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        dim: int,
        domain_size: float,
        noise_std: Optional[float] = None,
    ):
        self.dim = dim
        self._bounds = [(-domain_size, domain_size) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std)
        self.vae = vae
        self.selfies_encoder = selfies_encoder

    def get_molecule(self, z: torch.Tensor):
        x = self.vae.decode(z.unsqueeze(0))[0]
        s = self.selfies_encoder.decode_tensor(x)
        return Chem.MolFromSmiles(sf.decoder(s))


class SingleMoleculeProblem(VAETestProblem):
    def __init__(
        self,
        objective: Callable[[Any], float],
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        domain_size: float,
        noise_std: Optional[float] = None,
    ):
        super().__init__(
            vae,
            selfies_encoder,
            dim=vae.latent_size,
            domain_size=domain_size,
            noise_std=noise_std,
        )
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
                obj = self.min_objective_value
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
        noise_std: Optional[float] = None,
    ):
        super().__init__(molecules.penalized_logp, vae, selfies_encoder, domain_size, noise_std)


class PenalizedNP(SingleMoleculeProblem):
    """Evaluation of the penalized-LogP metric for a single molecule in latent space."""

    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        domain_size: float = 5.0,
        noise_std: Optional[float] = None,
    ):
        super().__init__(molecules.penalized_np, vae, selfies_encoder, domain_size, noise_std)


class WaterOctanolMixture(VAETestProblem):
    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        *,
        domain_size: float = 5.0,
        sas_scale: float = 0.0,
        kmax: float = 1e4,
        noise_std: Optional[float] = None,
    ):
        super().__init__(
            vae,
            selfies_encoder,
            dim=vae.latent_size + 3,
            domain_size=domain_size,
            noise_std=noise_std,
        )
        self.sas_scale = sas_scale
        self.kmax = kmax

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        objective_values = []
        for xi in X:
            x_comp, mol = self.get_scaled_candidate(xi)
            x_water, x_octanol, x_mol = x_comp
            try:
                logp = Chem.Crippen.MolLogP(mol)
                sas = sascorer.calculateScore(mol)
                if math.isnan(logp) or math.isnan(sas):
                    raise ValueError("Unable to compute objective for molecule")
                if x_water > x_octanol:
                    # Water > octanol, prefer low logP for partition into water phase
                    K = min(10 ** (-logp), self.kmax)
                    f_comp = max(x_water - 0.5, 0.0) * x_mol * K
                else:
                    # Octanol > water, prefer high logP for partition into octanol phase
                    K = min(10**logp, self.kmax)
                    f_comp = max(x_octanol - 0.5, 0.0) * x_mol * K
                objective = f_comp - self.sas_scale * sas
            except:
                objective = self.min_objective_value

            objective_values.append(objective)

        return torch.tensor(objective_values).unsqueeze(-1)

    def get_scaled_candidate(self, x: torch.Tensor):
        x_comp = x[:3]
        x_comp = F.softmax(x_comp).tolist()

        z_latent = x[3:]
        mol = self.get_molecule(z_latent)

        return x_comp, mol
