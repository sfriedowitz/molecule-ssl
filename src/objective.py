from typing import Optional
import selfies as sf
from rdkit import Chem

import torch
from botorch.test_functions.base import BaseTestProblem

from src import molecules
from src.vae import MolecularVAE
from src.selfies import SelfiesEncoder


class PenalizedLogP(BaseTestProblem):
    """Evaluation of the penalized-LogP metric for a single molecule in latent space."""

    def __init__(
        self,
        vae: MolecularVAE,
        selfies_encoder: SelfiesEncoder,
        noise: Optional[float] = None,
    ):
        self.dim = vae.latent_size
        self._bounds = [(-10, 10) for _ in range(self.dim)]
        super().__init__(noise)

        self.vae = vae
        self.selfies_encoder = selfies_encoder

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        decodings = self.vae.decode(X)
        selfies = [self.selfies_encoder.decode_tensor(x) for x in decodings]
        mols = [Chem.MolFromSmiles(sf.decoder(s)) for s in selfies]
        logps = []
        for mol in mols:
            if mol.GetNumAtoms() == 0:
                obj = float("-inf")
            else:
                obj = molecules.penalized_logp(mol)
            logps.append(obj)
        return torch.tensor(logps).unsqueeze(-1)
