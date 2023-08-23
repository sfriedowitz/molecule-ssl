from typing import Optional
import selfies as sf
from rdkit.Chem import Chem

import torch
from botorch.test_functions.base import BaseTestProblem

from src.molecules import penalized_logp
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
        logp_values = [penalized_logp(m) for m in mols]
        return torch.tensor(logp_values)
