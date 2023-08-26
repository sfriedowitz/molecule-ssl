from typing import Optional
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def get_fitted_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    bounds: torch.Tensor,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
):
    x_train = normalize(x_train, bounds)
    y_train = y_train.view(y_train.shape[0], -1)
    model = SingleTaskGP(
        train_X=x_train, train_Y=y_train, outcome_transform=Standardize(m=y_train.shape[-1])
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(x_train)
    fit_gpytorch_mll(mll)

    return model
