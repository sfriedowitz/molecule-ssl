from typing import Optional
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition import qExpectedImprovement
from botorch.test_functions.base import BaseTestProblem
from gpytorch.mlls import ExactMarginalLogLikelihood


class CandidateGeneration:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x.clone()
        self.y = y.clone()
        self._best_idx = self.y.argmax()

    @property
    def best_x(self):
        return self.x[self._best_idx]

    @property
    def best_y(self):
        return self.y[self._best_idx].item()


def generate_initial_data(n: int, problem: BaseTestProblem, *, gen=None):
    z = torch.randn(n, problem.dim, generator=gen)
    y = problem(z)
    return z, y


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


def optimize_and_evaluate(acqf, problem, *, batch_size=5, num_restarts=10, raw_samples=100):
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=torch.stack([torch.zeros(problem.dim), torch.ones(problem.dim)]),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        return_best_only=True,
    )
    new_z = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_z)
    finite_obj = torch.isfinite(new_obj).squeeze()
    return new_z[finite_obj], new_obj[finite_obj]


def optimize_sequential(
    problem: BaseTestProblem,
    *,
    n_generations: int,
    n_init: int,
    batch_size: int,
    seed: Optional[int] = None,
    print_every: Optional[int] = None,
) -> list[CandidateGeneration]:
    if seed is not None:
        torch.manual_seed(seed)

    state_dict = None
    train_x, train_obj = generate_initial_data(n_init, problem)
    generations = [CandidateGeneration(train_x, train_obj)]
    for n in range(1, n_generations + 1):
        # get objective
        gp = get_fitted_model(train_x, train_obj, problem.bounds, state_dict=state_dict)
        acqf = qExpectedImprovement(model=gp, best_f=train_obj.max())

        # optimize and get new observation
        candidate_x, candidate_obj = optimize_and_evaluate(acqf, problem, batch_size=batch_size)
        candidate_x = candidate_x.view(batch_size, -1)
        candidate_obj = candidate_obj.view(batch_size, -1)

        # update training points
        train_x = torch.cat((train_x, candidate_x))
        train_obj = torch.cat((train_obj, candidate_obj))

        # store progress
        state_dict = gp.state_dict()
        generations.append(CandidateGeneration(train_x, train_obj))

        if print_every is not None and (n % print_every == 0 or n == 1):
            best_y = max(gen.best_y for gen in generations)
            print(f"Generation = {n}, best objective = {best_y:.3f}")

    return generations


def run_sequential_trials(
    problem: BaseTestProblem,
    *,
    n_trials: int,
    n_generations: int,
    n_init: int,
    batch_size: int,
    seed=None,
):
    best_obj_trial, obj_improvement_trial = [], []
    for n in range(n_trials):
        print(f"Running BO trial {n+1}/{n_trials}")
        generations = optimize_sequential(
            problem, n_generations=n_generations, n_init=n_init, batch_size=batch_size, seed=seed
        )

        best_obj_trj = [g.best_y for g in generations]
        init_obj, final_obj = best_obj_trj[0], best_obj_trj[-1]
        best_obj_trial.append(final_obj)

        obj_improvement = final_obj - init_obj
        obj_improvement_trial.append(obj_improvement)

        print(
            f"Completed trial with best objective = {final_obj:.3f}, improvement = {obj_improvement:.3f}"
        )

    return best_obj_trial, obj_improvement_trial
