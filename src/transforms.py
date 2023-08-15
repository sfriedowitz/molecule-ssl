import torch


class StandardScaler:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self._mean = mean
        self._std = std

    @classmethod
    def build(cls, x: torch.Tensor) -> "StandardScaler":
        x_mean = x.mean(dim=0)
        x_std = x.std(dim=0, keepdim=True)
        x_std = x_std.where(x_std >= 1e-9, torch.full_like(x_std, 1.0))
        return cls(x_mean, x_std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def untransform(self, x: torch.Tensor) -> torch.Tensor:
        return self._mean + self._std * x


class UnitCubeScaler:
    def __init__(self, bounds: torch.Tensor):
        self._bounds = bounds

    @classmethod
    def build(cls, x: torch.Tensor) -> "UnitCubeScaler":
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        bounds = torch.stack((x_min, x_max))
        return cls(bounds)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._bounds[0]) / (self._bounds[1] - self._bounds[0])

    def untransform(self, x: torch.Tensor) -> torch.Tensor:
        return self._bounds[0] + (self._bounds[1] - self._bounds[0]) * x
