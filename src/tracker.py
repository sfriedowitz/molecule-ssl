import numpy as np


class MetricTracker:
    def __init__(self):
        self._metrics = []

    def record(self, epoch: int, metrics: dict[str, float | int]):
        self._metrics.append((epoch, metrics))

    def get(self, key: str) -> np.ndarray:
        values = [(epoch, metrics[key]) for epoch, metrics in self._metrics if key in metrics]
        return np.array(values)

    def log(self, keys: list[str]) -> None:
        epoch, metrics = self._metrics[-1]
        msg = f"{epoch}"
        for key in keys:
            if key in metrics:
                value = metrics[key]
                msg += f" | {key} = {value:.3f}"
        print(msg)
