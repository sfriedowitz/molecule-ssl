import numpy as np


class TrainingTracker:
    def __init__(self):
        self.metrics: dict[str, list[tuple(int, float)]] = dict()

    def record_metric(self, name: str, epoch: int, metric: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((epoch, metric))

    def get_metric(self, name: str) -> np.ndarray:
        values = self.metrics.get(name, [])
        return np.array(values)
