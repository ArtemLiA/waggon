import numpy as np

from .base import Function


class Rosenbrock(Function):
    def __init__(self, dim=20):
        super(Rosenbrock, self).__init__()

        self.dim      = dim
        self.domain   = np.tile([-2, 2], reps=(dim, 1))
        self.name     = f"Rosenbrock(dim={self.dim})"
        self.glob_min = np.ones(self.dim)
    
    def __call__(self, x):
        y = np.stack(
            [
                100 * (x[..., i+1] - x[..., i] ** 2) ** 2 + (1 - x[..., i]) ** 2
                for i in range(self.dim - 1)
            ],
            axis=-1
        )
        return np.sum(y, axis=-1, keepdims=True)

    def __repr__(self) -> str:
        return self.name