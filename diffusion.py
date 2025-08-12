from abc import ABC, abstractmethod

import numpy as np


class DiffusionProcess(ABC): 

    def __init__(self):
        return

    @abstractmethod
    def simulate_path(self, n_steps: int):
        pass


class GBM(DiffusionProcess):
    def __init__(self, s0: float, mu: float, sigma: float, seed: int | None = None):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def simulate_path(self, T: float, n_steps: int):
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        increments = self.rng.normal(
            loc=(self.mu - 0.5 * self.sigma**2) * dt,
            scale=self.sigma * np.sqrt(dt),
            size=n_steps
        )
        logS = np.empty(n_steps + 1)
        logS[0] = np.log(self.s0)
        logS[1:] = logS[0] + np.cumsum(increments)
        S = np.exp(logS)
        return times, S