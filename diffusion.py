from abc import ABC, abstractmethod
import numpy as np


class DiffusionProcess(ABC): 

    def __init__(self):
        return

    @abstractmethod
    def simulate_path(self, n_steps: int):
        raise NotImplementedError


class GBM(DiffusionProcess):
    def __init__(self, s0: float, mu: float, sigma: float, seed=None):
        if s0 <= 0:
            raise ValueError("s0 must be > 0")
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def simulate_path(self, T: float, n_steps: int):
        if T < 0:
            raise ValueError("T must be >= 0")
        if n_steps <= 0:
            raise ValueError("n_steps must be >= 1")

        times = np.linspace(0.0, float(T), n_steps + 1, dtype=float)

        # If there is no time to elapse, just return the flat path at s0.
        if T == 0.0:
            return times, np.full(n_steps + 1, self.s0, dtype=float)

        dt = T / n_steps
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * np.sqrt(dt)

        increments = self.rng.normal(loc=drift, scale=vol, size=n_steps)
        logS = np.empty(n_steps + 1, dtype=float)
        logS[0] = np.log(self.s0)
        logS[1:] = logS[0] + np.cumsum(increments)
        S = np.exp(logS)
        return S
