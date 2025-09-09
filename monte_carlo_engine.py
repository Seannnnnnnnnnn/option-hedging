import numpy as np
from scipy.stats import qmc, norm

from diffusion import DiffusionProcess
from pricing_engine import PricingEngine
from option import MultiLookOption


class MonteCarloEngine(PricingEngine):
    
    def __init__(self, diffusion: DiffusionProcess, seed=None): 
        self.paths = []
        self.seed  = seed
        self.diffusion = diffusion
    
    def price(self, option: MultiLookOption, r: float, n_paths: int, n_steps: int, sampler="pseudorandom") -> float:
        
        if sampler not in {"pseudorandom", "sobol"}: 
            raise ValueError(f"Invalid sampler: {sampler}. Must be one of 'pseudorandom', 'sobol'")
        
        if sampler == "sobol":
            return self.price_sobol(option, r, n_paths, n_steps)
        
        return self.price_psuedorandom(option, r, n_paths, n_steps)

    def price_psuedorandom(self, option: MultiLookOption, r: float, n_paths: int, n_steps: int) -> float:
        """Prices option according to standard psuedorandom Monte-Carlo path generation"""

        if not isinstance(option, MultiLookOption):
            raise ValueError("Multilook option expected")
        
        self.paths = [self.diffusion.simulate_path(T=option.expiry, n_steps=n_steps) for _ in range(n_paths)]
        payoffs    = [option.payoff(path) for path in self.paths]
        T = option.expiry
        
        return np.exp(-r * T) * np.average(payoffs)
    
    def price_sobol(self, option: MultiLookOption, r: float, n_paths: int, n_steps: int) -> float:
        """Prices option according to sobol generated paths"""

        T = option.expiry
        payoffs = []

        dim = n_steps
        sampler = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
        u = sampler.random(n_paths)                  # (n_paths, n_steps) uniform [0,1]
        z = norm.ppf(u)                              # inverse CDF -> standard normals

        dt = T / n_steps
        drift = (self.diffusion.mu - 0.5 * self.diffusion.sigma**2) * dt
        vol = self.diffusion.sigma * np.sqrt(dt)

        for i in range(n_paths):
            logS = np.empty(n_steps + 1)
            logS[0] = np.log(self.diffusion.s0)
            logS[1:] = logS[0] + np.cumsum(drift + vol * z[i])
            spot_path = np.exp(logS)
            payoffs.append(option.payoff(spot_path))
        
        return np.exp(-r * T) * np.mean(payoffs)
