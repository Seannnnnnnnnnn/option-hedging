import numpy as np

from diffusion import DiffusionProcess
from pricing_engine import PricingEngine
from option import MultiLookOption


class MonteCarloEngine(PricingEngine):
    
    def __init__(self, diffusion: DiffusionProcess): 
        self.diffusion = diffusion
        self.paths = []
    
    def price(self, option: MultiLookOption, r: float, n_paths: int, n_steps: int) -> float:
        
        if not isinstance(option, MultiLookOption):
            raise ValueError("Multilook option expected")
        
        self.paths = [self.diffusion.simulate_path(T=option.expiry, n_steps=n_steps) for _ in range(n_paths)]
        payoffs    = [option.payoff(path) for path in self.paths]
        T = option.expiry
        
        return np.exp(-r * T) * np.average(payoffs)
    