from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np



class Side(Enum):
    Buy  = "Buy"
    Sell = "Sell"


class OptionType(Enum):
    Call = "Call"
    Put  = "Put"


@dataclass
class Option(ABC):
    
    strike: float
    expiry: float  # TTE in years
    option_type: OptionType

    @abstractmethod
    def payoff(self, spot: float) -> float:
        return 
    

class EuropeanOption(Option): 

    def payoff(self, spot): 
        if self.option_type == OptionType.Call:
            return max(spot - self.strike, 0)
        return max(self.strike - spot, 0)


class MultiLookOption(Option):    
    pass
    
    
@dataclass
class AsianOption(MultiLookOption): 

    n_fixings : int   # defines the no. of fixings that are used in the payoff calculation

    def payoff(self, spot_path: np.array):
        
        n_points = len(spot_path)
        
        if n_points < self.n_fixings:
            raise ValueError("Spot path has fewer points than required fixings")
        
        indices = np.linspace(0, n_points-1, self.n_fixings, dtype=int)
        fixings = spot_path[indices]
        avg_spot = np.average(fixings)
        
        if self.option_type == OptionType.Call:
            return np.maximum(avg_spot - self.strike, 0)
        return np.maximum(self.strike - avg_spot, 0)
        