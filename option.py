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
    
    
class AsianOption(MultiLookOption): 
    
    def payoff(self, spot_fixings: List[float]):
        if self.option_type == OptionType.Call:
            return np.maximum(np.average(spot_fixings) - self.strike, 0)
        return np.maximum(self.strike - np.average(spot_fixings), 0)
        