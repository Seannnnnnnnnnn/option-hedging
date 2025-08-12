from abc import ABC, abstractmethod
from position import OptionsPosition
from pricing_engine import PricingEngine, BlackScholesPricingEngine


class HedgingStrategy(ABC):

    def __init__(self, position: OptionsPosition):
        
        self.option = position.instrument
        self.notional = position.notional

    @abstractmethod
    def target_holding(self) -> float: 
        """Returns how many units of the underlying to purchase to hedge the position"""
        return



class DeltaHedgingStrategy(HedgingStrategy):

    
    def target_holding(self, spot: float, t: float, pricing_engine: PricingEngine = BlackScholesPricingEngine): 
        """
        Returns target amount of shares to hold based on delta from the pricing engine. 
        If None, defaults to Black-Scholes Delta
        """
        delta = pricing_engine.delta(self.option, spot, t)
        return self.notional * delta
