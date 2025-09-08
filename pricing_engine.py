from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

from option import OptionType


class PricingEngine(ABC):
    """
    Abstract base class for any options pricing engine.
    """

    @abstractmethod
    def price(self, option):
        pass

    @abstractmethod
    def delta(self, option, spot, t):
        pass

    @abstractmethod
    def gamma(self, option, spot, t):
        pass

    @abstractmethod
    def vega(self, option, spot, t):
        pass

    @abstractmethod
    def theta(self, option, spot, t):
        pass

    @abstractmethod
    def rho(self, option, spot, t):
        pass


class BlackScholesPricingEngine(PricingEngine):
    """
    Black-Scholes pricing engine for European calls and puts.
    Assumes continuous compounding and no dividends unless specified.
    """

    def __init__(self, r: float, sigma: float, q: float = 0.0):
        """
        :param r: risk-free interest rate
        :param sigma: volatility
        :param q: continuous dividend yield (default 0)
        """
        self.r = r
        self.sigma = sigma
        self.q = q

    def _d1_d2(self, option, spot, t):
        K = option.strike
        T = option.expiry
        tau = T - t
        if tau < 0:
            raise ValueError("Option has expired.")
        d1 = (np.log(spot / K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / \
             (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)
        return d1, d2, tau

    def price(self, option, spot, t):
        d1, d2, tau = self._d1_d2(option, spot, t)
        K = option.strike
        
        if option.option_type == OptionType.Call: 
            return spot * np.exp(-self.q * tau) * norm.cdf(d1) - K * np.exp(-self.r * tau) * norm.cdf(d2)
        
        return K * np.exp(-self.r * tau) * norm.cdf(-d2) - spot * np.exp(-self.q * tau) * norm.cdf(-d1)
        

    def delta(self, option, spot, t):
        d1, _, tau = self._d1_d2(option, spot, t)
        if option.option_type == OptionType.Call:
            return np.exp(-self.q * tau) * norm.cdf(d1)
        return np.exp(-self.q * tau) * (norm.cdf(d1) - 1)

    def gamma(self, option, spot, t):
        d1, _, tau = self._d1_d2(option, spot, t)
        return (np.exp(-self.q * tau) * norm.pdf(d1)) / (spot * self.sigma * np.sqrt(tau))

    def vega(self, option, spot, t):
        d1, _, tau = self._d1_d2(option, spot, t)
        return spot * np.exp(-self.q * tau) * norm.pdf(d1) * np.sqrt(tau)

    def theta(self, option, spot, t):
        d1, d2, tau = self._d1_d2(option, spot, t)
        if option.option_type == OptionType.Call:
            term1 = - (spot * norm.pdf(d1) * self.sigma * np.exp(-self.q * tau)) / (2 * np.sqrt(tau))
            term2 = self.q * spot * np.exp(-self.q * tau) * norm.cdf(d1)
            term3 = - self.r * option.strike * np.exp(-self.r * tau) * norm.cdf(d2)
            return term1 - term2 + term3
        
        term1 = - (spot * norm.pdf(d1) * self.sigma * np.exp(-self.q * tau)) / (2 * np.sqrt(tau))
        term2 = self.q * spot * np.exp(-self.q * tau) * norm.cdf(-d1)
        term3 = self.r * option.strike * np.exp(-self.r * tau) * norm.cdf(-d2)
        return term1 + term2 + term3

    def rho(self, option, spot, t):
        _, d2, tau = self._d1_d2(option, spot, t)
        if option.option_type == OptionType.Call:
            return option.strike * tau * np.exp(-self.r * tau) * norm.cdf(d2)
        return -option.strike * tau * np.exp(-self.r * tau) * norm.cdf(-d2)
