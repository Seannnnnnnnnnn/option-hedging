import numpy as np

from diffusion import DiffusionProcess
from position import OptionsPosition
from pricing_engine import PricingEngine
from strategy import HedgingStrategy



class Backtest:

    def __init__(self, 
                 strategy: HedgingStrategy, 
                 diffusion_process: DiffusionProcess,
                 pricing_engine: PricingEngine,
                 n_steps: int = 252, 
                 initial_cash: float = 0.0,
                 r: float | None = None, 
                 ):
        
        self.diffusion_process = diffusion_process
        self.strategy = strategy
        self.pricing_engine = pricing_engine
        self.n_steps = n_steps
        self.initial_cash = initial_cash
        self.r = r 

    def simulate_strategy(self, T: float, position: OptionsPosition): 
        """
        Simulates the hedging strategy for an options position up to time T.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        position : OptionsPosition
            The options position (instrument + notional).

        Returns
        -------
        dict
            Dictionary containing:
                times: time grid
                S: simulated underlying price path
                deltas: hedge ratios over time
                cash: cash position over time
                portfolio_value: value of hedged portfolio over time
                pnl: final profit/loss
        """
        
            # Ensure we don’t try to hedge beyond expiry
        if position.instrument.expiry < T:
            T = position.instrument.expiry

        pricing_engine = self.pricing_engine

        # Simulate underlying path
        times, S = self.diffusion_process.simulate_path(T, self.n_steps)

        # Pre-allocate arrays
        hedge = np.zeros_like(S)
        cash = np.zeros_like(S)
        portfolio_value = np.zeros_like(S)

        # --- Initial hedge ---
        hedge[0] = self.strategy.target_holding(S[0], times[0], pricing_engine)
        option_value = pricing_engine.price(position.instrument, S[0], times[0]) * position.notional
        stock_value = hedge[0] * S[0]
        cash[0] = self.initial_cash - stock_value
        portfolio_value[0] = stock_value + cash[0] + option_value

        # --- Rebalancing loop ---
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]

            # Accrue interest on cash if r is set
            if self.r is not None:
                cash[i - 1] *= np.exp(self.r * dt)

            tau = max(T - times[i], 1e-12)

            # get the hedge amount 
            if tau <= 1e-12: 
                hedge[i] = 0.0
                option_value = position.notional * position.instrument.payoff(S[i])
            else:

                hedge[i] = self.strategy.target_holding(S[i], times[i], pricing_engine)
                option_value = pricing_engine.price(position.instrument, S[i], times[i]) * position.notional

            # Rebalance: adjust cash for stock purchases/sales
            delta_change = hedge[i] - hedge[i - 1]
            cash[i] = cash[i - 1] - delta_change * S[i]

            # Mark-to-market
            stock_value = hedge[i] * S[i]
            portfolio_value[i] = stock_value + cash[i] + option_value

        # Final PnL
        pnl = portfolio_value[-1]

        return {
            "times": times,
            "S": S,
            "hedge": hedge,
            "cash": cash,
            "portfolio_value": portfolio_value,
            "pnl": pnl
        }


