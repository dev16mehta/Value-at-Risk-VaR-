# var_project/portfolio.py
"""
Portfolio container that aggregates instruments and calculates delta-adjusted weights.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from .instruments import Instrument, Stock


@dataclass
class Portfolio:
    """
    A portfolio of mixed financial instruments (stocks, options, etc.).

    Handles aggregation of positions by underlying ticker and computes
    delta-adjusted weights for VaR calculations.
    """
    instruments: List[Instrument]
    tickers: List[str] = field(init=False)
    total_value: float = field(init=False, default=0.0)
    weights: np.ndarray = field(init=False)

    def __post_init__(self):
        self.tickers = sorted(list(set(i.ticker for i in self.instruments)))

    def rebalance(self, current_prices: Dict[str, float]):
        """
        Recalculates portfolio value and weights based on current prices.

        For options, this is essential since delta changes with price.
        Weights are delta-adjusted: each position is mapped to its equivalent
        stock exposure (delta * price) for consistent VaR treatment.
        """
        self.total_value = sum(
            instr.current_value(current_prices[instr.ticker])
            for instr in self.instruments
        )

        # Aggregate delta-dollar values by ticker
        delta_values = {t: 0.0 for t in self.tickers}
        for instr in self.instruments:
            price = current_prices[instr.ticker]
            delta_values[instr.ticker] += instr.calculate_delta(price) * price

        # Convert to weight vector (same order as self.tickers)
        self.weights = np.array([
            delta_values[t] / self.total_value if self.total_value > 0 else 0
            for t in self.tickers
        ])
