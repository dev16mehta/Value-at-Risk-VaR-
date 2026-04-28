# var_project/instruments.py
"""
Financial instrument definitions with Black-Scholes pricing for options.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm


class Instrument(ABC):
    """Abstract base class for financial instruments."""

    @abstractmethod
    def current_value(self, current_price: float) -> float:
        """Returns the total dollar value of the position."""
        pass

    @abstractmethod
    def calculate_delta(self, current_price: float) -> float:
        """Returns the position's sensitivity to a $1 change in the underlying."""
        pass


@dataclass
class Stock(Instrument):
    ticker: str
    shares: float

    def current_value(self, current_price: float) -> float:
        return self.shares * current_price

    def calculate_delta(self, current_price: float) -> float:
        return self.shares


@dataclass
class CallOption(Instrument):
    """
    European call option using Black-Scholes pricing.
    Each contract represents 100 shares of the underlying.
    """
    ticker: str
    strikes: float
    expiry_years: float
    risk_free_rate: float = 0.05
    volatility: float = 0.20
    contracts: float = 1.0

    def _d1(self, S):
        K, T, r, sigma = self.strikes, self.expiry_years, self.risk_free_rate, self.volatility
        if T <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def _d2(self, S):
        if self.expiry_years <= 0:
            return 0
        return self._d1(S) - self.volatility * np.sqrt(self.expiry_years)

    def current_value(self, current_price: float) -> float:
        """Black-Scholes call price * 100 shares * number of contracts."""
        S, K, T, r = current_price, self.strikes, self.expiry_years, self.risk_free_rate
        d1, d2 = self._d1(S), self._d2(S)
        call_price_per_share = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        return call_price_per_share * self.contracts * 100

    def calculate_delta(self, current_price: float) -> float:
        """Position delta = N(d1) * 100 shares * number of contracts."""
        delta_per_share = norm.cdf(self._d1(current_price))
        return delta_per_share * self.contracts * 100
