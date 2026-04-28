# var_project/var_methods.py
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from .data_reader import PriceHistory
from .portfolio import Portfolio
from .volatility import VolatilityEstimator, SimpleCovariance

class VaRMethod(ABC):
    """
    Abstract Base Class for all Value at Risk calculation methods.
    """
    def __init__(self, price_history: PriceHistory):
        self.price_history = price_history

    @abstractmethod
    def calculate_var(self, portfolio: Portfolio, confidence_level: float, returns: np.ndarray = None) -> float:
        """
        Calculate the Value at Risk.
        
        :param portfolio: The Portfolio object.
        :param confidence_level: The confidence level (e.g., 0.95).
        :param returns: Optional. A specific (T x N) matrix of returns to use. 
                        If None, uses the full history from price_history.
        """
        pass

class HistoricalVaR(VaRMethod):
    """Calculates VaR using the Historical Simulation method."""

    def calculate_var(self, portfolio: Portfolio, confidence_level: float, returns: np.ndarray = None) -> float:
        # Use provided returns window or default to full history
        if returns is None:
            returns = self.price_history.get_returns_matrix()
        
        # Calculate weighted portfolio returns: (T x N) @ (N) -> (T)
        portfolio_returns = returns @ portfolio.weights
        
        # Find the percentile
        alpha = 1.0 - confidence_level
        var_return = np.percentile(portfolio_returns, alpha * 100)
        
        # VaR is the negative of the return (a positive loss number)
        return -var_return * portfolio.total_value

class ParametricVaR(VaRMethod):
    """
    Calculates VaR using the parametric (Variance-Covariance) method.
    """
    
    def __init__(self, price_history: PriceHistory, estimator: VolatilityEstimator = None):
        super().__init__(price_history)
        self.estimator = estimator if estimator else SimpleCovariance()

    def calculate_var(self, portfolio: Portfolio, confidence_level: float, returns: np.ndarray = None) -> float:
        # Use provided returns window or default to full history
        if returns is None:
            returns = self.price_history.get_returns_matrix()

        # Delegate covariance calculation to the estimator
        cov_matrix = self.estimator.calculate_covariance(returns)
        
        # Calculate Portfolio Variance: w^T * Cov * w
        port_variance = portfolio.weights.T @ cov_matrix @ portfolio.weights
        port_std_dev = np.sqrt(port_variance)
        
        # Calculate Z-score and Final VaR
        z_score = norm.ppf(confidence_level)
        return z_score * port_std_dev * portfolio.total_value
