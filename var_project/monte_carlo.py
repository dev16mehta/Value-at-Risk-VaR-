# var_project/monte_carlo.py
"""
Monte Carlo VaR implementation using Cholesky decomposition for correlated asset simulation.
"""
import numpy as np
from .var_methods import VaRMethod
from .portfolio import Portfolio
from .volatility import VolatilityEstimator, SimpleCovariance


class MonteCarloVaR(VaRMethod):
    """
    Calculates VaR using Monte Carlo simulation.

    Uses Cholesky decomposition to generate correlated random returns that
    preserve the historical correlation structure between assets.
    """

    def __init__(self, price_history, estimator: VolatilityEstimator = None,
                 simulations: int = 10000):
        super().__init__(price_history)
        self.estimator = estimator if estimator else SimpleCovariance()
        self.simulations = simulations

    def calculate_var(self, portfolio: Portfolio, confidence_level: float,
                      returns: np.ndarray = None) -> float:
        if returns is None:
            returns = self.price_history.get_returns_matrix()

        cov_matrix = self.estimator.calculate_covariance(returns)

        # Cholesky decomposition: L @ L.T = cov_matrix
        # This lets us transform uncorrelated random normals into correlated ones
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive definite. Monte Carlo failed.")

        # Generate correlated returns: multiply uncorrelated normals by L
        n_assets = len(portfolio.tickers)
        uncorrelated_normals = np.random.normal(0, 1, size=(n_assets, self.simulations))
        simulated_returns = L @ uncorrelated_normals  # Shape: (n_assets, n_sims)

        # Calculate simulated P&L using delta-adjusted dollar exposures
        dollar_exposures = portfolio.weights * portfolio.total_value
        simulated_pnl = np.sum(dollar_exposures.reshape(-1, 1) * simulated_returns, axis=0)

        # VaR is the loss at the (1 - confidence) percentile, expressed as positive
        alpha = 1.0 - confidence_level
        return -np.percentile(simulated_pnl, alpha * 100)
