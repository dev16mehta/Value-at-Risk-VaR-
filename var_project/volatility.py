# var_project/volatility.py
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize 

class VolatilityEstimator(ABC):
    @abstractmethod
    def calculate_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculates the covariance matrix of the returns.
        :param returns: A (T x N) numpy array of returns.
        :return: A (N x N) covariance matrix.
        """
        pass

class SimpleCovariance(VolatilityEstimator):
    """
    Standard historical covariance using equal weights.
    """
    def calculate_covariance(self, returns: np.ndarray) -> np.ndarray:
        # rowvar=False because rows are days, cols are assets
        return np.cov(returns, rowvar=False)

class EWMACovariance(VolatilityEstimator):
    """
    Exponentially Weighted Moving Average (EWMA) covariance.
    Gives more weight to recent observations.
    """
    def __init__(self, lambda_param: float = 0.94):
        self.lambda_param = lambda_param

    def calculate_covariance(self, returns: np.ndarray) -> np.ndarray:
        T, N = returns.shape # T = Time steps, N = Assets
        
        # 1. Generate weights
        # We want the most recent observation (index T-1) to have the highest weight.
        weights = np.empty(T)
        for t in range(T):
            age = T - 1 - t
            weights[t] = self.lambda_param ** age
            
        # 2. Use numpy's cov function with 'aweights'
        cov_matrix = np.cov(returns, rowvar=False, aweights=weights)
        
        return cov_matrix

class GarchCovariance(VolatilityEstimator):
    """
    Implements a Multivariate GARCH approach using Constant Conditional Correlation (CCC).
    
    Methodology:
    1. Fit a Univariate GARCH(1,1) model to each asset individually to estimate 
       its specific volatility for the next time step (T+1).
    2. Calculate the historical correlation matrix (R) of the returns.
    3. Reconstruct the Covariance Matrix: Sigma = D * R * D
       where D is the diagonal matrix of the predicted volatilities.
       
    Model Equation: sigma^2_t = omega + alpha * r_{t-1}^2 + beta * sigma^2_{t-1}
    """
    def calculate_covariance(self, returns: np.ndarray) -> np.ndarray:
        T, N = returns.shape
        predicted_volatilities = np.zeros(N)

        # 1. Fit GARCH(1,1) for each asset
        for i in range(N):
            asset_returns = returns[:, i]
            
            # SCALING: Optimizers struggle with tiny numbers (e.g. 0.0001). 
            # We scale returns by 100 (into percentages) for stability, then scale back.
            scale = 100.0
            scaled_returns = asset_returns * scale
            
            # Initial guess: [omega, alpha, beta]
            # Typical values: omega small, alpha~0.05, beta~0.90
            initial_params = [0.05, 0.05, 0.90]
            
            # Bounds: All params > 0. Alpha + Beta < 1 (Stationarity constraint)
            bounds = ((1e-6, None), (1e-6, 1.0), (1e-6, 1.0))
            
            # Optimize Maximum Likelihood
            try:
                res = minimize(
                    self._garch_neg_log_likelihood, 
                    initial_params, 
                    args=(scaled_returns,),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                omega, alpha, beta = res.x
            except:
                # Fallback if optimization fails (rare): use sample variance
                predicted_volatilities[i] = np.std(asset_returns)
                continue
            
            # 2. Predict Variance for T+1 (Tomorrow)
            # We reconstruct the variance path using the optimal parameters
            variances = self._calculate_variance_path(omega, alpha, beta, scaled_returns)
            
            last_return_sq = scaled_returns[-1]**2
            last_variance = variances[-1]
            
            # GARCH Forecast step
            next_day_variance_scaled = omega + alpha * last_return_sq + beta * last_variance
            
            # Rescale back to original decimal units
            predicted_volatilities[i] = np.sqrt(next_day_variance_scaled) / scale

        # 3. Construct Covariance Matrix (CCC-GARCH)
        # Calculate standard correlation matrix
        correlation_matrix = np.corrcoef(returns, rowvar=False)
        
        # Create Diagonal matrix of volatilities (D)
        D = np.diag(predicted_volatilities)
        
        # Sigma = D @ R @ D
        cov_matrix = D @ correlation_matrix @ D
        
        return cov_matrix

    def _garch_neg_log_likelihood(self, params, returns):
        """
        Calculates negative log-likelihood for GARCH(1,1).
        Minimizing this is equivalent to maximizing likelihood.
        """
        omega, alpha, beta = params
        
        # Stationarity check: alpha + beta must be < 1
        if alpha + beta >= 0.999:
            return 1e10 # Return huge penalty
            
        variances = self._calculate_variance_path(omega, alpha, beta, returns)
        
        # Log Likelihood of Normal Distribution (ignoring constants):
        # Sum( log(sigma^2) + r^2 / sigma^2 )
        # We use a small epsilon to avoid log(0)
        variances = np.maximum(variances, 1e-6)
        log_likelihood = np.sum(np.log(variances) + (returns**2 / variances))
        
        return 0.5 * log_likelihood

    def _calculate_variance_path(self, omega, alpha, beta, returns):
        """Helper to iterate the GARCH recursion for the whole history"""
        T = len(returns)
        variances = np.zeros(T)
        
        # Initialize first variance as the overall sample variance
        variances[0] = np.var(returns)
        
        for t in range(1, T):
            # The Core GARCH Equation
            variances[t] = omega + alpha * (returns[t-1]**2) + beta * variances[t-1]
            
        return variances
