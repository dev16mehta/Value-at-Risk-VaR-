# var_project/backtester.py
import numpy as np
import pandas as pd
from scipy.stats import chi2
from .var_methods import VaRMethod
from .portfolio import Portfolio
from .data_reader import PriceHistory

class Backtester:
    """
    Performs backtesting of VaR models using a rolling window approach.
    """
    def __init__(self, price_history: PriceHistory, portfolio: Portfolio):
        self.price_history = price_history
        self.portfolio = portfolio
        # Get the full DataFrame to keep dates aligned
        self.returns_df = price_history.get_returns_df()
        self.returns_matrix = price_history.get_returns_matrix()

    def run_backtest(self, var_method: VaRMethod, confidence_level: float, window_size: int = 250):
        """
        Runs a rolling window backtest.
        
        :param var_method: The VaR calculation method to test.
        :param confidence_level: The confidence level used (e.g. 0.95).
        :param window_size: The number of historical days to use for prediction (e.g. 250).
        :return: A dictionary containing test results and statistics.
        """
        num_days = len(self.returns_matrix)
        if num_days <= window_size:
            raise ValueError("Not enough data to run backtest with specified window size.")

        breaches = 0
        total_predictions = 0
        
        # Lists to store timeseries for plotting later
        dates = []
        actual_returns = []
        var_estimates = []

        print(f"Starting backtest (Window={window_size} days, Confidence={confidence_level:.0%})...")

        # Rolling Window Loop
        # We start at 'window_size' and predict the VaR for that day using the previous 'window_size' days.
        for t in range(window_size, num_days):
            # 1. Define the window: returns from [t - window_size] to [t - 1]
            # This is the "past" data available to the model
            historical_window = self.returns_matrix[t-window_size : t]
            
            # 2. Calculate VaR for today (t) using the historical window
            var_value = var_method.calculate_var(self.portfolio, confidence_level, returns=historical_window)
            
            # 3. Get the ACTUAL portfolio return for today (t)
            # Actual return = weighted sum of individual asset returns on day t
            day_asset_returns = self.returns_matrix[t]
            actual_port_return_pct = np.dot(day_asset_returns, self.portfolio.weights)
            actual_pnl = actual_port_return_pct * self.portfolio.total_value

            # 4. Check for Breach
            # VaR is a positive number representing loss. 
            # A breach occurs if Actual PnL < -VaR
            if actual_pnl < -var_value:
                breaches += 1

            # Store data
            dates.append(self.returns_df.index[t])
            actual_returns.append(actual_pnl)
            var_estimates.append(var_value)
            total_predictions += 1

        # Calculate Statistics
        results = self._calculate_statistics(breaches, total_predictions, confidence_level)
        
        # Attach the time series data for plotting/reporting
        results['dates'] = dates
        results['actual_returns'] = actual_returns
        results['var_estimates'] = var_estimates
        
        return results

    def _calculate_statistics(self, breaches: int, total: int, confidence: float):
        """Calculates Kupiec POF test statistics."""
        p_expected = 1.0 - confidence
        p_observed = breaches / total
        
        # Kupiec POF Test (Likelihood Ratio)
        # LR = -2 * ln( ( (1-p)^(N-x) * p^x ) / ( (1-p_hat)^(N-x) * p_hat^x ) )
        # Numerator: probability of x failures under Null Hypothesis (p)
        # Denominator: probability of x failures under Alternative Hypothesis (p_hat)
        
        # Handle edge cases for log calculation
        if breaches == 0:
            lr_stat = -2 * np.log( (1 - p_expected)**total ) # Simplification when x=0
        else:
            numerator = (1 - p_expected)**(total - breaches) * p_expected**breaches
            denominator = (1 - p_observed)**(total - breaches) * p_observed**breaches
            lr_stat = -2 * np.log(numerator / denominator)

        # Critical value from Chi-Squared distribution (1 degree of freedom)
        # Usually 3.841 for 95% confidence in the *test* (alpha=0.05)
        p_value = 1 - chi2.cdf(lr_stat, df=1)
        
        decision = "ACCEPT" if p_value > 0.05 else "REJECT"

        return {
            "total_days": total,
            "breaches": breaches,
            "breach_rate": p_observed,
            "expected_rate": p_expected,
            "kupiec_lr": lr_stat,
            "p_value": p_value,
            "decision": decision
        }
