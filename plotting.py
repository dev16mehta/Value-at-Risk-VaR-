# var_project/plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

class VarPlotter:
    """
    Handles visualisation of VaR backtesting results.
    """
    
    @staticmethod
    def plot_backtest(results: dict, method_name: str, confidence_level: float):
        """
        Generates a PnL vs VaR chart with breaches highlighted.
        """
        dates = results['dates']
        actual_returns = results['actual_returns']
        var_estimates = results['var_estimates']
        
        # Convert to numpy arrays for easier masking
        dates = np.array(dates)
        actual = np.array(actual_returns)
        var = np.array(var_estimates)
        neg_var = -var  # The VaR limit line (negative)

        # Identify breaches (Loss < -VaR)
        breach_indices = np.where(actual < neg_var)[0]
        breach_dates = dates[breach_indices]
        breach_values = actual[breach_indices]

        # Setup Plot
        plt.figure(figsize=(12, 6))
        plt.title(f"VaR Backtest: {method_name} ({confidence_level:.0%})")
        
        # 1. Plot Actual PnL
        plt.plot(dates, actual, label="Actual PnL", color='grey', alpha=0.5, linewidth=1)
        
        # 2. Plot VaR Limit
        plt.plot(dates, neg_var, label="VaR Limit", color='blue', linewidth=1.5)
        
        # 3. Highlight Breaches
        plt.scatter(breach_dates, breach_values, color='red', label='Breach', zorder=5, s=30)

        # Formatting
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.ylabel("Portfolio Value Change ($)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format Date Axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gcf().autofmt_xdate()

        # Show Plot
        plt.tight_layout()
        plt.show()