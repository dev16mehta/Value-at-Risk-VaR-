# var_project/main.py
"""
Command-line interface for running VaR analysis and backtesting.
Demonstrates the Delta-Normal approximation for portfolios with options.
"""
from .data_reader import PriceHistory
from .var_methods import HistoricalVaR, ParametricVaR
from .monte_carlo import MonteCarloVaR
from .portfolio import Portfolio
from .volatility import EWMACovariance, GarchCovariance
from .backtester import Backtester
from .plotting import VarPlotter
from .instruments import Stock, CallOption


def main():
    # Configuration
    start_date = "2015-01-01"
    end_date = "2019-10-01"
    confidence_level = 0.95
    backtest_window = 250
    mc_simulations = 5000

    # Sample portfolio: stocks and options on split-adjusted tickers
    portfolio_instruments = [
        Stock(ticker="AAPL", shares=10000),
        Stock(ticker="MSFT", shares=3500),
        # GOOG ATM call: strike ~$60 post-split, 500 contracts
        CallOption(ticker="GOOG", strikes=60, expiry_years=0.5, contracts=500),
        # AMZN OTM call: strike $95 post-split, 300 contracts
        CallOption(ticker="AMZN", strikes=95, expiry_years=0.5, contracts=300)
    ]

    portfolio = Portfolio(portfolio_instruments)

    print("=" * 80)
    print(f"VALUATION DATE:   {end_date}")
    print(f"ASSETS INVOLVED:  {', '.join(portfolio.tickers)}")
    print("=" * 80)

    try:
        price_history = PriceHistory(portfolio.tickers, start_date, end_date)

        # Use latest prices to calculate option deltas and portfolio value
        current_prices = price_history.prices.iloc[-1].to_dict()

        print("\n--- Calibrating Option Sensitivities (Delta) ---")
        portfolio.rebalance(current_prices)

        print(f"Total Portfolio Value: ${portfolio.total_value:,.2f}")
        print("Delta-Adjusted Weights:")
        for ticker, weight in zip(portfolio.tickers, portfolio.weights):
            print(f"  {ticker:<5}: {weight:.2%}")

        # Setup volatility estimators and VaR engines
        # VaR engines use delta-adjusted weights (Delta-Normal approximation)
        ewma_est = EWMACovariance(lambda_param=0.94)
        garch_est = GarchCovariance()

        hist_var_method = HistoricalVaR(price_history)
        param_var_ewma = ParametricVaR(price_history, estimator=ewma_est)
        param_var_garch = ParametricVaR(price_history, estimator=garch_est)
        mc_var_ewma = MonteCarloVaR(price_history, estimator=ewma_est, simulations=mc_simulations)

        print("\n--- Current 1-Day VaR Estimates (Delta-Normal) ---")
        print(f"{'Historical Simulation:':<25} ${hist_var_method.calculate_var(portfolio, confidence_level):,.2f}")
        print(f"{'Parametric (EWMA):':<25} ${param_var_ewma.calculate_var(portfolio, confidence_level):,.2f}")
        print(f"{'Parametric (GARCH):':<25} ${param_var_garch.calculate_var(portfolio, confidence_level):,.2f}")
        print(f"{'Monte Carlo (EWMA):':<25} ${mc_var_ewma.calculate_var(portfolio, confidence_level):,.2f}")

        # Backtesting (assumes static delta - risk profile remains constant)
        print(f"\n{'='*30} BACKTESTING RESULTS {'='*30}")
        print("Note: This backtest assumes 'Static Delta' (Risk profile remains constant).")

        backtester = Backtester(price_history, portfolio)

        methods_to_test = {
            "Historical": hist_var_method,
            "Parametric (EWMA)": param_var_ewma,
            "Parametric (GARCH)": param_var_garch,
            "Monte Carlo (EWMA)": mc_var_ewma
        }

        header = f"{'Method':<20} | {'Breaches':<8} | {'Rate':<7} | {'Expected':<8} | {'P-Value':<7} | {'Result'}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for name, method in methods_to_test.items():
            res = backtester.run_backtest(method, confidence_level, window_size=backtest_window)

            print(f"{name:<20} | "
                  f"{res['breaches']:<8} | "
                  f"{res['breach_rate']:.1%}   | "
                  f"{1-confidence_level:.1%}     | "
                  f"{res['p_value']:.4f}  | "
                  f"{res['decision']}")

            print(f"Generating plot for {name}...")
            VarPlotter.plot_backtest(res, name, confidence_level)

        print("-" * len(header))

    except Exception as e:
        print(f"\n[ERROR] An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
