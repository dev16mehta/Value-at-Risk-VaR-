# var_project/data_reader.py
import numpy as np
import yfinance as yf
import pandas as pd
from typing import List

class PriceHistory:
    """
    Fetches historical price data for multiple assets and calculates returns.
    """
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = self._fetch_prices()
        self.returns = self._calculate_returns()

    def _fetch_prices(self) -> pd.DataFrame:
        print(f"Fetching prices for {', '.join(self.tickers)}...")
        
        # yfinance allows downloading multiple tickers at once
        data = yf.download(
            self.tickers, 
            start=self.start_date, 
            end=self.end_date, 
            auto_adjust=True, 
            progress=False
        )
        
        # Extract just the closing prices
        # If multiple tickers, data['Close'] is a DataFrame with columns=tickers
        # If single ticker, it might be a Series, so we ensure it's a DataFrame
        prices = data['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=self.tickers[0])
            
        # Reorder columns to match self.tickers order to be safe
        prices = prices[self.tickers]
        return prices.dropna()

    def _calculate_returns(self) -> pd.DataFrame:
        """Calculates percentage returns."""
        # pct_change() is a pandas method that handles the math for us
        return self.prices.pct_change().dropna()

    def get_returns_matrix(self) -> np.ndarray:
        """Returns a NumPy matrix of returns (Rows=Dates, Cols=Assets)"""
        return self.returns.to_numpy()
        
    def get_returns_df(self) -> pd.DataFrame:
        return self.returns