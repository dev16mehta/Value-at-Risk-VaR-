# Value at Risk (VaR) Engine

## Overview
This project is a professional-grade, Object-Oriented Python framework designed to calculate, simulate, and backtest **Value at Risk (VaR)** and **Expected Shortfall (CVaR)** for multi-asset portfolios. 

Built as a Final Year Computer Science project, the system bridges advanced financial mathematics with robust software engineering principles. It supports complex financial instruments (including non-linear derivatives) and provides a modern Graphical User Interface (GUI) for interactive risk analysis.

## Core Features
* **Modern GUI Dashboard:** A responsive, multithreaded interface built with `customtkinter`, allowing users to construct portfolios, set parameters, and visualize backtest charts seamlessly.
* **Advanced VaR Methodologies:**
    * **Historical Simulation:** Highly optimized vectorised implementation.
    * **Parametric VaR:** Variance-Covariance approach supporting robust distributions (Student's t-distribution for fat tails).
    * **Monte Carlo Simulation:** Utilises Cholesky Decomposition to generate correlated stochastic price paths.
* **Dynamic Volatility Models:**
    * **Simple Covariance:** Standard historical sample covariance.
    * **EWMA:** Exponentially Weighted Moving Average (with empirically optimized $\lambda$).
    * **CCC-GARCH(1,1):** Constant Conditional Correlation GARCH, utilizing Maximum Likelihood Estimation (MLE) via `scipy.optimize`.
* **Derivatives Support:** Handles non-linear assets (European Call Options) using Black-Scholes pricing and the Delta-Normal approximation for risk weighting.
* **Rigorous Backtesting & Stress Testing:** * A rolling-window backtesting framework validated by the **Kupiec Proportion of Failures (POF)** statistical test.
    * Deterministic Crash Metrics simulating extreme historical shocks (e.g., 1987 Black Monday).

## Installation

### Prerequisites
* Python 3.8 or higher
* Internet connection (for automated data ingestion via Yahoo Finance)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <project-folder>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the application in two ways: using the modern GUI or via the Command Line Interface (CLI).

### 1. Launch the GUI Dashboard (Recommended)
This launches the interactive application where you can dynamically add stocks and options, set dates, and visualize backtest charts.
```bash
python -m var_project.dashboard