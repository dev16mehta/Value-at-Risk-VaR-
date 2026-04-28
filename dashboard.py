# var_project/dashboard.py
"""
Value at Risk (VaR) Dashboard
A user-friendly interface for portfolio risk analysis.
"""

import customtkinter as ctk
from tkinter import messagebox
import threading

from .data_reader import PriceHistory
from .portfolio import Portfolio
from .instruments import Stock, CallOption
from .volatility import SimpleCovariance, EWMACovariance, GarchCovariance
from .var_methods import HistoricalVaR, ParametricVaR
from .monte_carlo import MonteCarloVaR
from .backtester import Backtester
from .plotting import VarPlotter

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")

COLORS = {
    "bg_main": "#f8fafc",
    "bg_card": "#ffffff",
    "bg_sidebar": "#f1f5f9",
    "bg_input": "#ffffff",
    "accent_primary": "#10b981",      # Green
    "accent_secondary": "#0ea5e9",    # Light blue
    "accent_red": "#ef4444",
    "accent_yellow": "#f59e0b",
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "text_muted": "#94a3b8",
    "border": "#e2e8f0",
    "border_focus": "#10b981",
    "shadow": "#cbd5e1",
}


class ModernEntry(ctk.CTkFrame):
    """Custom entry with label"""
    def __init__(self, master, label, default="", width=200):
        super().__init__(master, fg_color="transparent")

        self.label = ctk.CTkLabel(self, text=label, font=ctk.CTkFont(size=12, weight="bold"),
                                   text_color=COLORS["text_secondary"])
        self.label.pack(anchor="w", pady=(0, 4))

        self.entry = ctk.CTkEntry(self, width=width, height=38,
                                   fg_color=COLORS["bg_input"],
                                   border_color=COLORS["border"],
                                   text_color=COLORS["text_primary"],
                                   font=ctk.CTkFont(size=13))
        self.entry.pack(fill="x")
        self.entry.insert(0, default)

    def get(self):
        return self.entry.get()

    def set(self, value):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)


class VaRResultCard(ctk.CTkFrame):
    """Card displaying VaR result for a single method"""
    def __init__(self, master, method_name, var_value, decision, breach_rate, p_value, on_view_plot):
        super().__init__(master, fg_color=COLORS["bg_card"], corner_radius=12,
                        border_width=1, border_color=COLORS["border"])

        if "ACCEPT" in decision:
            status_color = COLORS["accent_primary"]
            status_text = "PASS"
        else:
            status_color = COLORS["accent_red"]
            status_text = "FAIL"

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=16)

        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(top_row, text=method_name, font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        status_badge = ctk.CTkLabel(top_row, text=f"  {status_text}  ",
                                     font=ctk.CTkFont(size=11, weight="bold"),
                                     fg_color=status_color,
                                     corner_radius=4, text_color="white")
        status_badge.pack(side="right")

        ctk.CTkLabel(content, text=f"${var_value:,.0f}",
                     font=ctk.CTkFont(size=32, weight="bold"),
                     text_color=COLORS["accent_secondary"]).pack(anchor="w")

        ctk.CTkLabel(content, text="1-Day Value at Risk",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 12))

        stats_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_sidebar"], corner_radius=8)
        stats_frame.pack(fill="x", pady=(0, 8))

        stats_inner = ctk.CTkFrame(stats_frame, fg_color="transparent")
        stats_inner.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(stats_inner, text="Backtest Breach Rate",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left")
        ctk.CTkLabel(stats_inner, text=f"{breach_rate:.1%}",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="right")

        pval_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_sidebar"], corner_radius=8)
        pval_frame.pack(fill="x", pady=(0, 12))

        pval_inner = ctk.CTkFrame(pval_frame, fg_color="transparent")
        pval_inner.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(pval_inner, text="P-Value",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left")
        ctk.CTkLabel(pval_inner, text=f"{p_value:.4f}",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="right")

        ctk.CTkButton(content, text="View Backtest Chart", height=36,
                      fg_color=COLORS["accent_secondary"],
                      hover_color="#0284c7",
                      font=ctk.CTkFont(size=12, weight="bold"),
                      command=lambda: on_view_plot(method_name)).pack(fill="x")


class AssetTableRow(ctk.CTkFrame):
    """Asset row in the portfolio table"""
    def __init__(self, master, parent_app, index, asset_type="Stock", ticker="", qty="",
                 strike="", expiry=""):
        super().__init__(master, fg_color=COLORS["bg_card"], corner_radius=8,
                        border_width=1, border_color=COLORS["border"], height=56)
        self.parent_app = parent_app
        self.index = index
        self.pack_propagate(False)

        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.grid_columnconfigure(5, weight=0)

        self.type_var = ctk.StringVar(value=asset_type)
        self.combo_type = ctk.CTkComboBox(
            self, values=["Stock", "Call Option"],
            variable=self.type_var, width=120, height=36,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            button_color=COLORS["accent_primary"], button_hover_color="#059669",
            dropdown_fg_color=COLORS["bg_card"],
            text_color=COLORS["text_primary"],
            command=self.toggle_fields
        )
        self.combo_type.grid(row=0, column=0, padx=(12, 6), pady=10, sticky="w")

        self.entry_ticker = ctk.CTkEntry(
            self, placeholder_text="AAPL", width=100, height=36,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            text_color=COLORS["text_primary"]
        )
        self.entry_ticker.grid(row=0, column=1, padx=6, pady=10)
        if ticker:
            self.entry_ticker.insert(0, ticker)

        placeholder = "Value ($)" if asset_type == "Stock" else "Contracts"
        self.entry_qty = ctk.CTkEntry(
            self, placeholder_text=placeholder, width=130, height=36,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            text_color=COLORS["text_primary"]
        )
        self.entry_qty.grid(row=0, column=2, padx=6, pady=10)
        if qty:
            self.entry_qty.insert(0, str(qty))

        self.entry_strike = ctk.CTkEntry(
            self, placeholder_text="Strike $", width=90, height=36,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            text_color=COLORS["text_primary"]
        )
        if strike:
            self.entry_strike.insert(0, str(strike))

        self.entry_expiry = ctk.CTkEntry(
            self, placeholder_text="Years", width=80, height=36,
            fg_color=COLORS["bg_input"], border_color=COLORS["border"],
            text_color=COLORS["text_primary"]
        )
        if expiry:
            self.entry_expiry.insert(0, str(expiry))

        self.btn_del = ctk.CTkButton(
            self, text="Remove", width=70, height=32,
            fg_color="transparent", hover_color=COLORS["bg_sidebar"],
            text_color=COLORS["accent_red"], border_width=1,
            border_color=COLORS["accent_red"],
            font=ctk.CTkFont(size=11), command=self.delete_row
        )
        self.btn_del.grid(row=0, column=5, padx=(6, 12), pady=10)
        self.toggle_fields(asset_type)

    def toggle_fields(self, choice):
        if "Option" in choice:
            self.entry_strike.grid(row=0, column=3, padx=6, pady=10)
            self.entry_expiry.grid(row=0, column=4, padx=6, pady=10)
            self.entry_qty.configure(placeholder_text="Contracts")
            if not self.entry_strike.get():
                self.entry_strike.insert(0, "60")
            if not self.entry_expiry.get():
                self.entry_expiry.insert(0, "0.5")
            self.entry_qty.delete(0, "end")
            self.entry_qty.insert(0, "500")
        else:
            self.entry_strike.grid_forget()
            self.entry_expiry.grid_forget()
            self.entry_qty.configure(placeholder_text="Value ($)")
            self.entry_qty.delete(0, "end")
            self.entry_qty.insert(0, "500000")

    def delete_row(self):
        self.parent_app.remove_asset_row(self)

    def get_raw_data(self):
        """Get raw data from the row for processing"""
        ticker = self.entry_ticker.get().upper().strip()
        try:
            value = float(self.entry_qty.get())
        except ValueError:
            value = 0.0

        if not ticker:
            return None

        asset_type = self.type_var.get()
        if asset_type == "Stock":
            return {"type": "stock", "ticker": ticker, "value": value}
        else:
            try:
                strike = float(self.entry_strike.get())
                expiry = float(self.entry_expiry.get())
                return {"type": "option", "ticker": ticker, "contracts": value,
                        "strike": strike, "expiry": expiry}
            except ValueError:
                return None


class VaRDashboard(ctk.CTk):
    """Value at Risk Dashboard with Tab System"""

    def __init__(self):
        super().__init__()

        self.title("Value at Risk (VaR)")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.configure(fg_color=COLORS["bg_main"])

        self.asset_rows = []
        self.backtest_results = {}
        self.result_cards = []

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._create_sidebar()
        self._create_main_content()
        self._add_default_assets()

    def _create_sidebar(self):
        """Create the configuration sidebar"""
        self.sidebar = ctk.CTkFrame(self, width=300, fg_color=COLORS["bg_sidebar"], corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=24, pady=(28, 8))

        ctk.CTkLabel(title_frame, text="Value at Risk",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="Portfolio Risk Analysis Tool",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_muted"]).pack(anchor="w", pady=(2, 0))

        self.btn_run = ctk.CTkButton(
            self.sidebar, text="Run Analysis", height=48,
            fg_color=COLORS["accent_primary"], hover_color="#059669",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.start_analysis_thread
        )
        self.btn_run.pack(fill="x", padx=24, pady=(20, 16))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=24, pady=8)

        ctk.CTkLabel(self.sidebar, text="Settings",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=24, pady=(16, 12))

        date_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        date_frame.pack(fill="x", padx=24, pady=(0, 8))

        self.entry_start = ModernEntry(date_frame, "Start Date", "2010-12-12")
        self.entry_start.pack(fill="x", pady=(0, 8))

        self.entry_end = ModernEntry(date_frame, "End Date", "2019-10-31")
        self.entry_end.pack(fill="x")

        ctk.CTkFrame(self.sidebar, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=24, pady=16)

        ctk.CTkLabel(self.sidebar, text="VaR Methods",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=24, pady=(0, 12))

        methods_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        methods_frame.pack(fill="x", padx=24)

        self.method_vars = {}
        methods = [
            ("Historical", "historical"),
            ("Parametric (EWMA)", "ewma"),
            ("Parametric (GARCH)", "garch"),
            ("Monte Carlo", "mc")
        ]

        for label, key in methods:
            var = ctk.BooleanVar(value=True)
            self.method_vars[key] = var
            cb = ctk.CTkCheckBox(methods_frame, text=label, variable=var,
                                  font=ctk.CTkFont(size=13),
                                  fg_color=COLORS["accent_primary"],
                                  hover_color="#059669",
                                  border_color=COLORS["border"],
                                  text_color=COLORS["text_primary"])
            cb.pack(anchor="w", pady=5)

        ctk.CTkFrame(self.sidebar, fg_color="transparent").pack(fill="both", expand=True)

        self.progress_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.progress_frame.pack(fill="x", padx=24, pady=(0, 24))

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="",
                                            font=ctk.CTkFont(size=12),
                                            text_color=COLORS["text_secondary"])
        self.progress_label.pack(anchor="w")

        self.progress = ctk.CTkProgressBar(self.progress_frame, mode="indeterminate",
                                            progress_color=COLORS["accent_primary"])
        self.progress.pack(fill="x", pady=(6, 0))
        self.progress.pack_forget()

    def _create_main_content(self):
        """Create the main content area with tabs"""
        self.main_area = ctk.CTkFrame(self, fg_color=COLORS["bg_main"], corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self.main_area, fg_color=COLORS["bg_main"],
                                       segmented_button_fg_color=COLORS["bg_sidebar"],
                                       segmented_button_selected_color=COLORS["accent_primary"],
                                       segmented_button_selected_hover_color="#059669",
                                       segmented_button_unselected_color=COLORS["bg_sidebar"],
                                       segmented_button_unselected_hover_color=COLORS["border"],
                                       text_color=COLORS["text_primary"],
                                       text_color_disabled=COLORS["text_muted"])
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)

        self.tab_portfolio = self.tabview.add("Portfolio")
        self.tab_results = self.tabview.add("Results")

        self.tab_portfolio.grid_columnconfigure(0, weight=1)
        self.tab_portfolio.grid_rowconfigure(1, weight=1)

        self.tab_results.grid_columnconfigure(0, weight=1)
        self.tab_results.grid_rowconfigure(1, weight=1)

        self._create_portfolio_tab()
        self._create_results_tab()

    def _create_portfolio_tab(self):
        """Create the portfolio composition tab"""
        header = ctk.CTkFrame(self.tab_portfolio, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(header, text="Portfolio Composition",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        self.btn_add = ctk.CTkButton(
            header, text="+ Add Asset", width=120, height=38,
            fg_color=COLORS["accent_primary"], hover_color="#059669",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.add_asset_row
        )
        self.btn_add.pack(side="right")

        portfolio_card = ctk.CTkFrame(self.tab_portfolio, fg_color=COLORS["bg_card"],
                                       corner_radius=12, border_width=1,
                                       border_color=COLORS["border"])
        portfolio_card.grid(row=1, column=0, sticky="nsew")
        portfolio_card.grid_columnconfigure(0, weight=1)
        portfolio_card.grid_rowconfigure(1, weight=1)

        headers_frame = ctk.CTkFrame(portfolio_card, fg_color=COLORS["bg_sidebar"],
                                      corner_radius=0, height=44)
        headers_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 0))
        headers_frame.pack_propagate(False)
        headers_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        headers_frame.grid_columnconfigure(5, weight=0)

        ctk.CTkLabel(headers_frame, text="Type", width=120,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["text_secondary"]).grid(row=0, column=0, padx=(24, 6), pady=12, sticky="w")
        ctk.CTkLabel(headers_frame, text="Ticker", width=100,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["text_secondary"]).grid(row=0, column=1, padx=6, pady=12, sticky="w")
        ctk.CTkLabel(headers_frame, text="Value / Qty", width=130,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["text_secondary"]).grid(row=0, column=2, padx=6, pady=12, sticky="w")
        ctk.CTkLabel(headers_frame, text="Strike", width=90,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["text_secondary"]).grid(row=0, column=3, padx=6, pady=12, sticky="w")
        ctk.CTkLabel(headers_frame, text="Expiry", width=80,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["text_secondary"]).grid(row=0, column=4, padx=6, pady=12, sticky="w")
        ctk.CTkLabel(headers_frame, text="", width=70,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=COLORS["text_secondary"]).grid(row=0, column=5, padx=(6, 24), pady=12, sticky="w")

        self.asset_scroll = ctk.CTkScrollableFrame(
            portfolio_card, fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent_primary"]
        )
        self.asset_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        info_frame = ctk.CTkFrame(portfolio_card, fg_color="transparent")
        info_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))

        ctk.CTkLabel(info_frame,
                     text="Enter dollar values for stocks and contract counts for options. Strike prices and expiry (in years) are required for options.",
                     font=ctk.CTkFont(size=11),
                     text_color=COLORS["text_muted"]).pack(anchor="w")

    def _create_results_tab(self):
        """Create the results tab"""
        header = ctk.CTkFrame(self.tab_results, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(header, text="Analysis Results",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        self.portfolio_value_label = ctk.CTkLabel(
            header, text="", font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["accent_secondary"]
        )
        self.portfolio_value_label.pack(side="right")

        self.results_scroll = ctk.CTkScrollableFrame(
            self.tab_results, fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent_primary"]
        )
        self.results_scroll.grid(row=1, column=0, sticky="nsew")
        self._show_initial_message()

    def _show_initial_message(self):
        """Show initial message in results tab"""
        self.initial_message = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["bg_card"],
                                             corner_radius=12, border_width=1,
                                             border_color=COLORS["border"])
        self.initial_message.pack(fill="x", pady=20, padx=4)

        msg_content = ctk.CTkFrame(self.initial_message, fg_color="transparent")
        msg_content.pack(pady=80)

        ctk.CTkLabel(msg_content, text="No Results Yet",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(msg_content,
                     text="Configure your portfolio in the Portfolio tab and click 'Run Analysis'",
                     font=ctk.CTkFont(size=14),
                     text_color=COLORS["text_muted"]).pack(pady=(12, 0))

    def add_asset_row(self, asset_type="Stock", ticker="", qty="", strike="", expiry=""):
        """Add a new asset row to the portfolio"""
        row = AssetTableRow(self.asset_scroll, self, len(self.asset_rows),
                           asset_type, ticker, qty, strike, expiry)
        row.pack(fill="x", pady=4)
        self.asset_rows.append(row)

    def remove_asset_row(self, row_obj):
        """Remove an asset row"""
        row_obj.pack_forget()
        row_obj.destroy()
        if row_obj in self.asset_rows:
            self.asset_rows.remove(row_obj)

    def _add_default_assets(self):
        """Add default portfolio assets - stocks only"""
        defaults = [
            ("Stock", "AAPL", "500000", "", ""),
            ("Stock", "MSFT", "500000", "", ""),
            ("Stock", "GOOG", "500000", "", ""),
            ("Stock", "AMZN", "500000", "", ""),
        ]
        for d in defaults:
            self.add_asset_row(*d)

    def start_analysis_thread(self):
        """Start analysis in a background thread"""
        self.btn_run.configure(state="disabled", text="Running...")
        self.progress_label.configure(text="Initializing...")
        self.progress.pack(fill="x", pady=(6, 0))
        self.progress.start()

        if hasattr(self, 'initial_message') and self.initial_message:
            self.initial_message.pack_forget()
            self.initial_message.destroy()
            self.initial_message = None

        for card in self.result_cards:
            card.pack_forget()
            card.destroy()
        self.result_cards = []
        self.backtest_results = {}

        threading.Thread(target=self.run_analysis, daemon=True).start()

    def update_progress(self, message):
        """Update progress label from main thread"""
        self.progress_label.configure(text=message)

    def run_analysis(self):
        """Run the VaR analysis"""
        try:
            start_date = self.entry_start.get()
            end_date = self.entry_end.get()
            conf_level = 0.95
            sims = 2000

            self.after(0, self.update_progress, "Building portfolio...")
            raw_assets = []
            for row in self.asset_rows:
                data = row.get_raw_data()
                if data:
                    raw_assets.append(data)

            if not raw_assets:
                self.after(0, lambda: messagebox.showerror("Error", "Portfolio is empty. Please add assets."))
                self.finish_analysis()
                return

            tickers = list(set(asset["ticker"] for asset in raw_assets))

            self.after(0, self.update_progress, "Fetching market data...")
            price_history = PriceHistory(tickers, start_date, end_date)
            current_prices = price_history.prices.iloc[-1].to_dict()

            instruments = []
            for asset in raw_assets:
                ticker = asset["ticker"]
                if asset["type"] == "stock":
                    price = current_prices.get(ticker, 1)
                    shares = asset["value"] / price if price > 0 else 0
                    instruments.append(Stock(ticker, shares))
                else:
                    instruments.append(CallOption(
                        ticker, asset["strike"], asset["expiry"],
                        contracts=asset["contracts"]
                    ))

            portfolio = Portfolio(instruments)
            portfolio.rebalance(current_prices)

            total_value = portfolio.total_value
            self.after(0, lambda v=total_value: self.portfolio_value_label.configure(
                text=f"Portfolio Value: ${v:,.2f}"
            ))

            ewma_est = EWMACovariance(0.94)
            garch_est = GarchCovariance()

            methods = {}
            if self.method_vars["historical"].get():
                methods["Historical"] = HistoricalVaR(price_history)
            if self.method_vars["ewma"].get():
                methods["Parametric (EWMA)"] = ParametricVaR(price_history, estimator=ewma_est)
            if self.method_vars["garch"].get():
                methods["Parametric (GARCH)"] = ParametricVaR(price_history, estimator=garch_est)
            if self.method_vars["mc"].get():
                methods["Monte Carlo (EWMA)"] = MonteCarloVaR(price_history, estimator=ewma_est, simulations=sims)

            if not methods:
                self.after(0, lambda: messagebox.showerror("Error", "Please select at least one VaR method."))
                self.finish_analysis()
                return

            backtester = Backtester(price_history, portfolio)

            for name, method in methods.items():
                self.after(0, self.update_progress, f"Analyzing {name}...")
                var_val = method.calculate_var(portfolio, conf_level)
                res = backtester.run_backtest(method, conf_level, window_size=250)
                self.backtest_results[name] = res
                self.after(0, self.create_result_card, name, var_val,
                          res['decision'], res['breach_rate'], res['p_value'])

            self.after(0, self.update_progress, "Analysis complete!")
            self.after(100, lambda: self.tabview.set("Results"))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            import traceback
            traceback.print_exc()
        finally:
            self.after(500, self.finish_analysis)

    def create_result_card(self, method_name, var_value, decision, breach_rate, p_value):
        """Create a result card for a VaR method"""
        card = VaRResultCard(self.results_scroll, method_name, var_value,
                            decision, breach_rate, p_value, self.show_plot)
        card.pack(fill="x", pady=6, padx=4)
        self.result_cards.append(card)

    def finish_analysis(self):
        """Clean up after analysis"""
        self.progress.stop()
        self.progress.pack_forget()
        self.progress_label.configure(text="")
        self.btn_run.configure(state="normal", text="Run Analysis")

    def show_plot(self, method_name):
        """Show backtest plot for a method"""
        if method_name in self.backtest_results:
            VarPlotter.plot_backtest(self.backtest_results[method_name], method_name, 0.95)


def main():
    """Entry point for the dashboard"""
    app = VaRDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
