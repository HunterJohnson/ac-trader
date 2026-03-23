"""
Vectorised backtesting engine with transaction costs and slippage.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from risk import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """
    Event-driven daily backtester.

    Workflow:
        1. On each day receive signals from the strategy.
        2. RiskManager sizes positions.
        3. Apply commission + slippage on entries/exits.
        4. Track P&L, drawdown, trade log.
    """

    def __init__(self, config: dict, risk_config: dict):
        self.initial_capital: float = config["initial_capital"]
        self.commission: float = config["commission_pct"]
        self.slippage: float = config["slippage_pct"]
        self.risk_mgr = RiskManager(risk_config)

    def run(
        self,
        signals: pd.DataFrame,      # (Date × Ticker) binary signals
        price_data: pd.DataFrame,   # MultiIndex (Date, Ticker) OHLCV
    ) -> BacktestResult:
        close = price_data["Close"].unstack("Ticker").reindex(columns=signals.columns)
        returns_df = close.pct_change()

        cash = self.initial_capital
        positions: Dict[str, float] = {}   # ticker → shares held
        equity_log = []
        trade_log = []

        dates = signals.index.intersection(close.index)

        for date in dates:
            prices_today = close.loc[date]
            sig_today = signals.loc[date]

            # ── Mark-to-market ──
            port_val = cash + sum(
                positions.get(t, 0) * prices_today.get(t, 0) for t in positions
            )
            self.risk_mgr.check_circuit_breaker(port_val)
            equity_log.append({"date": date, "equity": port_val})

            # ── Compute rolling volatility for sizing ──
            hist = returns_df.loc[:date].tail(21)
            vols = hist.std() * np.sqrt(252)

            # ── Check stops on existing positions ──
            tickers_to_exit = []
            for ticker, shares in list(positions.items()):
                price = prices_today.get(ticker, np.nan)
                if pd.isna(price) or shares == 0:
                    continue
                if self.risk_mgr.should_exit(ticker, price):
                    tickers_to_exit.append(ticker)

            for ticker in tickers_to_exit:
                price = prices_today[ticker]
                exit_price = price * (1 - self.slippage)
                shares = positions.pop(ticker, 0)
                proceeds = shares * exit_price * (1 - self.commission)
                cash += proceeds
                self.risk_mgr.clear_entry(ticker)
                trade_log.append(
                    {"date": date, "ticker": ticker, "action": "SELL",
                     "shares": shares, "price": exit_price}
                )

            # ── Exit positions no longer in signal ──
            desired = set(sig_today[sig_today == 1].index)
            for ticker in list(positions.keys()):
                if ticker not in desired:
                    price = prices_today.get(ticker, np.nan)
                    if pd.isna(price):
                        continue
                    exit_price = price * (1 - self.slippage)
                    shares = positions.pop(ticker)
                    proceeds = shares * exit_price * (1 - self.commission)
                    cash += proceeds
                    self.risk_mgr.clear_entry(ticker)
                    trade_log.append(
                        {"date": date, "ticker": ticker, "action": "SELL",
                         "shares": shares, "price": exit_price}
                    )

            # ── Enter new positions ──
            dollar_allocs = self.risk_mgr.size_positions(
                sig_today, prices_today, port_val, vols
            )
            for ticker, alloc in dollar_allocs.items():
                if alloc <= 0 or ticker in positions:
                    continue
                price = prices_today.get(ticker, np.nan)
                if pd.isna(price) or price <= 0:
                    continue
                entry_price = price * (1 + self.slippage)
                cost_per_share = entry_price * (1 + self.commission)
                shares = alloc / cost_per_share
                total_cost = shares * cost_per_share
                if total_cost > cash:
                    shares = cash / cost_per_share
                    total_cost = cash
                if shares <= 0:
                    continue
                cash -= total_cost
                positions[ticker] = shares
                self.risk_mgr.record_entry(ticker, entry_price)
                trade_log.append(
                    {"date": date, "ticker": ticker, "action": "BUY",
                     "shares": shares, "price": entry_price}
                )

        equity = pd.DataFrame(equity_log).set_index("date")["equity"]
        trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
            columns=["date", "ticker", "action", "shares", "price"]
        )
        daily_returns = equity.pct_change().fillna(0)
        metrics = self._compute_metrics(equity, daily_returns)

        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            daily_returns=daily_returns,
            metrics=metrics,
        )

    # ── Performance metrics ──────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-9)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / (ann_vol + 1e-9)

        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = drawdown.min()

        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = (returns.mean() * 252) / (downside + 1e-9)

        win_rate = (returns > 0).mean()

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "ann_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3) if not np.isnan(calmar) else None,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
        }
