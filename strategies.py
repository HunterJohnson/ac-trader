"""
Three concrete trading strategies for the S&P 500 universe.

Each strategy exposes:
    generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame
        data   : MultiIndex (Date, Ticker) OHLCV DataFrame
        returns: DataFrame with columns [ticker] and rows [date],
                 values in {-1, 0, 1}  (short / flat / long)
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

from indicators import (
    ema, rsi, bollinger_bands, atr, momentum_score, z_score
)

logger = logging.getLogger(__name__)


# ── 1. Cross-Sectional Momentum ──────────────────────────────────────────────

class MomentumStrategy:
    """
    Each month rank all tickers by risk-adjusted momentum over the past
    ~6 months (skipping the most-recent month to avoid short-term reversal).
    Go long the top-N names with equal weight; hold until next rebalance.
    """

    def generate_signals(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        lookback = params["lookback_days"]
        skip = params["skip_days"]
        top_n = params["top_n"]

        close = data["Close"].unstack("Ticker")
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)

        # Resample to month-end rebalance dates
        rebal_dates = close.resample("ME").last().index

        prev_signal = pd.Series(0, index=close.columns)
        for date in close.index:
            if date in rebal_dates:
                if close.loc[:date].shape[0] >= lookback:
                    scores = close.loc[:date].apply(
                        lambda s: momentum_score(s, lookback, skip)
                    )
                    top = scores.nlargest(top_n).index
                    prev_signal = pd.Series(0, index=close.columns)
                    prev_signal[top] = 1
            signals.loc[date] = prev_signal

        return signals


# ── 2. Mean-Reversion (RSI + Bollinger Bands) ────────────────────────────────

class MeanReversionStrategy:
    """
    Buy when RSI < oversold threshold AND price touches/breaks the lower
    Bollinger Band.  Exit after `holding_days` or when RSI crosses back above 50.
    """

    def generate_signals(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        rsi_period = params["rsi_period"]
        rsi_os = params["rsi_oversold"]
        rsi_ob = params["rsi_overbought"]
        bb_period = params["bb_period"]
        bb_std = params["bb_std"]
        hold = params["holding_days"]

        close = data["Close"].unstack("Ticker")
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)

        for ticker in close.columns:
            price = close[ticker].dropna()
            r = rsi(price, rsi_period)
            upper, mid, lower = bollinger_bands(price, bb_period, bb_std)

            entry = (r < rsi_os) & (price <= lower)
            exit_ = (r > rsi_ob) | (price >= mid)

            sig = pd.Series(0, index=price.index)
            in_trade = False
            days_held = 0
            for i, idx in enumerate(price.index):
                if not in_trade and entry.get(idx, False):
                    in_trade = True
                    days_held = 0
                if in_trade:
                    sig[idx] = 1
                    days_held += 1
                    if exit_.get(idx, False) or days_held >= hold:
                        in_trade = False

            signals[ticker] = sig.reindex(signals.index, fill_value=0)

        return signals


# ── 3. Trend Following (Dual EMA + ATR trailing stop) ────────────────────────

class TrendFollowingStrategy:
    """
    Go long when the 50-day EMA crosses above the 200-day EMA (golden cross).
    Trail a stop at close − ATR_multiplier × ATR(14).
    Exit on death cross or stop hit.
    """

    def generate_signals(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        fast = params["fast_ema"]
        slow = params["slow_ema"]
        atr_period = params["atr_period"]
        atr_mult = params["atr_multiplier"]

        close = data["Close"].unstack("Ticker")
        high = data["High"].unstack("Ticker")
        low = data["Low"].unstack("Ticker")
        signals = pd.DataFrame(0, index=close.index, columns=close.columns)

        for ticker in close.columns:
            c = close[ticker].dropna()
            h = high[ticker].reindex(c.index)
            l = low[ticker].reindex(c.index)

            fast_ema = ema(c, fast)
            slow_ema = ema(c, slow)
            atr_val = atr(h, l, c, atr_period)

            golden = fast_ema > slow_ema
            death = fast_ema <= slow_ema

            sig = pd.Series(0, index=c.index)
            in_trade = False
            trail_stop = 0.0

            for idx in c.index:
                price = c[idx]
                if not in_trade and golden.get(idx, False):
                    in_trade = True
                    trail_stop = price - atr_mult * atr_val.get(idx, 0)
                elif in_trade:
                    # Update trailing stop upward only
                    new_stop = price - atr_mult * atr_val.get(idx, 0)
                    trail_stop = max(trail_stop, new_stop)
                    if death.get(idx, False) or price <= trail_stop:
                        in_trade = False
                        trail_stop = 0.0
                    else:
                        sig[idx] = 1

            signals[ticker] = sig.reindex(signals.index, fill_value=0)

        return signals


# ── Ensemble: weighted combination of all three ──────────────────────────────

class EnsembleStrategy:
    """
    Combine momentum, mean-reversion, and trend-following signals with
    configurable weights.  Final signal is long (1) if weighted score > threshold.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {"momentum": 0.5, "mean_reversion": 0.25, "trend": 0.25}
        self.strategies = {
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "trend": TrendFollowingStrategy(),
        }

    def generate_signals(
        self,
        data: pd.DataFrame,
        all_params: dict,
        threshold: float = 0.4,
    ) -> pd.DataFrame:
        scored = None
        for name, strategy in self.strategies.items():
            logger.info("Running sub-strategy: %s", name)
            sig = strategy.generate_signals(data, all_params[name])
            w = self.weights[name]
            scored = sig * w if scored is None else scored + sig * w

        return (scored >= threshold).astype(int)
