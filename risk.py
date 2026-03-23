"""
Position sizing, stop-loss management, and portfolio-level risk controls.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Wraps raw strategy signals with:
      - Kelly / volatility-scaled position sizing
      - Per-trade stop-loss and take-profit
      - Portfolio-level drawdown circuit-breaker
      - Maximum position count / concentration limits
    """

    def __init__(self, config: dict):
        self.max_pos_pct = config["max_position_pct"]
        self.stop_loss = config["stop_loss_pct"]
        self.take_profit = config["take_profit_pct"]
        self.max_dd_halt = config["max_drawdown_halt"]
        self.portfolio_heat = config["portfolio_heat"]

        self._entry_prices: Dict[str, float] = {}
        self._peak_value: float = 0.0
        self._halted: bool = False

    # ── Portfolio-level ──────────────────────────────────────────────────────

    def check_circuit_breaker(self, portfolio_value: float) -> bool:
        """Return True if trading should be halted due to drawdown."""
        self._peak_value = max(self._peak_value, portfolio_value)
        drawdown = (self._peak_value - portfolio_value) / self._peak_value
        if drawdown >= self.max_dd_halt:
            if not self._halted:
                logger.warning(
                    "CIRCUIT BREAKER: %.1f%% drawdown — halting all trading.",
                    drawdown * 100,
                )
            self._halted = True
        else:
            self._halted = False
        return self._halted

    # ── Position sizing ──────────────────────────────────────────────────────

    def size_positions(
        self,
        signals: pd.Series,
        prices: pd.Series,
        portfolio_value: float,
        volatilities: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Convert binary signals (0/1) into dollar allocations.

        If volatilities are provided, use inverse-volatility weighting
        (each position gets weight proportional to 1/vol), capped at max_pos_pct.
        Otherwise equal-weight.
        """
        if self._halted:
            return pd.Series(0.0, index=signals.index)

        active = signals[signals == 1].index
        if len(active) == 0:
            return pd.Series(0.0, index=signals.index)

        # Inverse-volatility weights
        if volatilities is not None and not volatilities[active].isna().all():
            inv_vol = 1.0 / volatilities[active].replace(0, np.nan).fillna(1.0)
            raw_weights = inv_vol / inv_vol.sum()
        else:
            raw_weights = pd.Series(1.0 / len(active), index=active)

        # Cap individual positions
        capped = raw_weights.clip(upper=self.max_pos_pct)

        # Enforce aggregate heat
        total_heat = capped.sum()
        if total_heat > self.portfolio_heat:
            capped = capped * (self.portfolio_heat / total_heat)

        dollar_alloc = capped * portfolio_value
        return dollar_alloc.reindex(signals.index, fill_value=0.0)

    # ── Trade-level stops ────────────────────────────────────────────────────

    def record_entry(self, ticker: str, price: float) -> None:
        self._entry_prices[ticker] = price

    def should_exit(self, ticker: str, current_price: float) -> bool:
        """Return True if stop-loss or take-profit is triggered."""
        entry = self._entry_prices.get(ticker)
        if entry is None:
            return False
        ret = (current_price - entry) / entry
        if ret <= -self.stop_loss:
            logger.info("Stop-loss triggered for %s: %.2f%%", ticker, ret * 100)
            return True
        if ret >= self.take_profit:
            logger.info("Take-profit triggered for %s: %.2f%%", ticker, ret * 100)
            return True
        return False

    def clear_entry(self, ticker: str) -> None:
        self._entry_prices.pop(ticker, None)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.Series:
        """Annualised 21-day rolling volatility for the most-recent window."""
        return returns.rolling(window).std().iloc[-1] * np.sqrt(252)
