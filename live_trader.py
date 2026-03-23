"""
Live / paper trading execution via Alpaca Markets API.

Usage:
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env, then:

        python live_trader.py --strategy ensemble --dry-run
"""
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-trade-api not installed — live trading disabled.")


class LiveTrader:
    """
    Thin wrapper around the Alpaca REST API.

    In dry-run mode all order calls are logged but not sent.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.api = None

        if not dry_run:
            if not ALPACA_AVAILABLE:
                raise RuntimeError("Install alpaca-trade-api to enable live trading.")
            self.api = tradeapi.REST(
                key_id=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
            )
            logger.info("Connected to Alpaca (%s)", os.getenv("ALPACA_BASE_URL"))

    # ── Account ──────────────────────────────────────────────────────────────

    def get_portfolio_value(self) -> float:
        if self.dry_run:
            return 100_000.0
        account = self.api.get_account()
        return float(account.portfolio_value)

    def get_positions(self) -> Dict[str, float]:
        if self.dry_run:
            return {}
        positions = self.api.list_positions()
        return {p.symbol: float(p.qty) for p in positions}

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,          # "buy" | "sell"
        notional: float,    # dollar amount
    ) -> None:
        logger.info(
            "[%s] %s %s — notional $%.2f",
            "DRY-RUN" if self.dry_run else "LIVE",
            side.upper(),
            ticker,
            notional,
        )
        if self.dry_run:
            return
        try:
            self.api.submit_order(
                symbol=ticker,
                notional=round(notional, 2),
                side=side,
                type="market",
                time_in_force="day",
            )
        except Exception as exc:
            logger.error("Order failed for %s: %s", ticker, exc)

    def close_position(self, ticker: str) -> None:
        logger.info("[%s] CLOSE %s", "DRY-RUN" if self.dry_run else "LIVE", ticker)
        if self.dry_run:
            return
        try:
            self.api.close_position(ticker)
        except Exception as exc:
            logger.error("Close failed for %s: %s", ticker, exc)

    # ── Execution loop ───────────────────────────────────────────────────────

    def execute_signals(
        self,
        target_signals: pd.Series,      # ticker → 0 or 1
        dollar_allocations: pd.Series,  # ticker → dollar amount
    ) -> None:
        current_positions = self.get_positions()
        desired = set(target_signals[target_signals == 1].index)

        # Close positions no longer signalled
        for ticker in list(current_positions.keys()):
            if ticker not in desired:
                self.close_position(ticker)

        # Open new positions
        for ticker in desired:
            if ticker not in current_positions:
                alloc = dollar_allocations.get(ticker, 0)
                if alloc > 1.0:
                    self.place_order(ticker, "buy", alloc)

    def is_market_open(self) -> bool:
        if self.dry_run:
            return True
        clock = self.api.get_clock()
        return clock.is_open

    def wait_for_market_open(self, poll_seconds: int = 60) -> None:
        while not self.is_market_open():
            logger.info("Market closed — waiting %ds …", poll_seconds)
            time.sleep(poll_seconds)
