"""
Data fetching and caching utilities.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _cache_path(ticker: str, start: str, end: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.parquet")


def fetch_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted OHLCV data for a list of tickers.

    Returns a MultiIndex DataFrame: (Date, Ticker) → [Open, High, Low, Close, Volume].
    """
    frames = []
    for ticker in tickers:
        path = _cache_path(ticker, start, end)
        if use_cache and os.path.exists(path):
            df = pd.read_parquet(path)
            logger.debug("Cache hit: %s", ticker)
        else:
            try:
                raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if raw.empty:
                    logger.warning("No data returned for %s", ticker)
                    continue
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                if use_cache:
                    df.to_parquet(path)
            except Exception as exc:
                logger.error("Failed to download %s: %s", ticker, exc)
                continue

        df["Ticker"] = ticker
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined.reset_index().rename(columns={"index": "Date", "Price": "Date"})
    # Handle yfinance returning 'Date' or 'Datetime' as column name
    if "Date" not in combined.columns and "Datetime" in combined.columns:
        combined = combined.rename(columns={"Datetime": "Date"})
    combined = combined.set_index(["Date", "Ticker"]).sort_index()
    return combined


def fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Return daily close prices for the benchmark."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return raw["Close"].squeeze()


def get_sp500_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot OHLCV MultiIndex data to a (Date × Ticker) returns DataFrame.
    """
    close = data["Close"].unstack("Ticker")
    returns = close.pct_change().dropna(how="all")
    return returns
