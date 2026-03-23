"""
Technical indicator helpers built on top of pandas / numpy.
No external TA library required — pure vectorised implementations.
"""
import numpy as np
import pandas as pd


# ── Moving Averages ──────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ── Momentum / Trend ─────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def momentum_score(series: pd.Series, lookback: int = 126, skip: int = 21) -> float:
    """
    Risk-adjusted momentum: total return over (lookback - skip) days,
    normalised by volatility.
    """
    if len(series) < lookback:
        return np.nan
    window = series.iloc[-(lookback):-skip] if skip else series.iloc[-lookback:]
    ret = window.iloc[-1] / window.iloc[0] - 1
    vol = window.pct_change().std() * np.sqrt(252)
    return ret / vol if vol > 0 else np.nan


# ── Volatility ───────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


# ── Volume ───────────────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ── Composite signal helpers ─────────────────────────────────────────────────

def z_score(series: pd.Series, window: int = 20) -> pd.Series:
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)
