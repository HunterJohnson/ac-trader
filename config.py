"""
Central configuration for the S&P 500 trading algorithm.
"""

# --- Universe ---
SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG", "HD", "JNJ",
    "COST", "MRK", "CVX", "ABBV", "WMT", "CRM", "BAC", "MCD", "NFLX",
    "AMD", "PEP", "TMO", "ACN", "LIN", "ADBE", "DIS", "CSCO", "TXN",
    "WFC", "NKE", "DHR", "NEE", "RTX", "AMGN", "BMY", "LOW", "QCOM",
    "HON", "INTC", "UPS", "CAT",
]

# Benchmark
BENCHMARK = "SPY"

# --- Strategy parameters ---
STRATEGY_PARAMS = {
    "momentum": {
        "lookback_days": 126,       # ~6 months
        "rebalance_freq": "monthly",
        "top_n": 10,                # hold top N momentum stocks
        "skip_days": 21,            # skip most-recent month (reversal)
    },
    "mean_reversion": {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "holding_days": 5,
    },
    "trend_following": {
        "fast_ema": 50,
        "slow_ema": 200,
        "atr_period": 14,
        "atr_multiplier": 2.0,     # trailing stop = ATR * multiplier
    },
}

# --- Risk management ---
RISK = {
    "max_position_pct": 0.05,      # max 5 % of portfolio per position
    "stop_loss_pct": 0.03,         # hard stop-loss
    "take_profit_pct": 0.09,       # take-profit target
    "max_drawdown_halt": 0.20,     # halt trading if drawdown exceeds 20 %
    "portfolio_heat": 0.06,        # max aggregate at-risk capital
}

# --- Backtest defaults ---
BACKTEST = {
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100_000,
    "commission_pct": 0.001,       # 0.1 % per trade (round-trip ~0.2 %)
    "slippage_pct": 0.0005,
}
