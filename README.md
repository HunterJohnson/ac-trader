# S&P 500 Trading Algorithm

A Python algorithmic trading system for S&P 500 equities with three strategies, a vectorised backtester, risk management, and optional live trading via Alpaca.

## Features

| Component | Description |
|---|---|
| **Momentum** | Cross-sectional 6-month risk-adjusted momentum, monthly rebalance |
| **Mean Reversion** | RSI + Bollinger Bands oversold entries, 5-day hold |
| **Trend Following** | Dual EMA golden/death cross with ATR trailing stop |
| **Ensemble** | Weighted combination of all three strategies |
| **Risk Manager** | Inverse-vol sizing, stop-loss/take-profit, drawdown circuit-breaker |
| **Backtester** | Daily event-driven engine with commission + slippage |
| **Live Trader** | Alpaca paper/live API integration |
| **Analytics** | Equity curve, drawdown, monthly heatmap, return distribution |

## Quick Start

```bash
pip install -r requirements.txt
```

### Backtest

```bash
# Run ensemble strategy backtest with charts
python main.py backtest --strategy ensemble --plot

# Available strategies: momentum | mean_reversion | trend | ensemble
python main.py backtest --strategy momentum
```

### Scan (today's signals, no trading)

```bash
python main.py scan --strategy ensemble
```

### Paper / Live Trading

```bash
cp .env.example .env
# Fill in your Alpaca API keys in .env

# Dry-run (logs orders, doesn't submit)
python main.py live --strategy ensemble --dry-run

# Submit to Alpaca paper account
python main.py live --strategy ensemble --no-dry-run
```

## File Structure

```
ac-trader/
├── main.py           # CLI entry-point
├── config.py         # Tickers, strategy params, risk settings
├── data.py           # yfinance data fetching with parquet cache
├── indicators.py     # Pure-pandas technical indicators
├── strategies.py     # Momentum, MeanReversion, TrendFollowing, Ensemble
├── risk.py           # Position sizing, stops, circuit-breaker
├── backtest.py       # Daily event-driven backtester
├── live_trader.py    # Alpaca execution layer
├── analytics.py      # Performance metrics + charts
├── requirements.txt
└── .env.example
```

## Risk Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always paper-trade before risking real capital.
