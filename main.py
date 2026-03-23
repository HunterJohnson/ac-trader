"""
Entry-point for the S&P 500 trading algorithm.

Modes:
    backtest  — run historical simulation and print metrics / charts
    live      — run live/paper trading loop using Alpaca (requires .env)
    scan      — print today's signals without trading

Examples:
    python main.py backtest --strategy ensemble --plot
    python main.py backtest --strategy momentum
    python main.py live --strategy ensemble --dry-run
    python main.py scan --strategy trend
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta

import pandas as pd

import config
from data import fetch_ohlcv, fetch_benchmark, get_sp500_returns
from strategies import MomentumStrategy, MeanReversionStrategy, TrendFollowingStrategy, EnsembleStrategy
from backtest import Backtester
from risk import RiskManager
from analytics import print_metrics, plot_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "momentum": MomentumStrategy(),
    "mean_reversion": MeanReversionStrategy(),
    "trend": TrendFollowingStrategy(),
    "ensemble": EnsembleStrategy(),
}


def get_signals(strategy_name: str, data: pd.DataFrame) -> pd.DataFrame:
    strategy = STRATEGY_MAP[strategy_name]
    if strategy_name == "ensemble":
        return strategy.generate_signals(data, config.STRATEGY_PARAMS)
    return strategy.generate_signals(data, config.STRATEGY_PARAMS[
        {"momentum": "momentum", "mean_reversion": "mean_reversion", "trend": "trend_following"}[strategy_name]
    ])


def run_backtest(args) -> None:
    logger.info("Fetching data for backtest (%s → %s)…", config.BACKTEST["start_date"], config.BACKTEST["end_date"])

    # Extend fetch window for indicator warm-up (extra 300 trading days)
    fetch_start = (
        datetime.strptime(config.BACKTEST["start_date"], "%Y-%m-%d") - timedelta(days=300)
    ).strftime("%Y-%m-%d")

    data = fetch_ohlcv(
        config.SP500_TICKERS,
        start=fetch_start,
        end=config.BACKTEST["end_date"],
    )

    if data.empty:
        logger.error("No data fetched — check your internet connection.")
        sys.exit(1)

    logger.info("Generating %s signals…", args.strategy)
    signals = get_signals(args.strategy, data)

    # Trim to official backtest window
    signals = signals.loc[config.BACKTEST["start_date"]:]

    logger.info("Running backtest…")
    backtester = Backtester(config.BACKTEST, config.RISK)
    result = backtester.run(signals, data)

    benchmark = fetch_benchmark(
        config.BENCHMARK,
        start=config.BACKTEST["start_date"],
        end=config.BACKTEST["end_date"],
    )
    bm_returns = benchmark.pct_change().fillna(0)

    print_metrics(result, bm_returns)

    if not result.trades.empty:
        print(f"  Total trades    : {len(result.trades)}")
        print(f"  Unique tickers  : {result.trades['ticker'].nunique()}")

    if args.plot:
        plot_results(result, benchmark, save_path=args.save_chart)


def run_live(args) -> None:
    from live_trader import LiveTrader

    trader = LiveTrader(dry_run=args.dry_run)

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    logger.info("Fetching recent data for signal generation…")
    data = fetch_ohlcv(config.SP500_TICKERS, start=start, end=end)
    if data.empty:
        logger.error("No data — cannot generate signals.")
        sys.exit(1)

    logger.info("Generating signals…")
    signals = get_signals(args.strategy, data)
    latest_signals = signals.iloc[-1]

    portfolio_value = trader.get_portfolio_value()
    risk_mgr = RiskManager(config.RISK)

    close_latest = data["Close"].unstack("Ticker").iloc[-1]
    returns_recent = get_sp500_returns(data)
    vols = RiskManager.rolling_volatility(returns_recent)

    dollar_allocs = risk_mgr.size_positions(
        latest_signals, close_latest, portfolio_value, vols
    )

    logger.info("Active signals: %s", list(latest_signals[latest_signals == 1].index))
    trader.execute_signals(latest_signals, dollar_allocs)


def run_scan(args) -> None:
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    logger.info("Fetching data…")
    data = fetch_ohlcv(config.SP500_TICKERS, start=start, end=end)
    if data.empty:
        logger.error("No data returned.")
        sys.exit(1)

    signals = get_signals(args.strategy, data)
    latest = signals.iloc[-1]
    active = sorted(latest[latest == 1].index.tolist())

    print(f"\nStrategy   : {args.strategy}")
    print(f"Date       : {signals.index[-1].date()}")
    print(f"Long tickers ({len(active)}): {', '.join(active) if active else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S&P 500 Trading Algorithm")
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- backtest
    p_bt = sub.add_parser("backtest", help="Run historical backtest")
    p_bt.add_argument("--strategy", choices=list(STRATEGY_MAP), default="ensemble")
    p_bt.add_argument("--plot", action="store_true", help="Show performance charts")
    p_bt.add_argument("--save-chart", default=None, metavar="FILE",
                      help="Save chart to file instead of displaying")

    # -- live
    p_live = sub.add_parser("live", help="Run live/paper trading")
    p_live.add_argument("--strategy", choices=list(STRATEGY_MAP), default="ensemble")
    p_live.add_argument("--dry-run", action="store_true", default=True,
                        help="Log orders but don't submit (default: True)")
    p_live.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Submit real orders to Alpaca")

    # -- scan
    p_scan = sub.add_parser("scan", help="Print today's signals")
    p_scan.add_argument("--strategy", choices=list(STRATEGY_MAP), default="ensemble")

    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "live":
        run_live(args)
    elif args.mode == "scan":
        run_scan(args)


if __name__ == "__main__":
    main()
