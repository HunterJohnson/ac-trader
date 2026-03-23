"""
Performance analytics and charting.
"""
import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from backtest import BacktestResult

logger = logging.getLogger(__name__)


def print_metrics(result: BacktestResult, benchmark_returns: Optional[pd.Series] = None) -> None:
    """Pretty-print performance metrics to stdout."""
    m = result.metrics
    print("\n" + "=" * 50)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Total Return    : {m['total_return_pct']:>8.2f} %")
    print(f"  CAGR            : {m['cagr_pct']:>8.2f} %")
    print(f"  Ann. Volatility : {m['ann_volatility_pct']:>8.2f} %")
    print(f"  Sharpe Ratio    : {m['sharpe_ratio']:>8.3f}")
    print(f"  Sortino Ratio   : {m['sortino_ratio']:>8.3f}")
    if m.get("calmar_ratio") is not None:
        print(f"  Calmar Ratio    : {m['calmar_ratio']:>8.3f}")
    print(f"  Max Drawdown    : {m['max_drawdown_pct']:>8.2f} %")
    print(f"  Win Rate        : {m['win_rate_pct']:>8.2f} %")

    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(result.daily_returns.index).fillna(0)
        excess = result.daily_returns - bm
        info_ratio = (excess.mean() * 252) / (excess.std() * np.sqrt(252) + 1e-9)
        beta = np.cov(result.daily_returns, bm)[0, 1] / (np.var(bm) + 1e-9)
        alpha = (result.daily_returns.mean() - beta * bm.mean()) * 252
        print(f"  Beta vs SPY     : {beta:>8.3f}")
        print(f"  Alpha (ann.)    : {alpha * 100:>8.2f} %")
        print(f"  Info. Ratio     : {info_ratio:>8.3f}")
    print("=" * 50 + "\n")


def plot_results(
    result: BacktestResult,
    benchmark_prices: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> None:
    """Four-panel performance chart."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("S&P 500 Algorithm — Backtest Results", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    equity = result.equity_curve
    returns = result.daily_returns

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    norm_eq = equity / equity.iloc[0] * 100
    ax1.plot(norm_eq.index, norm_eq.values, label="Strategy", linewidth=1.5)
    if benchmark_prices is not None:
        bm = benchmark_prices.reindex(equity.index, method="ffill")
        norm_bm = bm / bm.iloc[0] * 100
        ax1.plot(norm_bm.index, norm_bm.values, label="SPY", linestyle="--",
                 linewidth=1.2, alpha=0.8)
    ax1.set_title("Equity Curve (rebased to 100)")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max * 100
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color="red")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)

    # 3. Monthly returns heatmap
    ax3 = fig.add_subplot(gs[2, 0])
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100,
    })
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    im = ax3.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                    vmin=-10, vmax=10)
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(pivot.columns, fontsize=7, rotation=45)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index, fontsize=7)
    ax3.set_title("Monthly Returns (%)")
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # 4. Return distribution
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(returns * 100, bins=60, edgecolor="white", color="steelblue", alpha=0.8)
    ax4.axvline(0, color="black", linewidth=0.8)
    ax4.set_title("Daily Return Distribution")
    ax4.set_xlabel("Daily Return (%)")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Chart saved to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
