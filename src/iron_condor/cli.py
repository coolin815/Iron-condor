"""Backtest CLI.

Examples:

    python -m iron_condor.cli --smoke
    python -m iron_condor.cli --days 365
    python -m iron_condor.cli --start 2024-05-13 --end 2025-05-12
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, run_sweep, simulate_day
from .config import StrategyParams
from .metrics import summarize_run, summarize_sweep
from .polygon_client import PolygonClient

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SPY 0DTE long iron condor backtester")
    p.add_argument("--smoke", action="store_true", help="Run a single recent day to verify wiring")
    p.add_argument("--days", type=int, default=365, help="Lookback window in calendar days (default 365)")
    p.add_argument("--start", type=_parse_date, help="Explicit start date YYYY-MM-DD")
    p.add_argument("--end", type=_parse_date, help="Explicit end date YYYY-MM-DD")
    p.add_argument("--sweep", action="store_true", help="Run the full parameter sweep (default: single config)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = PolygonClient()

    if args.smoke:
        # Use yesterday (or last trading day) to validate the pipeline end-to-end.
        from .backtest import _trading_days
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=10)
        days = _trading_days(start, end)
        if not days:
            print("No recent trading day found.", file=sys.stderr)
            return 1
        target = days[-1]
        params = StrategyParams()
        print(f"Smoke test on {target} with {params.strike_rule.name}, rsi {params.rsi_period}, pt {params.profit_target_pct:.0%}")
        result = simulate_day(target, params, params.starting_balance, client)
        print(result)
        return 0

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=args.days)

    print(f"Backtest window: {start} -> {end}")

    if args.sweep:
        sweep_df = run_sweep(start, end, client=client)
        sweep_df.to_csv(RESULTS_DIR / "sweep_trades.csv", index=False)
        summary = summarize_sweep(sweep_df, StrategyParams().starting_balance)
        summary.to_csv(RESULTS_DIR / "sweep_summary.csv", index=False)
        print("\n=== Sweep summary (top configs by return) ===")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(summary.head(20).to_string(index=False))
        print(f"\nFull output: {RESULTS_DIR}")
    else:
        params = StrategyParams()
        df = run_backtest(params, start, end, client=client)
        df.to_csv(RESULTS_DIR / "single_run_trades.csv", index=False)
        summary = summarize_run(df, params.starting_balance)
        print("\n=== Single-run summary ===")
        for k, v in summary.items():
            print(f"  {k:>22}: {v}")
        print(f"\nTrade log: {RESULTS_DIR / 'single_run_trades.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
