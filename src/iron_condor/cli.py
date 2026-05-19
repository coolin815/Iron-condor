"""Backtest CLI for the SPY 20-min-candle trigger strategy.

Examples:
    python -m iron_condor.cli --smoke
    python -m iron_condor.cli --days 30 --sweep
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import run_backtest, run_sweep, simulate_day
from .config import DTE_VALUES, PROFIT_SCENARIOS, STOP_SCENARIOS, StrategyParams
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
    p = argparse.ArgumentParser(
        description="SPY 20-min-candle trigger backtester"
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", type=_parse_date)
    p.add_argument("--end", type=_parse_date)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = PolygonClient()
    base_params = StrategyParams()

    if args.smoke:
        from .backtest import _trading_days
        et_today = datetime.now(ZoneInfo("America/New_York")).date()
        # Pick a day with buffer so 2DTE expiry is past
        end = et_today - timedelta(days=5)
        start = end - timedelta(days=10)
        days = _trading_days(start, end)
        target = days[-1] if days else end
        print(
            f"Smoke test on {target} — candle={base_params.candle_minutes}min, "
            f"latest_entry={base_params.latest_entry}, dte={base_params.dte}, "
            f"pt={base_params.profit_target_pct:.0%}({base_params.pnl_mode}), "
            f"sl={base_params.stop_loss_pct:.0%}"
        )
        result = simulate_day(target, base_params, base_params.starting_balance, client)
        print(result)
        return 0

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        et_today = datetime.now(ZoneInfo("America/New_York")).date()
        end = et_today - timedelta(days=1)
        start = end - timedelta(days=args.days)
    print(f"Backtest window: {start} -> {end}")

    if args.sweep:
        n = len(DTE_VALUES) * len(PROFIT_SCENARIOS) * len(STOP_SCENARIOS)
        print(
            f"Sweep: {n} configs "
            f"(DTE={list(DTE_VALUES)}, "
            f"PT={[f'{int(pt*100)}%({m})' for pt, m in PROFIT_SCENARIOS]}, "
            f"SL={[f'{int(s*100)}%' for s in STOP_SCENARIOS]})"
        )
        sweep_df = run_sweep(start, end, base_params=base_params, client=client)
        sweep_df.to_csv(RESULTS_DIR / "sweep_trades.csv", index=False)
        summary = summarize_sweep(sweep_df, base_params.starting_balance)
        summary.to_csv(RESULTS_DIR / "sweep_summary.csv", index=False)
        print("\n=== Sweep summary (top configs by return) ===")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(summary.head(40).to_string(index=False))
        print(f"\nFull output: {RESULTS_DIR}")
    else:
        df = run_backtest(base_params, start, end, client=client)
        df.to_csv(RESULTS_DIR / "single_run_trades.csv", index=False)
        summary = summarize_run(df, base_params.starting_balance)
        print("\n=== Single-run summary ===")
        for k, v in summary.items():
            print(f"  {k:>22}: {v}")
        print(f"\nTrade log: {RESULTS_DIR / 'single_run_trades.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
