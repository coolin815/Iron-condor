"""Backtest CLI for the SPY 0DTE Opening Range Breakout strategy.

Examples:

    # Smoke test on most recent trading day
    python -m iron_condor.cli --smoke

    # Default sweep over 30 days (216 configs, ~5 min with warm cache)
    python -m iron_condor.cli --sweep

    # Narrowed sweep on the most promising config family
    python -m iron_condor.cli --sweep --or-window 15 --confluence any

    # Pin a single config
    python -m iron_condor.cli --sweep --or-window 15 --confluence any \\
        --pt 0.05 --sl 0.10 --time-stop 30 --co 11:00
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import run_backtest, run_sweep, simulate_day
from .config import (
    CONFLUENCE_LEVELS,
    ENTRY_CUTOFFS,
    OR_WINDOWS,
    PROFIT_TARGETS,
    STOP_LOSSES,
    TIME_STOPS,
    StrategyParams,
)
from .metrics import summarize_run, summarize_sweep
from .polygon_client import PolygonClient

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

_VALID_CONFLUENCES = {"none", "pdh_pdl", "pmh_pml", "onh_onl", "any"}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _parse_time(s: str) -> time:
    return time.fromisoformat(s)


def _parse_confluence(s: str) -> str:
    if s not in _VALID_CONFLUENCES:
        raise argparse.ArgumentTypeError(
            f"invalid confluence {s!r}; must be one of {sorted(_VALID_CONFLUENCES)}"
        )
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SPY 0DTE Opening Range Breakout backtester")
    p.add_argument("--smoke", action="store_true", help="Run a single recent day to verify wiring")
    p.add_argument("--days", type=int, default=30, help="Lookback window in calendar days (default 30; pass 365 for a year)")
    p.add_argument("--start", type=_parse_date, help="Explicit start date YYYY-MM-DD")
    p.add_argument("--end", type=_parse_date, help="Explicit end date YYYY-MM-DD")
    p.add_argument("--sweep", action="store_true", help="Run a parameter sweep (default: single config)")
    p.add_argument("--verbose", "-v", action="store_true")

    # Sweep dimension filters. Each is repeatable.
    p.add_argument("--or-window", type=int, action="append",
                   help=f"Restrict to these OR windows in minutes (default: {list(OR_WINDOWS)}).")
    p.add_argument("--confluence", type=_parse_confluence, action="append",
                   help=f"Restrict to these confluence rules. Default: {list(CONFLUENCE_LEVELS)}.")
    p.add_argument("--pt", type=float, action="append",
                   help=f"Net profit targets (e.g. 0.05). Default: {list(PROFIT_TARGETS)}.")
    p.add_argument("--sl", type=float, action="append",
                   help=f"Net stop losses (e.g. 0.10). Default: {list(STOP_LOSSES)}.")
    p.add_argument("--time-stop", type=int, action="append",
                   help=f"Time stops in minutes. Default: {list(TIME_STOPS)}.")
    p.add_argument("--co", type=_parse_time, action="append",
                   help="Entry cutoffs HH:MM ET (e.g. 11:00). Repeatable.")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = PolygonClient()

    if args.smoke:
        from .backtest import _trading_days
        et_today = datetime.now(ZoneInfo("America/New_York")).date()
        end = et_today - timedelta(days=1)
        start = end - timedelta(days=10)
        days = _trading_days(start, end)
        if not days:
            print("No recent trading day found.", file=sys.stderr)
            return 1
        target = days[-1]
        params = StrategyParams()
        print(
            f"Smoke test on {target} — OR{params.or_window_min}, "
            f"confluence={params.confluence}, "
            f"pt={params.profit_target_pct:.0%}, sl={params.stop_loss_pct:.0%}, "
            f"time_stop={params.time_stop_min}m"
        )
        result = simulate_day(target, params, params.starting_balance, client)
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
        or_windows = args.or_window or list(OR_WINDOWS)
        confluences = args.confluence or list(CONFLUENCE_LEVELS)
        profit_targets = args.pt or list(PROFIT_TARGETS)
        stop_losses = args.sl or list(STOP_LOSSES)
        time_stops = args.time_stop or list(TIME_STOPS)
        entry_cutoffs = args.co or list(ENTRY_CUTOFFS)

        n = (
            len(or_windows) * len(confluences) * len(profit_targets)
            * len(stop_losses) * len(time_stops) * len(entry_cutoffs)
        )
        print(
            f"Sweep: {n} configs "
            f"(or={or_windows}, conf={confluences}, "
            f"pt={profit_targets}, sl={stop_losses}, "
            f"ts={time_stops}, "
            f"co={[c.isoformat(timespec='minutes') for c in entry_cutoffs]})"
        )

        sweep_df = run_sweep(
            start, end,
            or_windows=or_windows,
            confluences=confluences,
            profit_targets=profit_targets,
            stop_losses=stop_losses,
            time_stops=time_stops,
            entry_cutoffs=entry_cutoffs,
            client=client,
        )
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
