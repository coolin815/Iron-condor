"""Backtest CLI.

Examples:

    # Smoke test on most recent trading day
    python -m iron_condor.cli --smoke

    # Full 840-config sweep
    python -m iron_condor.cli --days 30 --sweep

    # Targeted sweep: just the winning family from prior runs
    python -m iron_condor.cli --days 30 --sweep --strike fixed_1.0x3 --rsi 14

    # Single specific config
    python -m iron_condor.cli --days 30 --sweep \\
        --strike fixed_1.0x3 --rsi 14 --pt 0.20 --sl 0.50 --co 12:30
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
    ENTRY_CUTOFFS,
    PROFIT_TARGETS,
    RSI_PERIODS,
    RSI_THRESHOLDS,
    STOP_LOSSES,
    STRIKE_RULES,
    StrategyParams,
)
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


def _parse_time(s: str) -> time:
    """Parse HH:MM (24-hour, ET)."""
    return time.fromisoformat(s)


def _parse_rsi_thresh(s: str) -> tuple[float, float]:
    """Parse 'upper/lower' like '75/25' into a (upper, lower) tuple."""
    up_s, lo_s = s.split("/", 1)
    return (float(up_s), float(lo_s))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SPY 0DTE long iron condor backtester")
    p.add_argument("--smoke", action="store_true", help="Run a single recent day to verify wiring")
    p.add_argument("--days", type=int, default=365, help="Lookback window in calendar days (default 365)")
    p.add_argument("--start", type=_parse_date, help="Explicit start date YYYY-MM-DD")
    p.add_argument("--end", type=_parse_date, help="Explicit end date YYYY-MM-DD")
    p.add_argument("--sweep", action="store_true", help="Run a parameter sweep (default: single config)")
    p.add_argument("--verbose", "-v", action="store_true")

    # Sweep dimension filters. Repeat the flag to include multiple values.
    p.add_argument("--rsi", type=int, action="append",
                   help="Restrict sweep to these RSI periods (default: 9, 14). Repeatable.")
    p.add_argument("--rsi-thresh", type=_parse_rsi_thresh, action="append",
                   help="Restrict to these RSI thresholds 'upper/lower' (e.g. 75/25). "
                        "Default: 70/30, 75/25, 80/20. Repeatable.")
    p.add_argument("--strike", action="append",
                   help="Restrict to these strike-rule names. e.g. fixed_1.0x3. Repeatable.")
    p.add_argument("--pt", type=float, action="append",
                   help="Restrict to these profit targets (e.g. 0.20). Repeatable.")
    p.add_argument("--sl", type=float, action="append",
                   help="Restrict to these stop losses (e.g. 0.50). Repeatable.")
    p.add_argument("--co", type=_parse_time, action="append",
                   help="Restrict to these entry cutoffs HH:MM ET (e.g. 12:30). Repeatable.")

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
        print(f"Smoke test on {target} with {params.strike_rule.name}, rsi {params.rsi_period}, pt {params.profit_target_pct:.0%}")
        result = simulate_day(target, params, params.starting_balance, client)
        print(result)
        return 0

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        # Use ET-aware date so Colab (which runs in UTC) doesn't pick "today"
        # when ET is still on the previous day. We always end on yesterday ET
        # to ensure Polygon has the full day's aggregates ingested.
        et_today = datetime.now(ZoneInfo("America/New_York")).date()
        end = et_today - timedelta(days=1)
        start = end - timedelta(days=args.days)

    print(f"Backtest window: {start} -> {end}")

    if args.sweep:
        # Apply filters; missing flag means "use all defaults".
        rsi_periods = args.rsi or list(RSI_PERIODS)
        rsi_thresholds = args.rsi_thresh or list(RSI_THRESHOLDS)
        profit_targets = args.pt or list(PROFIT_TARGETS)
        stop_losses = args.sl or list(STOP_LOSSES)
        entry_cutoffs = args.co or list(ENTRY_CUTOFFS)
        if args.strike:
            wanted = set(args.strike)
            strike_rules = [r for r in STRIKE_RULES if r.name in wanted]
            missing = wanted - {r.name for r in strike_rules}
            if missing:
                print(f"WARN: unknown strike rule(s) ignored: {missing}", file=sys.stderr)
        else:
            strike_rules = list(STRIKE_RULES)

        n = (
            len(rsi_periods) * len(rsi_thresholds) * len(strike_rules)
            * len(profit_targets) * len(stop_losses) * len(entry_cutoffs)
        )
        print(
            f"Sweep: {n} configs "
            f"(rsi={rsi_periods}, thresh={[f'{int(u)}/{int(l)}' for u,l in rsi_thresholds]}, "
            f"strikes={[r.name for r in strike_rules]}, "
            f"pt={profit_targets}, sl={stop_losses}, "
            f"co={[c.isoformat(timespec='minutes') for c in entry_cutoffs]})"
        )

        sweep_df = run_sweep(
            start, end,
            rsi_periods=rsi_periods,
            rsi_thresholds=rsi_thresholds,
            strike_rules=strike_rules,
            profit_targets=profit_targets,
            stop_losses=stop_losses,
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
