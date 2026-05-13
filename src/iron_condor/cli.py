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
    MIN_BREAK_PCTS,
    OR_WINDOWS,
    PREMARKET_BIASES,
    PROFIT_TARGETS,
    STOP_LOSSES,
    TIME_STOPS,
    VOL_MULTS,
    VWAP_FILTERS,
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


def _parse_yes_no(s: str) -> bool:
    s = s.lower()
    if s in ("yes", "y", "true", "on", "1"):
        return True
    if s in ("no", "n", "false", "off", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"invalid yes/no value {s!r}; use yes/no"
    )


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

    # Filter sweep dimensions. Repeatable to sweep multiple values.
    p.add_argument("--min-break", type=float, action="append",
                   help="Min break magnitude as fraction past ORH/ORL (e.g. 0.001 = 10bp). "
                        "Default: 0 (no filter). Repeatable.")
    p.add_argument("--vol-mult", type=float, action="append",
                   help="Require break-bar volume >= N * 20-bar avg (e.g. 1.5). "
                        "Default: 0 (no filter). Repeatable.")
    p.add_argument("--vwap", type=_parse_yes_no, action="append",
                   help="VWAP-alignment filter (yes/no). Default: no. Repeatable to sweep both.")
    p.add_argument("--pm-bias", type=_parse_yes_no, action="append",
                   help="Premarket-direction filter (yes/no). Default: no. Repeatable.")

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
        min_break_pcts = args.min_break or list(MIN_BREAK_PCTS)
        vol_mults = args.vol_mult or list(VOL_MULTS)
        vwap_filters = args.vwap or list(VWAP_FILTERS)
        premarket_biases = args.pm_bias or list(PREMARKET_BIASES)

        n = (
            len(or_windows) * len(confluences) * len(profit_targets)
            * len(stop_losses) * len(time_stops) * len(entry_cutoffs)
            * len(min_break_pcts) * len(vol_mults)
            * len(vwap_filters) * len(premarket_biases)
        )
        print(
            f"Sweep: {n} configs "
            f"(or={or_windows}, conf={confluences}, "
            f"pt={profit_targets}, sl={stop_losses}, "
            f"ts={time_stops}, "
            f"co={[c.isoformat(timespec='minutes') for c in entry_cutoffs]}, "
            f"min_break={min_break_pcts}, vol_mult={vol_mults}, "
            f"vwap={vwap_filters}, pm_bias={premarket_biases})"
        )

        sweep_df = run_sweep(
            start, end,
            or_windows=or_windows,
            confluences=confluences,
            profit_targets=profit_targets,
            stop_losses=stop_losses,
            time_stops=time_stops,
            entry_cutoffs=entry_cutoffs,
            min_break_pcts=min_break_pcts,
            vol_mults=vol_mults,
            vwap_filters=vwap_filters,
            premarket_biases=premarket_biases,
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
