"""Backtest CLI for the SPY 0DTE candle-pattern strategy."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import run_backtest, run_sweep, simulate_day
from .config import ENTRY_CUTOFFS, TIME_STOPS, StrategyParams
from .metrics import summarize_run, summarize_sweep
from .polygon_client import PolygonClient

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
_VALID_PNL_MODES = {"gross", "net"}


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


def _parse_pnl_mode(s: str) -> str:
    if s not in _VALID_PNL_MODES:
        raise argparse.ArgumentTypeError(
            f"invalid pnl-mode {s!r}; must be one of {sorted(_VALID_PNL_MODES)}"
        )
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SPY 0DTE Candle-Pattern backtester (10 patterns, 5-min)"
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", type=_parse_date)
    p.add_argument("--end", type=_parse_date)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--time-stop", type=int, action="append",
                   help=f"Time stops in minutes. Default: {list(TIME_STOPS)}. Repeatable.")
    p.add_argument("--co", type=_parse_time, action="append",
                   help="Entry cutoffs HH:MM ET. Default: 11:30, 13:00, 15:00. Repeatable.")
    p.add_argument("--pnl-mode", type=_parse_pnl_mode, action="append",
                   help="P&L measurement: gross (mid-to-mid) or net (after fees). "
                        "Default: gross. Repeatable.")
    p.add_argument("--include-fridays", action="store_true",
                   help="Override the default Friday-skip rule.")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = PolygonClient()

    base_params = StrategyParams(skip_fridays=not args.include_fridays)

    if args.smoke:
        from .backtest import _trading_days
        et_today = datetime.now(ZoneInfo("America/New_York")).date()
        end = et_today - timedelta(days=1)
        start = end - timedelta(days=10)
        days = _trading_days(start, end)
        target = None
        for d in reversed(days):
            if not (base_params.skip_fridays and d.weekday() == 4):
                target = d
                break
        if target is None:
            target = days[-1] if days else end
        print(f"Smoke test on {target}")
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
        time_stops = args.time_stop or list(TIME_STOPS)
        entry_cutoffs = args.co or list(ENTRY_CUTOFFS)
        pnl_modes = args.pnl_mode or [base_params.pnl_mode]
        n = len(time_stops) * len(entry_cutoffs) * len(pnl_modes)
        print(
            f"Sweep: {n} configs (ts={time_stops}, "
            f"co={[c.isoformat(timespec='minutes') for c in entry_cutoffs]}, "
            f"pnl={pnl_modes}, skip_fridays={base_params.skip_fridays})"
        )

        sweep_df = run_sweep(
            start, end,
            time_stops=time_stops,
            entry_cutoffs=entry_cutoffs,
            pnl_modes=pnl_modes,
            base_params=base_params,
            client=client,
        )
        sweep_df.to_csv(RESULTS_DIR / "sweep_trades.csv", index=False)
        summary = summarize_sweep(sweep_df, base_params.starting_balance)
        summary.to_csv(RESULTS_DIR / "sweep_summary.csv", index=False)
        print("\n=== Sweep summary (top configs by return) ===")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(summary.head(20).to_string(index=False))
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
