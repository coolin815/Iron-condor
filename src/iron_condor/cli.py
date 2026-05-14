"""Backtest CLI for the SPY 0DTE flow-following strategy.

Examples:
    python -m iron_condor.cli --smoke
    python -m iron_condor.cli --days 365 --sweep
    python -m iron_condor.cli --sweep --size-threshold 1500 --pt 0.30
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
    PROFIT_TARGETS,
    SIZE_THRESHOLDS,
    STOP_LOSSES,
    TIME_STOPS,
    StrategyParams,
)
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
        description="SPY 0DTE flow-following backtester (large-print copy)"
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", type=_parse_date)
    p.add_argument("--end", type=_parse_date)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--size-threshold", type=int, action="append",
                   help=f"Minimum print size to copy (contracts). "
                        f"Default sweep: {list(SIZE_THRESHOLDS)}. Repeatable.")
    p.add_argument("--pt", type=float, action="append",
                   help=f"Profit target as fraction of option price (e.g. 0.30). "
                        f"Default: {list(PROFIT_TARGETS)}. Repeatable.")
    p.add_argument("--sl", type=float, action="append",
                   help=f"Stop loss as fraction. Default: {list(STOP_LOSSES)}. Repeatable.")
    p.add_argument("--time-stop", type=int, action="append",
                   help=f"Time stops in minutes. Default: {list(TIME_STOPS)}. Repeatable.")
    p.add_argument("--pnl-mode", type=_parse_pnl_mode, action="append",
                   help="P&L mode: gross (mid-to-mid) or net (after fills). Default: gross.")
    p.add_argument("--strike-window", type=float, default=None,
                   help="Half-width $ around opening spot for scanned strikes. Default: 5.0.")
    p.add_argument("--entry-mode", choices=["instant", "next_bar_open"], action="append",
                   help="Entry timing: instant (fill at print + spread) or next_bar_open "
                        "(fill at next minute's open). Default: sweep both.")
    p.add_argument("--include-fridays", action="store_true",
                   help="Override the default Friday-skip rule.")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = PolygonClient()

    overrides = {"skip_fridays": not args.include_fridays}
    if args.strike_window is not None:
        overrides["strike_window"] = args.strike_window
    # For smoke / single-config use first size threshold passed if any
    if args.size_threshold:
        overrides["size_threshold"] = args.size_threshold[0]
    base_params = StrategyParams(**overrides)

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
        print(
            f"Smoke test on {target} — size_threshold={base_params.size_threshold}, "
            f"strike_window=±${base_params.strike_window}, "
            f"pt={base_params.profit_target_pct:.0%}, sl={base_params.stop_loss_pct:.0%}"
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
        from .config import ENTRY_MODES
        size_thresholds = args.size_threshold or list(SIZE_THRESHOLDS)
        pts = args.pt or list(PROFIT_TARGETS)
        sls = args.sl or list(STOP_LOSSES)
        time_stops = args.time_stop or list(TIME_STOPS)
        pnl_modes = args.pnl_mode or [base_params.pnl_mode]
        entry_modes = args.entry_mode or list(ENTRY_MODES)
        n = (len(size_thresholds) * len(pts) * len(sls)
             * len(time_stops) * len(pnl_modes) * len(entry_modes))
        print(
            f"Sweep: {n} configs (size_threshold={size_thresholds}, "
            f"pt={pts}, sl={sls}, ts={time_stops}, pnl={pnl_modes}, "
            f"entry_mode={entry_modes}, "
            f"strike_window=±${base_params.strike_window})"
        )

        sweep_df = run_sweep(
            start, end,
            size_thresholds=size_thresholds,
            profit_targets=pts,
            stop_losses=sls,
            time_stops=time_stops,
            pnl_modes=pnl_modes,
            entry_modes=entry_modes,
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
