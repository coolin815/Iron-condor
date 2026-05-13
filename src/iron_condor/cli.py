"""Backtest CLI for the SPY 0DTE breakout+reversal strategy.

Examples:

    python -m iron_condor.cli --smoke
    python -m iron_condor.cli --sweep                  # 30 days, all signal modes
    python -m iron_condor.cli --days 365 --sweep
    python -m iron_condor.cli --sweep --mode breakout  # only breakout
    python -m iron_condor.cli --sweep --pnl-mode gross # mid-price exits (default)
    python -m iron_condor.cli --sweep --pnl-mode net   # net-of-fees exits
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
    SIGNAL_MODES,
    TIME_STOPS,
    StrategyParams,
)
from .metrics import summarize_run, summarize_sweep
from .polygon_client import PolygonClient

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

_VALID_MODES = {"both", "breakout", "reversal"}
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


def _parse_mode(s: str) -> str:
    if s not in _VALID_MODES:
        raise argparse.ArgumentTypeError(
            f"invalid mode {s!r}; must be one of {sorted(_VALID_MODES)}"
        )
    return s


def _parse_pnl_mode(s: str) -> str:
    if s not in _VALID_PNL_MODES:
        raise argparse.ArgumentTypeError(
            f"invalid pnl-mode {s!r}; must be one of {sorted(_VALID_PNL_MODES)}"
        )
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SPY 0DTE Breakout + Reversal backtester"
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", type=_parse_date)
    p.add_argument("--end", type=_parse_date)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")

    p.add_argument("--mode", type=_parse_mode, action="append",
                   help=f"Signal mode: {sorted(_VALID_MODES)}. Default: all 3. Repeatable.")
    p.add_argument("--time-stop", type=int, action="append",
                   help=f"Time stops in minutes. Default: {list(TIME_STOPS)}. Repeatable.")
    p.add_argument("--pnl-mode", type=_parse_pnl_mode, action="append",
                   help="P&L measurement mode: gross (mid-price) or net (after fees). "
                        "Default: gross. Repeatable to sweep both.")
    p.add_argument("--breakout-co", type=_parse_time,
                   help="Latest breakout entry time HH:MM ET. Default: 13:00 (10:00 PT).")
    p.add_argument("--reversal-co", type=_parse_time,
                   help="Latest reversal entry time HH:MM ET. Default: 12:30 (9:30 PT).")
    p.add_argument("--include-fridays", action="store_true",
                   help="Override the default Friday-skip rule.")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = PolygonClient()

    # Build base params with any CLI overrides for per-signal cutoffs
    overrides = {"skip_fridays": not args.include_fridays}
    if args.breakout_co is not None:
        overrides["breakout_latest_entry"] = args.breakout_co
    if args.reversal_co is not None:
        overrides["reversal_latest_entry"] = args.reversal_co
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
            f"Smoke test on {target} — mode={base_params.signal_mode}, "
            f"pnl_mode={base_params.pnl_mode}, "
            f"breakout_co={base_params.breakout_latest_entry}, "
            f"reversal_co={base_params.reversal_latest_entry}, "
            f"pt={base_params.profit_target_pct:.0%}, sl={base_params.stop_loss_pct:.0%}, "
            f"time_stop={base_params.time_stop_min}m"
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
        modes = args.mode or list(SIGNAL_MODES)
        time_stops = args.time_stop or list(TIME_STOPS)
        pnl_modes = args.pnl_mode or [base_params.pnl_mode]
        n = len(modes) * len(time_stops) * len(pnl_modes)
        print(
            f"Sweep: {n} configs "
            f"(modes={modes}, ts={time_stops}, pnl={pnl_modes}, "
            f"breakout_co={base_params.breakout_latest_entry.strftime('%H:%M')}, "
            f"reversal_co={base_params.reversal_latest_entry.strftime('%H:%M')}, "
            f"skip_fridays={base_params.skip_fridays})"
        )

        sweep_df = run_sweep(
            start, end,
            signal_modes=modes,
            time_stops=time_stops,
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
