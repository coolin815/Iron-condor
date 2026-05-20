"""Backtest CLI for the SPY 0DTE short put credit spread.

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
from .config import (
    FILTER_MODES,
    PROFIT_TARGETS,
    SHORT_OTM_PCTS,
    SPREAD_WIDTHS,
    STOP_LOSS_MULTS,
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="SPY 0DTE short put credit spread backtester"
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
        end = et_today - timedelta(days=1)
        start = end - timedelta(days=10)
        days = _trading_days(start, end)
        target = days[-1] if days else end
        print(
            f"Smoke test on {target} — entry={base_params.entry_time}, "
            f"short_otm={base_params.short_otm_pct:.1%}, "
            f"width=${base_params.spread_width:.0f}, "
            f"pt={base_params.profit_target_pct:.0%} of credit, "
            f"sl={base_params.stop_loss_mult:g}x credit"
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
        n = (
            len(FILTER_MODES) * len(SHORT_OTM_PCTS) * len(SPREAD_WIDTHS)
            * len(PROFIT_TARGETS) * len(STOP_LOSS_MULTS)
        )
        print(
            f"Sweep: {n} configs (filter_modes={[m[0] for m in FILTER_MODES]}, "
            f"short_otm={[f'{o*100:.1f}%' for o in SHORT_OTM_PCTS]}, "
            f"width={[f'${w:.0f}' for w in SPREAD_WIDTHS]}, "
            f"pt={[f'{p:.0%}' for p in PROFIT_TARGETS]}, "
            f"sl={[f'{s:g}x' for s in STOP_LOSS_MULTS]})"
        )
        sweep_df = run_sweep(
            start,
            end,
            base_params=base_params,
            client=client,
            checkpoint_dir=RESULTS_DIR,
        )
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
