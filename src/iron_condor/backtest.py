"""Per-day backtest engine for the SPY 0DTE flow-following strategy.

Single-leg long option, copied from a large BUY-aggressor print on the
0DTE option tape. Entry at the NEXT 1-min bar's OPEN after the print.
Exit on PT / SL / time-stop measured on option mid price.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, time, timedelta
from math import floor
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .config import (
    PROFIT_TARGETS,
    SIZE_THRESHOLDS,
    STOP_LOSSES,
    TIME_STOPS,
    StrategyParams,
    UNDERLYING,
)
from .orb import Signal, find_signal
from .polygon_client import PolygonClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TradeResult:
    day: date
    # config
    pnl_mode: str
    size_threshold: int
    time_stop_min: int
    profit_target: float
    stop_loss: float
    skip_fridays: bool
    # signal
    print_time: datetime | None
    signal_time: datetime | None
    contract: str | None
    strike: float | None
    right: str | None
    print_size: int | None
    print_price: float | None
    # position
    qty: int
    entry_price: float | None       # ask, per share
    exit_price: float | None        # bid, per share
    exit_time: datetime | None
    minutes_held: float | None
    exit_reason: str
    gross_pnl: float
    fees: float
    net_pnl: float
    balance_after: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trading_days(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def _reindex_option_bars(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    if bars.empty:
        return bars
    et_index = bars.index.tz_convert("America/New_York")
    bars = bars.copy()
    bars.index = et_index
    start = pd.Timestamp(datetime.combine(day, time(9, 30))).tz_localize("America/New_York")
    end = pd.Timestamp(datetime.combine(day, time(16, 0))).tz_localize("America/New_York")
    grid = pd.date_range(start, end, freq="1min")
    return bars.reindex(grid).ffill()


def _leg_price(bars: pd.DataFrame, ts: pd.Timestamp, column: str = "close") -> float | None:
    try:
        v = bars.loc[ts, column]
    except KeyError:
        return None
    if pd.isna(v):
        return None
    return float(v)


# ---------------------------------------------------------------------------
# Single-day simulation
# ---------------------------------------------------------------------------


def simulate_day(
    day: date,
    params: StrategyParams,
    balance: float,
    client: PolygonClient,
) -> TradeResult:
    base = TradeResult(
        day=day,
        pnl_mode=params.pnl_mode,
        size_threshold=params.size_threshold,
        time_stop_min=params.time_stop_min,
        profit_target=params.profit_target_pct,
        stop_loss=params.stop_loss_pct,
        skip_fridays=params.skip_fridays,
        print_time=None,
        signal_time=None,
        contract=None,
        strike=None,
        right=None,
        print_size=None,
        print_price=None,
        qty=0,
        entry_price=None,
        exit_price=None,
        exit_time=None,
        minutes_held=None,
        exit_reason="no_signal",
        gross_pnl=0.0, fees=0.0, net_pnl=0.0,
        balance_after=balance,
    )
    if params.skip_fridays and day.weekday() == 4:
        base.exit_reason = "friday"
        return base

    today_bars = client.get_minute_bars(UNDERLYING, day)
    if today_bars.empty:
        base.exit_reason = "no_data"
        return base

    contracts = client.get_option_contracts(UNDERLYING, day)
    if not contracts:
        base.exit_reason = "no_data"
        return base

    signal = find_signal(today_bars, contracts, client, params, UNDERLYING)
    if signal is None:
        return base

    base.print_time = signal.print_timestamp.to_pydatetime()
    base.signal_time = signal.timestamp.to_pydatetime()
    base.contract = signal.contract
    base.strike = signal.strike
    base.right = signal.right
    base.print_size = signal.size
    base.print_price = signal.trade_price

    # Pull option's 1-min bars and reindex.
    opt_bars = _reindex_option_bars(
        client.get_option_minute_bars(signal.contract, day), day
    )
    if opt_bars.empty:
        base.exit_reason = "no_data"
        return base

    h = params.leg_half_spread
    entry_ts = pd.Timestamp(signal.timestamp).tz_convert("America/New_York").floor("min")
    entry_open = _leg_price(opt_bars, entry_ts, "open")
    if entry_open is None or entry_open <= 0.05:
        base.exit_reason = "no_data"
        return base
    entry_ask = entry_open + h

    capital = min(balance, params.max_capital_per_trade)
    per_contract_open = entry_ask * 100 + params.commission_per_contract
    qty = int(floor(capital / per_contract_open))
    if qty < 1:
        base.exit_reason = "no_data"
        return base
    base.qty = qty
    base.entry_price = entry_ask

    # Walk for exit
    time_stop_ts = entry_ts + pd.Timedelta(minutes=params.time_stop_min)
    hard_close_ts = pd.Timestamp(
        datetime.combine(day, params.hard_close)
    ).tz_localize("America/New_York")
    walk_end_ts = min(time_stop_ts, hard_close_ts)
    after = opt_bars.index[opt_bars.index > entry_ts]
    forward = [ts for ts in after if ts <= walk_end_ts]

    exit_ts: pd.Timestamp | None = None
    exit_bid: float | None = None
    exit_reason = "time_stop"
    use_gross = params.pnl_mode == "gross"
    for ts in forward:
        mid = _leg_price(opt_bars, ts, "close")
        if mid is None:
            continue
        bid = mid - h
        if use_gross:
            exit_pct = (mid - entry_open) / entry_open
        else:
            gross = (bid - entry_ask) * 100 * qty
            fees = 2 * params.commission_per_contract * qty
            net = gross - fees
            capital_deployed = entry_ask * 100 * qty + params.commission_per_contract * qty
            exit_pct = net / capital_deployed
        if exit_pct >= params.profit_target_pct:
            exit_ts, exit_bid, exit_reason = ts, bid, "profit"
            break
        if exit_pct <= -params.stop_loss_pct:
            exit_ts, exit_bid, exit_reason = ts, bid, "stop"
            break

    if exit_ts is None:
        cand = opt_bars.index[opt_bars.index <= walk_end_ts]
        if len(cand) == 0:
            base.exit_reason = "no_data"
            return base
        exit_ts = cand[-1]
        exit_mid = _leg_price(opt_bars, exit_ts, "close")
        if exit_mid is None:
            base.exit_reason = "no_data"
            return base
        exit_bid = exit_mid - h

    gross = (exit_bid - entry_ask) * 100 * qty
    fees = 2 * params.commission_per_contract * qty
    net = gross - fees
    base.exit_price = exit_bid
    base.exit_time = exit_ts.to_pydatetime()
    base.minutes_held = (exit_ts - entry_ts).total_seconds() / 60.0
    base.exit_reason = exit_reason
    base.gross_pnl = gross
    base.fees = fees
    base.net_pnl = net
    base.balance_after = balance + net
    return base


# ---------------------------------------------------------------------------
# Multi-day + sweep
# ---------------------------------------------------------------------------


def run_backtest(
    params: StrategyParams,
    start: date,
    end: date,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    client = client or PolygonClient()
    days = _trading_days(start, end)
    balance = params.starting_balance
    results: list[TradeResult] = []
    desc = (
        f"sz{params.size_threshold}"
        f"|pt{int(params.profit_target_pct*100)}"
        f"|sl{int(params.stop_loss_pct*100)}"
        f"|ts{params.time_stop_min}"
    )
    for day in tqdm(days, desc=desc):
        result = simulate_day(day, params, balance, client)
        balance = result.balance_after
        results.append(result)
    return pd.DataFrame([asdict(r) for r in results])


def run_sweep(
    start: date,
    end: date,
    size_thresholds: Iterable[int] = SIZE_THRESHOLDS,
    profit_targets: Iterable[float] = PROFIT_TARGETS,
    stop_losses: Iterable[float] = STOP_LOSSES,
    time_stops: Iterable[int] = TIME_STOPS,
    pnl_modes: Iterable[str] = ("gross",),
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [
        (sz, pt, sl, ts, pm)
        for sz in size_thresholds
        for pt in profit_targets
        for sl in stop_losses
        for ts in time_stops
        for pm in pnl_modes
    ]
    for sz, pt, sl, ts_min, pm in combos:
        params = StrategyParams(
            earliest_entry=base.earliest_entry,
            latest_entry=base.latest_entry,
            time_stop_min=ts_min,
            hard_close=base.hard_close,
            skip_fridays=base.skip_fridays,
            size_threshold=sz,
            strike_window=base.strike_window,
            pnl_mode=pm,
            profit_target_pct=pt,
            stop_loss_pct=sl,
            commission_per_contract=base.commission_per_contract,
            leg_half_spread=base.leg_half_spread,
            starting_balance=base.starting_balance,
            max_capital_per_trade=base.max_capital_per_trade,
        )
        df = run_backtest(params, start, end, client=client)
        df["config"] = (
            f"sz{sz}|pt{int(pt*100)}|sl{int(sl*100)}|ts{ts_min}|pnl={pm}"
        )
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
