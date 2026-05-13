"""Per-day backtest engine and parameter sweep for the ORB strategy.

Single-leg 0DTE call/put. Entry at the option's ASK on the signal bar; exit at
the BID. All exit thresholds (profit, stop) are measured on **net P&L /
capital deployed** — i.e. they already include round-trip commissions and the
bid-ask spread.
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
    ConfluenceLevel,
    StrategyParams,
    UNDERLYING,
)
from .orb import ORBSignal, compute_levels, find_orb_signal, pick_atm_contract
from .polygon_client import PolygonClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TradeResult:
    day: date
    # signal config
    or_window_min: int
    confluence: str
    min_break_pct: float
    vol_mult: float
    vwap_filter: bool
    premarket_bias: bool
    profit_target: float
    stop_loss: float
    time_stop_min: int
    entry_cutoff: str
    # signal details
    signal_time: datetime | None
    signal_direction: str | None
    signal_spot: float | None
    orh: float | None
    orl: float | None
    pdh: float | None
    pdl: float | None
    level_broken: str | None
    # position
    contract: str | None
    strike: float | None
    qty: int
    entry_price: float | None     # ask, per share
    exit_price: float | None      # bid, per share
    exit_time: datetime | None
    minutes_held: float | None
    exit_reason: str              # 'no_signal' | 'no_data' | 'profit' | 'stop' | 'time'
    gross_pnl: float
    fees: float
    net_pnl: float
    balance_after: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trading_days(start: date, end: date) -> list[date]:
    """Business days in [start, end]. Holidays are skipped at fetch time."""
    return [d.date() for d in pd.bdate_range(start, end)]


def _previous_trading_day_with_data(
    day: date, client: PolygonClient, max_lookback: int = 5
) -> tuple[date | None, pd.DataFrame]:
    """Most recent prior trading day with non-empty SPY bars."""
    bdays = pd.bdate_range(
        end=pd.Timestamp(day) - pd.Timedelta(days=1), periods=max_lookback
    )
    for d in reversed(list(bdays)):
        bars = client.get_minute_bars(UNDERLYING, d.date())
        if not bars.empty:
            return d.date(), bars
    return None, pd.DataFrame()


def _reindex_minute_bars(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    """Reindex option bars to a 9:30-16:00 ET 1-min grid with forward-fill."""
    if bars.empty:
        return bars
    et_index = bars.index.tz_convert("America/New_York")
    bars = bars.copy()
    bars.index = et_index
    start = pd.Timestamp(
        datetime.combine(day, time(9, 30))
    ).tz_localize("America/New_York")
    end = pd.Timestamp(
        datetime.combine(day, time(16, 0))
    ).tz_localize("America/New_York")
    grid = pd.date_range(start, end, freq="1min")
    return bars.reindex(grid).ffill()


def _leg_mid(bars: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """1-min close as a proxy for mid."""
    try:
        v = bars.loc[ts, "close"]
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
    """Simulate one trading day with the ORB single-leg strategy."""
    base = TradeResult(
        day=day,
        or_window_min=params.or_window_min,
        confluence=params.confluence,
        min_break_pct=params.min_break_pct,
        vol_mult=params.vol_mult,
        vwap_filter=params.vwap_filter,
        premarket_bias=params.premarket_bias,
        profit_target=params.profit_target_pct,
        stop_loss=params.stop_loss_pct,
        time_stop_min=params.time_stop_min,
        entry_cutoff=params.latest_entry.strftime("%H:%M"),
        signal_time=None,
        signal_direction=None,
        signal_spot=None,
        orh=None, orl=None, pdh=None, pdl=None,
        level_broken=None,
        contract=None,
        strike=None,
        qty=0,
        entry_price=None,
        exit_price=None,
        exit_time=None,
        minutes_held=None,
        exit_reason="no_signal",
        gross_pnl=0.0, fees=0.0, net_pnl=0.0,
        balance_after=balance,
    )

    today_bars = client.get_minute_bars(UNDERLYING, day)
    if today_bars.empty:
        base.exit_reason = "no_data"
        return base
    _, yesterday_bars = _previous_trading_day_with_data(day, client)

    levels = compute_levels(today_bars, yesterday_bars, params.or_window_min)
    base.orh = levels.orh
    base.orl = levels.orl
    base.pdh = levels.pdh
    base.pdl = levels.pdl

    signal = find_orb_signal(today_bars, levels, params)
    if signal is None:
        return base

    base.signal_time = (
        signal.timestamp.to_pydatetime()
        if hasattr(signal.timestamp, "to_pydatetime") else signal.timestamp
    )
    base.signal_direction = signal.direction
    base.signal_spot = signal.spot
    base.level_broken = signal.level_broken

    # Pick the ATM call / put for today's 0DTE expiry.
    contracts = client.get_option_contracts(UNDERLYING, day)
    if not contracts:
        base.exit_reason = "no_data"
        return base
    pick = pick_atm_contract(signal, day, contracts, UNDERLYING)
    if pick is None:
        base.exit_reason = "no_data"
        return base
    ticker, strike = pick
    base.contract = ticker
    base.strike = strike

    # Fetch option's 1-min bars; reindex to the regular session.
    opt_bars = client.get_option_minute_bars(ticker, day)
    opt_bars = _reindex_minute_bars(opt_bars, day)
    if opt_bars.empty:
        base.exit_reason = "no_data"
        return base

    h = params.leg_half_spread
    entry_ts = pd.Timestamp(signal.timestamp).tz_convert(
        "America/New_York"
    ).floor("min")
    entry_mid = _leg_mid(opt_bars, entry_ts)
    if entry_mid is None or entry_mid <= 0.05:
        base.exit_reason = "no_data"
        return base
    entry_ask = entry_mid + h  # what you actually pay per share

    # Position sizing.
    capital = min(balance, params.max_capital_per_trade)
    per_contract_open = entry_ask * 100 + params.commission_per_contract
    qty = int(floor(capital / per_contract_open))
    if qty < 1:
        base.exit_reason = "no_data"
        return base
    base.qty = qty
    base.entry_price = entry_ask

    capital_deployed = entry_ask * 100 * qty + params.commission_per_contract * qty

    # Walk minute by minute looking for a net-P&L exit.
    time_stop_ts = entry_ts + pd.Timedelta(minutes=params.time_stop_min)
    hard_close_ts = pd.Timestamp(
        datetime.combine(day, params.hard_close)
    ).tz_localize("America/New_York")
    walk_end_ts = min(time_stop_ts, hard_close_ts)

    after = opt_bars.index[opt_bars.index > entry_ts]
    forward = [ts for ts in after if ts <= walk_end_ts]

    exit_ts: pd.Timestamp | None = None
    exit_bid: float | None = None
    exit_reason = "time"

    for ts in forward:
        mid = _leg_mid(opt_bars, ts)
        if mid is None:
            continue
        bid = mid - h
        gross = (bid - entry_ask) * 100 * qty
        fees = 2 * params.commission_per_contract * qty
        net = gross - fees
        net_pct = net / capital_deployed
        if net_pct >= params.profit_target_pct:
            exit_ts, exit_bid, exit_reason = ts, bid, "profit"
            break
        if net_pct <= -params.stop_loss_pct:
            exit_ts, exit_bid, exit_reason = ts, bid, "stop"
            break

    if exit_ts is None:
        # Time stop: use the last available bar at-or-before walk_end_ts.
        candidate = opt_bars.index[opt_bars.index <= walk_end_ts]
        if len(candidate) == 0:
            base.exit_reason = "no_data"
            return base
        exit_ts = candidate[-1]
        exit_mid = _leg_mid(opt_bars, exit_ts)
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
# Multi-day run + sweep
# ---------------------------------------------------------------------------


def run_backtest(
    params: StrategyParams,
    start: date,
    end: date,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    """Run a single strategy configuration over [start, end]. Returns trade log."""
    client = client or PolygonClient()
    days = _trading_days(start, end)
    balance = params.starting_balance
    results: list[TradeResult] = []
    desc = (
        f"or{params.or_window_min}"
        f"|conf={params.confluence}"
        f"|{_filter_tag(params.min_break_pct, params.vol_mult, params.vwap_filter, params.premarket_bias)}"
        f"|pt{int(params.profit_target_pct*100)}"
        f"|sl{int(params.stop_loss_pct*100)}"
        f"|ts{params.time_stop_min}"
        f"|co{params.latest_entry.strftime('%H%M')}"
    )
    for day in tqdm(days, desc=desc):
        result = simulate_day(day, params, balance, client)
        balance = result.balance_after
        results.append(result)
    return pd.DataFrame([asdict(r) for r in results])


def _filter_tag(mb: float, vm: float, vwap: bool, pmb: bool) -> str:
    """Compact label for the filter tuple."""
    parts = []
    if mb > 0:
        parts.append(f"mb{int(mb*10000)}bp")  # 0.001 -> "10bp"
    if vm > 0:
        parts.append(f"vol{vm:g}x")
    if vwap:
        parts.append("vwap")
    if pmb:
        parts.append("pmb")
    return "|".join(parts) if parts else "nofilt"


def run_sweep(
    start: date,
    end: date,
    or_windows: Iterable[int] = OR_WINDOWS,
    confluences: Iterable[ConfluenceLevel] = CONFLUENCE_LEVELS,
    profit_targets: Iterable[float] = PROFIT_TARGETS,
    stop_losses: Iterable[float] = STOP_LOSSES,
    time_stops: Iterable[int] = TIME_STOPS,
    entry_cutoffs=ENTRY_CUTOFFS,
    min_break_pcts: Iterable[float] = MIN_BREAK_PCTS,
    vol_mults: Iterable[float] = VOL_MULTS,
    vwap_filters: Iterable[bool] = VWAP_FILTERS,
    premarket_biases: Iterable[bool] = PREMARKET_BIASES,
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    """Run every combination of all sweep dimensions including filters."""
    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [
        (orw, cf, pt, sl, ts, co, mb, vm, vwap, pmb)
        for orw in or_windows
        for cf in confluences
        for pt in profit_targets
        for sl in stop_losses
        for ts in time_stops
        for co in entry_cutoffs
        for mb in min_break_pcts
        for vm in vol_mults
        for vwap in vwap_filters
        for pmb in premarket_biases
    ]
    for orw, cf, pt, sl, ts_min, co, mb, vm, vwap, pmb in combos:
        params = StrategyParams(
            or_window_min=orw,
            confluence=cf,
            earliest_entry=base.earliest_entry,
            latest_entry=co,
            time_stop_min=ts_min,
            hard_close=base.hard_close,
            min_break_pct=mb,
            vol_mult=vm,
            vwap_filter=vwap,
            premarket_bias=pmb,
            profit_target_pct=pt,
            stop_loss_pct=sl,
            commission_per_contract=base.commission_per_contract,
            leg_half_spread=base.leg_half_spread,
            starting_balance=base.starting_balance,
            max_capital_per_trade=base.max_capital_per_trade,
        )
        df = run_backtest(params, start, end, client=client)
        df["config"] = (
            f"or{orw}|conf={cf}"
            f"|{_filter_tag(mb, vm, vwap, pmb)}"
            f"|pt{int(pt*100)}|sl{int(sl*100)}|ts{ts_min}"
            f"|co{co.strftime('%H%M')}"
        )
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
