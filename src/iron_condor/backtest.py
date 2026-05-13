"""Per-day backtest engine and parameter sweep for the breakout-and-reversal
strategy.

Single-leg 0DTE call/put. Entry at the option's ASK on the signal bar; exit at
the BID. Exit thresholds (profit, stop) are measured on **net P&L /
capital deployed** — round-trip commissions and the bid-ask spread are baked
into entry_price / exit_price.
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
    ENTRY_CUTOFFS,
    SIGNAL_MODES,
    TIME_STOPS,
    SignalMode,
    StrategyParams,
    UNDERLYING,
)
from .orb import Signal, find_signal, opening_range, pick_atm_contract
from .polygon_client import PolygonClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TradeResult:
    day: date
    # config
    signal_mode: str
    time_stop_min: int
    entry_cutoff: str
    skip_fridays: bool
    # signal details
    signal_time: datetime | None
    signal_type: str | None        # 'breakout' or 'reversal'
    signal_direction: str | None   # 'call' or 'put'
    signal_spot: float | None
    orh: float | None
    orl: float | None
    vwap_at_signal: float | None
    ema9_at_signal: float | None
    ema21_at_signal: float | None
    rsi_cross_day_at_signal: float | None
    rsi_intraday_at_signal: float | None
    # position
    contract: str | None
    strike: float | None
    qty: int
    entry_price: float | None      # ask, per share
    exit_price: float | None       # bid, per share
    exit_time: datetime | None
    minutes_held: float | None
    exit_reason: str               # 'no_signal' | 'no_data' | 'profit' | 'stop' | 'time_stop' | 'friday'
    gross_pnl: float
    fees: float
    net_pnl: float
    balance_after: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trading_days(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def _previous_trading_day_with_data(
    day: date, client: PolygonClient, max_lookback: int = 5
) -> tuple[date | None, pd.DataFrame]:
    bdays = pd.bdate_range(
        end=pd.Timestamp(day) - pd.Timedelta(days=1), periods=max_lookback
    )
    for d in reversed(list(bdays)):
        bars = client.get_minute_bars(UNDERLYING, d.date())
        if not bars.empty:
            return d.date(), bars
    return None, pd.DataFrame()


def _reindex_option_bars(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    """Reindex option bars to a 9:30-16:00 ET 1-min grid with forward-fill."""
    if bars.empty:
        return bars
    et_index = bars.index.tz_convert("America/New_York")
    bars = bars.copy()
    bars.index = et_index
    start = pd.Timestamp(datetime.combine(day, time(9, 30))).tz_localize("America/New_York")
    end = pd.Timestamp(datetime.combine(day, time(16, 0))).tz_localize("America/New_York")
    grid = pd.date_range(start, end, freq="1min")
    return bars.reindex(grid).ffill()


def _leg_mid(bars: pd.DataFrame, ts: pd.Timestamp) -> float | None:
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
    """Simulate one trading day with the breakout+reversal strategy."""
    base = TradeResult(
        day=day,
        signal_mode=params.signal_mode,
        time_stop_min=params.time_stop_min,
        entry_cutoff=params.latest_entry.strftime("%H:%M"),
        skip_fridays=params.skip_fridays,
        signal_time=None,
        signal_type=None,
        signal_direction=None,
        signal_spot=None,
        orh=None, orl=None,
        vwap_at_signal=None,
        ema9_at_signal=None,
        ema21_at_signal=None,
        rsi_cross_day_at_signal=None,
        rsi_intraday_at_signal=None,
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

    # Friday skip — short-circuit before any data fetching
    if params.skip_fridays and day.weekday() == 4:
        base.exit_reason = "friday"
        return base

    today_bars = client.get_minute_bars(UNDERLYING, day)
    if today_bars.empty:
        base.exit_reason = "no_data"
        return base
    _, yesterday_bars = _previous_trading_day_with_data(day, client)

    # OR levels for the trade record
    levels = opening_range(today_bars, params.or_window_min)
    base.orh = levels.orh
    base.orl = levels.orl

    signal = find_signal(today_bars, yesterday_bars, params)
    if signal is None:
        return base

    base.signal_time = (
        signal.timestamp.to_pydatetime()
        if hasattr(signal.timestamp, "to_pydatetime") else signal.timestamp
    )
    base.signal_type = signal.signal_type
    base.signal_direction = signal.direction
    base.signal_spot = signal.spot
    base.vwap_at_signal = signal.vwap
    base.ema9_at_signal = signal.ema9
    base.ema21_at_signal = signal.ema21
    base.rsi_cross_day_at_signal = signal.rsi_cross_day
    base.rsi_intraday_at_signal = signal.rsi_intraday

    # Pick the ATM call/put for today's 0DTE expiry
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

    opt_bars = client.get_option_minute_bars(ticker, day)
    opt_bars = _reindex_option_bars(opt_bars, day)
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
    entry_ask = entry_mid + h

    capital = min(balance, params.max_capital_per_trade)
    per_contract_open = entry_ask * 100 + params.commission_per_contract
    qty = int(floor(capital / per_contract_open))
    if qty < 1:
        base.exit_reason = "no_data"
        return base
    base.qty = qty
    base.entry_price = entry_ask
    capital_deployed = entry_ask * 100 * qty + params.commission_per_contract * qty

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
        cand = opt_bars.index[opt_bars.index <= walk_end_ts]
        if len(cand) == 0:
            base.exit_reason = "no_data"
            return base
        exit_ts = cand[-1]
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
    client = client or PolygonClient()
    days = _trading_days(start, end)
    balance = params.starting_balance
    results: list[TradeResult] = []
    desc = (
        f"mode={params.signal_mode}"
        f"|ts{params.time_stop_min}"
        f"|co{params.latest_entry.strftime('%H%M')}"
        f"|pt{int(params.profit_target_pct*100)}"
        f"|sl{int(params.stop_loss_pct*100)}"
    )
    for day in tqdm(days, desc=desc):
        result = simulate_day(day, params, balance, client)
        balance = result.balance_after
        results.append(result)
    return pd.DataFrame([asdict(r) for r in results])


def run_sweep(
    start: date,
    end: date,
    signal_modes: Iterable[SignalMode] = SIGNAL_MODES,
    time_stops: Iterable[int] = TIME_STOPS,
    entry_cutoffs=ENTRY_CUTOFFS,
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    """Sweep (signal_mode × time_stop × entry_cutoff). PT and SL are fixed."""
    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [
        (sm, ts, co)
        for sm in signal_modes
        for ts in time_stops
        for co in entry_cutoffs
    ]
    for sm, ts_min, co in combos:
        params = StrategyParams(
            or_window_min=base.or_window_min,
            earliest_entry=base.earliest_entry,
            latest_entry=co,
            time_stop_min=ts_min,
            hard_close=base.hard_close,
            rsi_long_thresh=base.rsi_long_thresh,
            rsi_short_thresh=base.rsi_short_thresh,
            rsi_extreme_high=base.rsi_extreme_high,
            rsi_extreme_low=base.rsi_extreme_low,
            reversal_call_skip_lo=base.reversal_call_skip_lo,
            reversal_call_skip_hi=base.reversal_call_skip_hi,
            skip_fridays=base.skip_fridays,
            signal_mode=sm,
            profit_target_pct=base.profit_target_pct,
            stop_loss_pct=base.stop_loss_pct,
            commission_per_contract=base.commission_per_contract,
            leg_half_spread=base.leg_half_spread,
            starting_balance=base.starting_balance,
            max_capital_per_trade=base.max_capital_per_trade,
        )
        df = run_backtest(params, start, end, client=client)
        df["config"] = (
            f"mode={sm}|ts{ts_min}|co{co.strftime('%H%M')}"
        )
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
