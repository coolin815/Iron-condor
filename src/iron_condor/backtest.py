"""Per-day backtest engine for the SPY 0DTE credit-spread strategy.

For each trading day, find an ORB-direction signal, build the 2-leg credit
spread, simulate entry at the credit (bid on short, ask on long), then walk
minute-by-minute checking exits on (entry_credit - current_value) / capital.
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
    STOP_LOSSES,
    TIME_STOPS,
    StrategyParams,
    UNDERLYING,
)
from .orb import (
    Signal,
    SpreadLegs,
    find_signal,
    pick_spread_legs,
    spread_width_dollars,
)
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
    time_stop_min: int
    entry_cutoff: str
    profit_target: float
    stop_loss: float
    skip_fridays: bool
    # signal
    signal_time: datetime | None
    signal_direction: str | None
    signal_spot: float | None
    orh: float | None
    orl: float | None
    # position
    short_strike: float | None
    long_strike: float | None
    right: str | None
    spread_width: float | None
    qty: int
    entry_credit: float | None      # per share, what you received
    exit_value: float | None        # per share, what you paid to close
    capital_deployed: float | None  # max loss in $
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


def _spread_mid(short_bars, long_bars, ts) -> float | None:
    s = _leg_price(short_bars, ts, "close")
    l = _leg_price(long_bars, ts, "close")
    if s is None or l is None:
        return None
    return s - l


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
        time_stop_min=params.time_stop_min,
        entry_cutoff=params.latest_entry.strftime("%H:%M"),
        profit_target=params.profit_target_pct,
        stop_loss=params.stop_loss_pct,
        skip_fridays=params.skip_fridays,
        signal_time=None,
        signal_direction=None,
        signal_spot=None,
        orh=None, orl=None,
        short_strike=None,
        long_strike=None,
        right=None,
        spread_width=None,
        qty=0,
        entry_credit=None,
        exit_value=None,
        capital_deployed=None,
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

    signal = find_signal(today_bars, params)
    if signal is None:
        return base

    base.signal_time = (
        signal.timestamp.to_pydatetime()
        if hasattr(signal.timestamp, "to_pydatetime") else signal.timestamp
    )
    base.signal_direction = signal.direction
    base.signal_spot = signal.spot
    base.orh = signal.orh
    base.orl = signal.orl

    contracts = client.get_option_contracts(UNDERLYING, day)
    if not contracts:
        base.exit_reason = "no_data"
        return base
    legs = pick_spread_legs(signal, day, contracts, params)
    if legs is None:
        base.exit_reason = "no_data"
        return base
    base.short_strike = legs.short_strike
    base.long_strike = legs.long_strike
    base.right = legs.right
    base.spread_width = spread_width_dollars(legs)

    tickers = legs.tickers(UNDERLYING)
    short_bars = _reindex_option_bars(
        client.get_option_minute_bars(tickers["short"], day), day
    )
    long_bars = _reindex_option_bars(
        client.get_option_minute_bars(tickers["long"], day), day
    )
    if short_bars.empty or long_bars.empty:
        base.exit_reason = "no_data"
        return base

    h = params.leg_half_spread
    entry_ts = pd.Timestamp(signal.timestamp).tz_convert("America/New_York").floor("min")
    short_open = _leg_price(short_bars, entry_ts, "open")
    long_open = _leg_price(long_bars, entry_ts, "open")
    if short_open is None or long_open is None:
        base.exit_reason = "no_data"
        return base

    # Realistic entry: SELL short at its BID, BUY long at its ASK.
    # Mid is the 1-min open; bid = mid - h, ask = mid + h.
    short_fill = short_open - h          # we sell, so we get the bid
    long_fill = long_open + h            # we buy, so we pay the ask
    entry_credit = short_fill - long_fill
    if entry_credit <= 0.0:
        base.exit_reason = "no_data"
        return base

    width = spread_width_dollars(legs)
    capital_per_spread = (width - entry_credit) * 100  # max loss in $
    if capital_per_spread <= 0:
        base.exit_reason = "no_data"
        return base

    # Sizing: full balance / capital_per_spread
    capital = min(balance, params.max_capital_per_trade)
    qty = int(floor(capital / capital_per_spread))
    if qty < 1:
        base.exit_reason = "no_data"
        return base

    base.qty = qty
    base.entry_credit = entry_credit
    base.capital_deployed = capital_per_spread * qty

    # Walk forward
    time_stop_ts = entry_ts + pd.Timedelta(minutes=params.time_stop_min)
    hard_close_ts = pd.Timestamp(
        datetime.combine(day, params.hard_close)
    ).tz_localize("America/New_York")
    walk_end_ts = min(time_stop_ts, hard_close_ts)
    after = short_bars.index[short_bars.index > entry_ts]
    forward = [ts for ts in after if ts <= walk_end_ts]

    exit_ts: pd.Timestamp | None = None
    exit_value: float | None = None
    exit_reason = "time_stop"
    use_gross = params.pnl_mode == "gross"

    for ts in forward:
        cur_mid = _spread_mid(short_bars, long_bars, ts)
        if cur_mid is None:
            continue
        # gross trigger uses mid-to-mid value; realized fill is at the
        # adverse side (close cost = pay short's ask, receive long's bid).
        if use_gross:
            # Net P&L per share = entry_credit - current_mid; capital uses entry credit
            pnl_per_share = entry_credit - cur_mid
            pnl_pct = pnl_per_share / (width - entry_credit)
        else:
            close_cost = cur_mid + 2 * h     # full spread to close
            pnl_per_share = entry_credit - close_cost
            pnl_pct = pnl_per_share / (width - entry_credit)
        if pnl_pct >= params.profit_target_pct:
            exit_ts, exit_value, exit_reason = ts, cur_mid, "profit"
            break
        if pnl_pct <= -params.stop_loss_pct:
            exit_ts, exit_value, exit_reason = ts, cur_mid, "stop"
            break

    if exit_ts is None:
        cand = short_bars.index[short_bars.index <= walk_end_ts]
        if len(cand) == 0:
            base.exit_reason = "no_data"
            return base
        exit_ts = cand[-1]
        exit_value = _spread_mid(short_bars, long_bars, exit_ts)
        if exit_value is None:
            base.exit_reason = "no_data"
            return base

    # Realized P&L at close: pay short's ask, receive long's bid
    close_cost_realized = exit_value + 2 * h
    # gross = (entry_credit - close_cost_realized) * 100 * qty
    gross = (entry_credit - close_cost_realized) * 100 * qty
    # Commissions: 2 legs × 2 sides × qty
    fees = 4 * params.commission_per_contract * qty
    net = gross - fees

    base.exit_value = exit_value
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
        f"pt{int(params.profit_target_pct*100)}"
        f"|sl{int(params.stop_loss_pct*100)}"
        f"|ts{params.time_stop_min}"
        f"|co{params.latest_entry.strftime('%H%M')}"
    )
    for day in tqdm(days, desc=desc):
        result = simulate_day(day, params, balance, client)
        balance = result.balance_after
        results.append(result)
    return pd.DataFrame([asdict(r) for r in results])


def run_sweep(
    start: date,
    end: date,
    profit_targets: Iterable[float] = PROFIT_TARGETS,
    stop_losses: Iterable[float] = STOP_LOSSES,
    time_stops: Iterable[int] = TIME_STOPS,
    pnl_modes: Iterable[str] = ("gross",),
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    """Sweep (pt × sl × time_stop × pnl_mode)."""
    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [
        (pt, sl, ts, pm)
        for pt in profit_targets
        for sl in stop_losses
        for ts in time_stops
        for pm in pnl_modes
    ]
    for pt, sl, ts_min, pm in combos:
        params = StrategyParams(
            or_window_min=base.or_window_min,
            earliest_entry=base.earliest_entry,
            latest_entry=base.latest_entry,
            time_stop_min=ts_min,
            hard_close=base.hard_close,
            skip_fridays=base.skip_fridays,
            short_strike_offset=base.short_strike_offset,
            spread_width=base.spread_width,
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
            f"pt{int(pt*100)}|sl{int(sl*100)}|ts{ts_min}|pnl={pm}"
        )
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
