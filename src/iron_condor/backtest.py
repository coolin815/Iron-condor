"""Per-day backtest engine for the SPY 20-min-candle trigger strategy.

For each trading day:
  1. Pull SPY 1-min bars; find the first 2-consecutive-direction 20-min pattern.
  2. If a signal fires, fetch the option chain for the chosen expiry (0DTE or 2DTE),
     pick the ATM strike, buy at the signal minute's option open.
  3. Walk minute-by-minute. Exit on whichever fires first:
       - profit target (% mid-to-mid in gross mode, % net after fees in net mode)
       - stop loss (same % convention as PT, paired by pnl_mode)
       - hard close (3:55 PM ET)
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
    DTE_VALUES,
    PROFIT_SCENARIOS,
    STOP_SCENARIOS,
    StrategyParams,
    UNDERLYING,
)
from .orb import Signal, _expiry_for_dte, find_signal
from .polygon_client import PolygonClient

log = logging.getLogger(__name__)


@dataclass
class TradeResult:
    day: date
    # config
    dte: int
    pnl_mode: str
    profit_target_pct: float
    stop_loss_pct: float
    # signal
    signal_time: datetime | None
    contract: str | None
    strike: float | None
    right: str | None
    direction: str | None
    expiry: date | None
    spot_at_entry: float | None
    # position
    qty: int
    entry_price: float | None
    exit_price: float | None
    exit_time: datetime | None
    minutes_held: float | None
    exit_reason: str
    gross_pnl: float
    fees: float
    net_pnl: float
    balance_after: float


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


def _empty_result(day: date, params: StrategyParams, balance: float, reason: str) -> TradeResult:
    return TradeResult(
        day=day,
        dte=params.dte,
        pnl_mode=params.pnl_mode,
        profit_target_pct=params.profit_target_pct,
        stop_loss_pct=params.stop_loss_pct,
        signal_time=None,
        contract=None, strike=None, right=None, direction=None,
        expiry=None, spot_at_entry=None,
        qty=0, entry_price=None, exit_price=None, exit_time=None,
        minutes_held=None, exit_reason=reason,
        gross_pnl=0.0, fees=0.0, net_pnl=0.0,
        balance_after=balance,
    )


def simulate_day(
    day: date,
    params: StrategyParams,
    balance: float,
    client: PolygonClient,
) -> TradeResult:
    today_bars = client.get_minute_bars(UNDERLYING, day)
    if today_bars.empty:
        return _empty_result(day, params, balance, "no_data")

    expiry = _expiry_for_dte(day, params.dte)
    contracts = client.get_option_contracts(UNDERLYING, expiry)
    if not contracts:
        return _empty_result(day, params, balance, "no_data")

    signal = find_signal(today_bars, contracts, client, params, UNDERLYING)
    if signal is None:
        return _empty_result(day, params, balance, "no_signal")

    opt_bars = _reindex_option_bars(
        client.get_option_minute_bars(signal.contract, day), day
    )
    if opt_bars.empty:
        return _empty_result(day, params, balance, "no_data")

    entry_ts = signal.timestamp
    entry_open = _leg_price(opt_bars, entry_ts, "open")
    if entry_open is None or entry_open <= 0.05:
        return _empty_result(day, params, balance, "no_data")

    h = params.leg_half_spread
    entry_mid = entry_open
    entry_ask = entry_open + h

    capital = min(balance, params.max_capital_per_trade)
    per_contract_open = entry_ask * 100 + params.commission_per_contract
    qty = int(floor(capital / per_contract_open))
    if qty < 1:
        return _empty_result(day, params, balance, "no_data")

    hard_close_ts = pd.Timestamp(
        datetime.combine(day, params.hard_close)
    ).tz_localize("America/New_York")
    forward = opt_bars.index[(opt_bars.index > entry_ts) & (opt_bars.index <= hard_close_ts)]
    use_net = params.pnl_mode == "net"

    exit_ts: pd.Timestamp | None = None
    exit_bid: float | None = None
    exit_reason = "hard_close"

    for ts in forward:
        mid = _leg_price(opt_bars, ts, "close")
        if mid is None:
            continue
        bid = mid - h
        if use_net:
            gross_val = (bid - entry_ask) * 100 * qty
            fees_val = 2 * params.commission_per_contract * qty
            net_val = gross_val - fees_val
            capital_deployed = entry_ask * 100 * qty + params.commission_per_contract * qty
            exit_pct = net_val / capital_deployed
        else:
            exit_pct = (mid - entry_mid) / entry_mid

        if exit_pct >= params.profit_target_pct:
            exit_ts, exit_bid, exit_reason = ts, bid, "profit"
            break
        if exit_pct <= -params.stop_loss_pct:
            exit_ts, exit_bid, exit_reason = ts, bid, "stop"
            break

    if exit_ts is None:
        cand = opt_bars.index[opt_bars.index <= hard_close_ts]
        if len(cand) == 0:
            return _empty_result(day, params, balance, "no_data")
        exit_ts = cand[-1]
        exit_mid = _leg_price(opt_bars, exit_ts, "close")
        if exit_mid is None:
            return _empty_result(day, params, balance, "no_data")
        exit_bid = exit_mid - h

    gross = (exit_bid - entry_ask) * 100 * qty
    fees = 2 * params.commission_per_contract * qty
    net = gross - fees

    return TradeResult(
        day=day,
        dte=params.dte,
        pnl_mode=params.pnl_mode,
        profit_target_pct=params.profit_target_pct,
        stop_loss_pct=params.stop_loss_pct,
        signal_time=entry_ts.to_pydatetime(),
        contract=signal.contract,
        strike=signal.strike,
        right=signal.right,
        direction=signal.direction,
        expiry=signal.expiry,
        spot_at_entry=signal.spot_at_entry,
        qty=qty,
        entry_price=entry_ask,
        exit_price=exit_bid,
        exit_time=exit_ts.to_pydatetime(),
        minutes_held=(exit_ts - entry_ts).total_seconds() / 60.0,
        exit_reason=exit_reason,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        balance_after=balance + net,
    )


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
        f"d{params.dte}|pt{int(params.profit_target_pct * 100)}{params.pnl_mode[0]}"
        f"|sl{int(params.stop_loss_pct * 100)}"
    )
    for day in tqdm(days, desc=desc):
        result = simulate_day(day, params, balance, client)
        balance = result.balance_after
        results.append(result)
    return pd.DataFrame([asdict(r) for r in results])


def run_sweep(
    start: date,
    end: date,
    dte_values: Iterable[int] = DTE_VALUES,
    profit_scenarios: Iterable[tuple[float, str]] = PROFIT_SCENARIOS,
    stop_scenarios: Iterable[float] = STOP_SCENARIOS,
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [(d, pt, mode, sl)
              for d in dte_values
              for (pt, mode) in profit_scenarios
              for sl in stop_scenarios]
    for d, pt, mode, sl in combos:
        params = StrategyParams(
            candle_minutes=base.candle_minutes,
            latest_entry=base.latest_entry,
            dte=d,
            strike_step=base.strike_step,
            hard_close=base.hard_close,
            profit_target_pct=pt,
            stop_loss_pct=sl,
            pnl_mode=mode,  # type: ignore[arg-type]
            commission_per_contract=base.commission_per_contract,
            leg_half_spread=base.leg_half_spread,
            starting_balance=base.starting_balance,
            max_capital_per_trade=base.max_capital_per_trade,
        )
        df = run_backtest(params, start, end, client=client)
        df["config"] = (
            f"d{d}|pt{int(pt*100)}{mode[0]}|sl{int(sl*100)}"
        )
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
