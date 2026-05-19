"""Per-day backtest engine for the SPY 0DTE short put credit spread.

For each trading day:
  1. Pull SPY 1-min bars. Find signal at params.entry_time (spot snap + strike picks).
  2. Fetch the option chain (0DTE puts), build the two leg tickers, get their bars.
  3. Open the spread: sell short put at bid (= mid - h), buy long put at ask (= mid + h).
     Net credit per share = short_bid - long_ask.
  4. Position size: floor(capital / (max_loss_per_spread + commission_round_trip)).
  5. Walk forward minute by minute. The spread is worth `cost_to_close` per share:
       cost_to_close = (short_mid + h) - (long_mid - h)
     P&L per spread = (entry_credit - cost_to_close) * 100.
     Exit on PT (>= PT_pct * entry_credit_dollars),
             SL (<= -SL_mult * entry_credit_dollars),
             or hard close 3:55 PM.
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
    SHORT_OTM_PCTS,
    SPREAD_WIDTHS,
    STOP_LOSS_MULTS,
    StrategyParams,
    UNDERLYING,
)
from .orb import Signal, find_signal
from .polygon_client import PolygonClient

log = logging.getLogger(__name__)


@dataclass
class TradeResult:
    day: date
    # config
    short_otm_pct: float
    spread_width: float
    profit_target_pct: float
    stop_loss_mult: float
    # signal
    signal_time: datetime | None
    short_strike: float | None
    long_strike: float | None
    spot_at_entry: float | None
    # position
    qty: int
    entry_credit_per_share: float | None    # net credit received (positive)
    exit_cost_per_share: float | None       # cost to close spread (positive when adverse)
    exit_time: datetime | None
    minutes_held: float | None
    exit_reason: str
    gross_pnl: float                        # before commissions
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
        short_otm_pct=params.short_otm_pct,
        spread_width=params.spread_width,
        profit_target_pct=params.profit_target_pct,
        stop_loss_mult=params.stop_loss_mult,
        signal_time=None,
        short_strike=None, long_strike=None, spot_at_entry=None,
        qty=0,
        entry_credit_per_share=None, exit_cost_per_share=None,
        exit_time=None, minutes_held=None,
        exit_reason=reason,
        gross_pnl=0.0, fees=0.0, net_pnl=0.0,
        balance_after=balance,
    )


def _prior_trading_day(day: date) -> date:
    return (pd.Timestamp(day) - pd.tseries.offsets.BusinessDay(1)).date()


def _spy_prior_close(client: PolygonClient, day: date) -> float | None:
    """Return SPY's prior trading day regular-session close, or None."""
    prior = _prior_trading_day(day)
    bars = client.get_minute_bars(UNDERLYING, prior)
    if bars.empty:
        return None
    et = bars.copy()
    et.index = bars.index.tz_convert("America/New_York")
    rth = et[
        (et.index.time >= time(9, 30))
        & (et.index.time < time(16, 0))
    ]
    if rth.empty:
        return None
    last = rth.iloc[-1]["close"]
    if pd.isna(last):
        return None
    return float(last)


def simulate_day(
    day: date,
    params: StrategyParams,
    balance: float,
    client: PolygonClient,
) -> TradeResult:
    today_bars = client.get_minute_bars(UNDERLYING, day)
    if today_bars.empty:
        return _empty_result(day, params, balance, "no_data")

    # Regime filter: VIX prior-day close
    if params.vix_filter_enabled:
        vix = client.get_vix_close(_prior_trading_day(day))
        if vix is not None and vix > params.vix_max:
            log.debug("%s: VIX prior close %.2f > %.2f, skipping",
                      day, vix, params.vix_max)
            return _empty_result(day, params, balance, "vix_filter")

    # Regime filter: SPY overnight gap (today's 9:30 open vs prior 16:00 close)
    if params.gap_filter_enabled:
        prior_close = _spy_prior_close(client, day)
        if prior_close is not None:
            et = today_bars.copy()
            et.index = today_bars.index.tz_convert("America/New_York")
            today_open_bars = et[et.index.time == time(9, 30)]
            if not today_open_bars.empty:
                today_open = float(today_open_bars.iloc[0]["open"])
                gap_pct = (today_open - prior_close) / prior_close
                if gap_pct < params.gap_min_pct:
                    log.debug(
                        "%s: gap %.3f%% below threshold %.3f%%, skipping",
                        day, gap_pct * 100, params.gap_min_pct * 100,
                    )
                    return _empty_result(day, params, balance, "gap_filter")

    contracts = client.get_option_contracts(UNDERLYING, day)  # 0DTE
    if not contracts:
        return _empty_result(day, params, balance, "no_data")

    signal = find_signal(today_bars, contracts, client, params, UNDERLYING)
    if signal is None:
        return _empty_result(day, params, balance, "no_signal")

    short_bars = _reindex_option_bars(
        client.get_option_minute_bars(signal.short_ticker, day), day
    )
    long_bars = _reindex_option_bars(
        client.get_option_minute_bars(signal.long_ticker, day), day
    )
    if short_bars.empty or long_bars.empty:
        return _empty_result(day, params, balance, "no_data")

    entry_ts = signal.timestamp
    short_mid_entry = _leg_price(short_bars, entry_ts, "open")
    long_mid_entry = _leg_price(long_bars, entry_ts, "open")
    if short_mid_entry is None or long_mid_entry is None:
        return _empty_result(day, params, balance, "no_data")

    h = params.leg_half_spread
    # Sell short put at bid; buy long put at ask. Net credit per share.
    short_bid_entry = max(short_mid_entry - h, 0.0)
    long_ask_entry = long_mid_entry + h
    entry_credit = short_bid_entry - long_ask_entry  # per share

    if entry_credit <= 0:
        # Inversion (long leg priced richer than short — happens at very wide
        # OTM where both legs are near zero). No edge to take.
        return _empty_result(day, params, balance, "no_credit")

    spread_width_dollars = (signal.short_strike - signal.long_strike) * 100
    max_loss_per_spread = spread_width_dollars - entry_credit * 100
    commission_round_trip = params.commission_per_contract * 4  # 2 legs × open + close

    capital = min(balance, params.max_capital_per_trade)
    per_spread_capital = max_loss_per_spread + commission_round_trip
    qty = int(floor(capital / per_spread_capital)) if per_spread_capital > 0 else 0
    if qty < 1:
        return _empty_result(day, params, balance, "no_data")

    entry_credit_dollars = entry_credit * 100
    pt_target_dollars = params.profit_target_pct * entry_credit_dollars
    sl_threshold_dollars = -params.stop_loss_mult * entry_credit_dollars

    hard_close_ts = pd.Timestamp(
        datetime.combine(day, params.hard_close)
    ).tz_localize("America/New_York")
    forward = short_bars.index[
        (short_bars.index > entry_ts) & (short_bars.index <= hard_close_ts)
    ]

    exit_ts: pd.Timestamp | None = None
    exit_cost: float | None = None
    exit_reason = "hard_close"

    for ts in forward:
        short_mid = _leg_price(short_bars, ts, "close")
        long_mid = _leg_price(long_bars, ts, "close")
        if short_mid is None or long_mid is None:
            continue
        # To close: buy back short at ask, sell long at bid.
        short_ask_close = short_mid + h
        long_bid_close = max(long_mid - h, 0.0)
        cost_to_close = short_ask_close - long_bid_close  # per share, positive normally
        pnl_per_spread = (entry_credit - cost_to_close) * 100

        if pnl_per_spread >= pt_target_dollars:
            exit_ts, exit_cost, exit_reason = ts, cost_to_close, "profit"
            break
        if pnl_per_spread <= sl_threshold_dollars:
            exit_ts, exit_cost, exit_reason = ts, cost_to_close, "stop"
            break

    if exit_ts is None:
        cand = short_bars.index[short_bars.index <= hard_close_ts]
        if len(cand) == 0:
            return _empty_result(day, params, balance, "no_data")
        exit_ts = cand[-1]
        short_mid = _leg_price(short_bars, exit_ts, "close")
        long_mid = _leg_price(long_bars, exit_ts, "close")
        if short_mid is None or long_mid is None:
            return _empty_result(day, params, balance, "no_data")
        short_ask_close = short_mid + h
        long_bid_close = max(long_mid - h, 0.0)
        exit_cost = short_ask_close - long_bid_close

    pnl_per_spread = (entry_credit - exit_cost) * 100
    gross = pnl_per_spread * qty
    fees = commission_round_trip * qty
    net = gross - fees

    return TradeResult(
        day=day,
        short_otm_pct=params.short_otm_pct,
        spread_width=params.spread_width,
        profit_target_pct=params.profit_target_pct,
        stop_loss_mult=params.stop_loss_mult,
        signal_time=entry_ts.to_pydatetime(),
        short_strike=signal.short_strike,
        long_strike=signal.long_strike,
        spot_at_entry=signal.spot_at_entry,
        qty=qty,
        entry_credit_per_share=entry_credit,
        exit_cost_per_share=exit_cost,
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
        f"otm{int(params.short_otm_pct * 1000)/10:.1f}|w{int(params.spread_width)}"
        f"|pt{int(params.profit_target_pct * 100)}|sl{params.stop_loss_mult:g}x"
    )
    for day in tqdm(days, desc=desc):
        result = simulate_day(day, params, balance, client)
        balance = result.balance_after
        results.append(result)
    return pd.DataFrame([asdict(r) for r in results])


def run_sweep(
    start: date,
    end: date,
    short_otm_pcts: Iterable[float] = SHORT_OTM_PCTS,
    spread_widths: Iterable[float] = SPREAD_WIDTHS,
    profit_targets: Iterable[float] = PROFIT_TARGETS,
    stop_loss_mults: Iterable[float] = STOP_LOSS_MULTS,
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [(otm, w, pt, sl)
              for otm in short_otm_pcts
              for w in spread_widths
              for pt in profit_targets
              for sl in stop_loss_mults]
    for otm, w, pt, sl in combos:
        params = StrategyParams(
            entry_time=base.entry_time,
            hard_close=base.hard_close,
            short_otm_pct=otm,
            spread_width=w,
            strike_step=base.strike_step,
            vix_filter_enabled=base.vix_filter_enabled,
            vix_max=base.vix_max,
            gap_filter_enabled=base.gap_filter_enabled,
            gap_min_pct=base.gap_min_pct,
            profit_target_pct=pt,
            stop_loss_mult=sl,
            commission_per_contract=base.commission_per_contract,
            leg_half_spread=base.leg_half_spread,
            starting_balance=base.starting_balance,
            max_capital_per_trade=base.max_capital_per_trade,
        )
        df = run_backtest(params, start, end, client=client)
        df["config"] = (
            f"otm{int(otm * 1000)/10:.1f}|w{int(w)}"
            f"|pt{int(pt * 100)}|sl{sl:g}x"
        )
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
