"""Per-day backtest engine and parameter sweep."""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from math import floor
from typing import Iterable

import exchange_calendars as xcals
import pandas as pd
from tqdm import tqdm

from .config import (
    ENTRY_CUTOFFS,
    PROFIT_TARGETS,
    RSI_PERIODS,
    RSI_THRESHOLDS,
    STOP_LOSSES,
    STRIKE_RULES,
    StrategyParams,
    StrikeRule,
    UNDERLYING,
)
from .polygon_client import PolygonClient
from .strategy import CondorLegs, Signal, find_first_signal, pick_legs

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TradeResult:
    day: date
    rsi_period: int
    rsi_upper: float
    rsi_lower: float
    strike_rule: str
    profit_target: float
    stop_loss: float
    entry_cutoff: str
    signal_time: datetime | None
    signal_direction: str | None
    signal_spot: float | None
    long_put: float | None
    short_put: float | None
    long_call: float | None
    short_call: float | None
    qty: int
    entry_debit: float | None     # paid at ask-side combo on entry
    exit_credit: float | None     # received at bid-side combo on exit
    exit_time: datetime | None
    minutes_held: float | None    # minutes from signal_time to exit_time
    exit_reason: str          # 'no_signal', 'no_data', 'profit', 'stop', 'time_stop'
    gross_pnl: float
    fees: float
    net_pnl: float
    balance_after: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trading_days(start: date, end: date) -> list[date]:
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range(
        pd.Timestamp(start), pd.Timestamp(end)
    )
    return [d.date() for d in sessions]


def _reindex_minute_bars(
    bars: pd.DataFrame, day: date
) -> pd.DataFrame:
    """Reindex to a 1-min grid 09:30–16:00 ET, forward-fill the close column."""
    if bars.empty:
        return bars
    et_index = bars.index.tz_convert("America/New_York")
    bars = bars.copy()
    bars.index = et_index
    start = pd.Timestamp(datetime.combine(day, datetime.min.time().replace(hour=9, minute=30))).tz_localize("America/New_York")
    end = pd.Timestamp(datetime.combine(day, datetime.min.time().replace(hour=16, minute=0))).tz_localize("America/New_York")
    grid = pd.date_range(start, end, freq="1min")
    return bars.reindex(grid).ffill()


def _combo_price(
    leg_bars: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    half_spread: float,
    action: str,
) -> float | None:
    """Long-IC combo price at `ts`, synthesized from 1-min closes + half spread.

    - action="enter": you BUY the combo. Pay ASK on long legs, sell at BID on
      shorts. Returns the debit per share you'd pay.
    - action="exit":  you SELL the combo. Sell at BID on long legs, buy at
      ASK on shorts. Returns the credit per share you'd receive.

    Equivalent to: combo_mid + 4*h on entry, combo_mid - 4*h on exit.
    Slippage is therefore baked into the entry/exit numbers — no separate
    deduction needed.
    """
    try:
        lp = leg_bars["long_put"].loc[ts, "close"]
        sp = leg_bars["short_put"].loc[ts, "close"]
        lc = leg_bars["long_call"].loc[ts, "close"]
        sc = leg_bars["short_call"].loc[ts, "close"]
    except KeyError:
        return None
    if pd.isna(lp) or pd.isna(sp) or pd.isna(lc) or pd.isna(sc):
        return None
    h = half_spread
    if action == "enter":
        # buy longs at ask (mid+h), sell shorts at bid (mid-h)
        return float((lp + h) + (lc + h) - (sp - h) - (sc - h))
    # exit: sell longs at bid (mid-h), buy shorts at ask (mid+h)
    return float((lp - h) + (lc - h) - (sp + h) - (sc + h))


def _round_trip_fees(qty: int, params: StrategyParams) -> float:
    """Commission cost for opening AND closing the 4-leg combo.

    Slippage / bid-ask cost is already baked into entry_debit (ask-side) and
    exit_credit (bid-side), so this is commissions only.
    """
    legs = 4
    sides = 2
    return qty * legs * sides * params.commission_per_contract


def _open_costs(qty: int, params: StrategyParams) -> float:
    """Commissions paid at entry (used for position sizing).

    entry_debit already includes the spread above mid on the buy side.
    """
    return qty * 4 * params.commission_per_contract


# ---------------------------------------------------------------------------
# Single-day simulation
# ---------------------------------------------------------------------------


def simulate_day(
    day: date,
    params: StrategyParams,
    balance: float,
    client: PolygonClient,
    spy_bars_cache: pd.DataFrame | None = None,
) -> TradeResult:
    """Simulate one trading day. Returns a TradeResult (which may be a no-trade)."""
    base = TradeResult(
        day=day,
        rsi_period=params.rsi_period,
        rsi_upper=params.rsi_upper,
        rsi_lower=params.rsi_lower,
        strike_rule=params.strike_rule.name,
        profit_target=params.profit_target_pct,
        stop_loss=params.stop_loss_pct,
        entry_cutoff=params.latest_entry.strftime("%H:%M"),
        signal_time=None,
        signal_direction=None,
        signal_spot=None,
        long_put=None,
        short_put=None,
        long_call=None,
        short_call=None,
        qty=0,
        entry_debit=None,
        exit_credit=None,
        exit_time=None,
        minutes_held=None,
        exit_reason="no_signal",
        gross_pnl=0.0,
        fees=0.0,
        net_pnl=0.0,
        balance_after=balance,
    )

    spy_bars = spy_bars_cache if spy_bars_cache is not None else client.get_minute_bars(
        UNDERLYING, day
    )
    if spy_bars.empty:
        base.exit_reason = "no_data"
        return base

    signal = find_first_signal(spy_bars, params)
    if signal is None:
        return base

    base.signal_time = signal.timestamp.to_pydatetime() if hasattr(signal.timestamp, "to_pydatetime") else signal.timestamp
    base.signal_direction = signal.direction
    base.signal_spot = signal.spot

    contracts = client.get_option_contracts(UNDERLYING, day)
    if not contracts:
        base.exit_reason = "no_data"
        return base

    legs = pick_legs(signal, day, params.strike_rule, contracts, client, UNDERLYING)
    if legs is None:
        base.exit_reason = "no_data"
        return base
    base.long_put = legs.long_put_strike
    base.short_put = legs.short_put_strike
    base.long_call = legs.long_call_strike
    base.short_call = legs.short_call_strike

    # Pull minute bars for the four legs and reindex onto a common grid.
    tickers = legs.tickers(UNDERLYING)
    leg_bars: dict[str, pd.DataFrame] = {}
    for name, ticker in tickers.items():
        df = client.get_option_minute_bars(ticker, day)
        leg_bars[name] = _reindex_minute_bars(df, day)
        if leg_bars[name].empty:
            base.exit_reason = "no_data"
            return base

    entry_ts = pd.Timestamp(signal.timestamp).tz_convert("America/New_York").floor("min")
    h = params.leg_half_spread
    entry_debit = _combo_price(leg_bars, entry_ts, h, "enter")
    if entry_debit is None or entry_debit <= 0.05:
        # Non-positive debit means our legs are mis-ordered or data is bad.
        base.exit_reason = "no_data"
        return base

    # Position sizing: full balance up to the cap. entry_debit is already at ask.
    capital = min(balance, params.max_capital_per_trade)
    per_contract_open = entry_debit * 100 + 4 * params.commission_per_contract
    qty = int(floor(capital / per_contract_open))
    if qty < 1:
        base.exit_reason = "no_data"
        return base
    base.qty = qty
    base.entry_debit = entry_debit

    # Walk minute-by-minute. P&L is measured on REALIZED exit credit (bid-side)
    # vs paid entry debit (ask-side), so the spread is already baked into the
    # target threshold — when we trigger, the bid is genuinely at +pt% of paid.
    time_stop_ts = pd.Timestamp(
        datetime.combine(day, params.time_stop)
    ).tz_localize("America/New_York")
    after_entry = leg_bars["long_put"].index[leg_bars["long_put"].index > entry_ts]
    forward = [ts for ts in after_entry if ts <= time_stop_ts]

    exit_ts: pd.Timestamp | None = None
    exit_credit: float | None = None
    exit_reason = "time_stop"

    for ts in forward:
        credit = _combo_price(leg_bars, ts, h, "exit")
        if credit is None:
            continue
        pnl_pct = (credit - entry_debit) / entry_debit
        if pnl_pct >= params.profit_target_pct:
            exit_ts, exit_credit, exit_reason = ts, credit, "profit"
            break
        if pnl_pct <= -params.stop_loss_pct:
            exit_ts, exit_credit, exit_reason = ts, credit, "stop"
            break

    if exit_ts is None:
        # Time stop: use the bar at or just before the time stop.
        candidate_idx = leg_bars["long_put"].index[leg_bars["long_put"].index <= time_stop_ts]
        if len(candidate_idx) == 0:
            base.exit_reason = "no_data"
            return base
        exit_ts = candidate_idx[-1]
        exit_credit = _combo_price(leg_bars, exit_ts, h, "exit")
        if exit_credit is None:
            base.exit_reason = "no_data"
            return base

    gross = (exit_credit - entry_debit) * 100 * qty
    fees = _round_trip_fees(qty, params)
    net = gross - fees

    base.exit_time = exit_ts.to_pydatetime()
    base.exit_credit = exit_credit
    base.exit_reason = exit_reason
    base.minutes_held = (exit_ts - entry_ts).total_seconds() / 60.0
    base.gross_pnl = gross
    base.fees = fees
    base.net_pnl = net
    base.balance_after = balance + net
    return base


# ---------------------------------------------------------------------------
# Multi-day run
# ---------------------------------------------------------------------------


def run_backtest(
    params: StrategyParams,
    start: date,
    end: date,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    """Run a single strategy configuration over [start, end]. Returns trade-log DataFrame."""
    client = client or PolygonClient()
    days = _trading_days(start, end)
    balance = params.starting_balance
    results: list[TradeResult] = []
    desc = (
        f"{params.strike_rule.name}"
        f"|rsi{params.rsi_period}_{int(params.rsi_upper)}-{int(params.rsi_lower)}"
        f"|pt{int(params.profit_target_pct*100)}"
        f"|sl{int(params.stop_loss_pct*100)}"
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
    rsi_periods: Iterable[int] = RSI_PERIODS,
    rsi_thresholds: Iterable[tuple[float, float]] = RSI_THRESHOLDS,
    strike_rules: Iterable[StrikeRule] = STRIKE_RULES,
    profit_targets: Iterable[float] = PROFIT_TARGETS,
    stop_losses: Iterable[float] = STOP_LOSSES,
    entry_cutoffs=ENTRY_CUTOFFS,
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    """Run every (rsi period × rsi threshold × strike × pt × sl × cutoff) combo.

    Returns a single DataFrame with a 'config' column distinguishing runs.
    """
    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [
        (rp, thr, sr, pt, sl, co)
        for rp in rsi_periods
        for thr in rsi_thresholds
        for sr in strike_rules
        for pt in profit_targets
        for sl in stop_losses
        for co in entry_cutoffs
    ]
    for rp, (up, lo), sr, pt, sl, co in combos:
        params = StrategyParams(
            rsi_period=rp,
            rsi_upper=up,
            rsi_lower=lo,
            earliest_entry=base.earliest_entry,
            latest_entry=co,
            time_stop=base.time_stop,
            hard_close=base.hard_close,
            profit_target_pct=pt,
            stop_loss_pct=sl,
            strike_rule=sr,
            commission_per_contract=base.commission_per_contract,
            leg_half_spread=base.leg_half_spread,
            starting_balance=base.starting_balance,
            max_capital_per_trade=base.max_capital_per_trade,
        )
        df = run_backtest(params, start, end, client=client)
        df["config"] = (
            f"{sr.name}|rsi{rp}_{int(up)}-{int(lo)}"
            f"|pt{int(pt*100)}|sl{int(sl*100)}|co{co.strftime('%H%M')}"
        )
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
