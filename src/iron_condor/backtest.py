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
    FILTER_MODES,
    PROFIT_TARGETS,
    ROLL_MODES,
    SHORT_OTM_PCTS,
    SPREAD_WIDTHS,
    STOP_LOSS_MULTS,
    StrategyParams,
    UNDERLYING,
)
from .orb import Signal, _nearest_strike, find_signal
from .polygon_client import PolygonClient, build_option_ticker

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
    gross_pnl: float                        # before commissions (chain total if rolled)
    fees: float                             # round-trip commissions × qty × (1 + roll_count)
    net_pnl: float
    balance_after: float
    roll_count: int = 0                     # number of successful same-day rolls in this chain


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
    """Return SPY's prior trading day regular-session close (16:00 ET), or None."""
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


def _spy_premarket_change(et_bars: pd.DataFrame, day: date) -> float | None:
    """Return SPY's % change between 9:00 ET (6:00 PT) and 9:30 ET (6:30 PT)
    on `day`, computed against the 1-min bars in `et_bars` (ET-indexed).
    Returns None if either bar is missing."""
    start_ts = pd.Timestamp(
        datetime.combine(day, time(9, 0))
    ).tz_localize("America/New_York")
    end_ts = pd.Timestamp(
        datetime.combine(day, time(9, 30))
    ).tz_localize("America/New_York")
    try:
        start_price = float(et_bars.loc[start_ts, "open"])
        end_price = float(et_bars.loc[end_ts, "open"])
    except KeyError:
        return None
    if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
        return None
    return (end_price - start_price) / start_price


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

    # Regime filters: overnight gap and/or premarket move
    et = today_bars.copy()
    et.index = today_bars.index.tz_convert("America/New_York")

    overnight_fires = False
    premarket_fires = False
    overnight_pct: float | None = None
    premarket_pct: float | None = None

    if params.overnight_filter_enabled:
        prior_close = _spy_prior_close(client, day)
        if prior_close is not None:
            today_open_bars = et[et.index.time == time(9, 30)]
            if not today_open_bars.empty:
                today_open = float(today_open_bars.iloc[0]["open"])
                overnight_pct = (today_open - prior_close) / prior_close
                if overnight_pct < params.overnight_min_pct:
                    overnight_fires = True

    if params.premarket_filter_enabled:
        pm = _spy_premarket_change(et, day)
        if pm is not None:
            premarket_pct = pm
            if pm < params.premarket_min_pct:
                premarket_fires = True

    if params.overnight_filter_enabled and params.premarket_filter_enabled:
        skip = (overnight_fires or premarket_fires
                if params.filter_combine == "any"
                else (overnight_fires and premarket_fires))
    else:
        skip = overnight_fires or premarket_fires

    if skip:
        if overnight_fires and not premarket_fires:
            reason = "overnight_filter"
        elif premarket_fires and not overnight_fires:
            reason = "premarket_filter"
        else:
            reason = "regime_filter"
        log.debug(
            "%s: skip via %s (overnight=%s, premarket=%s, combine=%s)",
            day, reason,
            f"{overnight_pct * 100:.2f}%" if overnight_pct is not None else "n/a",
            f"{premarket_pct * 100:.2f}%" if premarket_pct is not None else "n/a",
            params.filter_combine,
        )
        return _empty_result(day, params, balance, reason)

    contracts = client.get_option_contracts(UNDERLYING, day)  # 0DTE
    if not contracts:
        return _empty_result(day, params, balance, "no_data")

    # Cache the chain's put strikes (used for picking roll strikes intraday).
    put_strikes: set[float] = set()
    for c in contracts:
        if c.get("contract_type", "").lower() != "put":
            continue
        try:
            put_strikes.add(float(c["strike_price"]))
        except (TypeError, ValueError, KeyError):
            continue

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

    hard_close_ts = pd.Timestamp(
        datetime.combine(day, params.hard_close)
    ).tz_localize("America/New_York")

    # Position chain: each iteration walks one open spread to PT/SL/hard_close.
    # On a non-final stop with rolling enabled, close this leg and open a new
    # one at fresh strikes for the rest of today.
    chain_gross_pnl = 0.0
    chain_fees = 0.0
    roll_count = 0

    cur_short_bars = short_bars
    cur_long_bars = long_bars
    cur_entry_ts = entry_ts
    cur_entry_credit = entry_credit

    final_exit_ts: pd.Timestamp | None = None
    final_exit_cost: float | None = None
    final_exit_reason = "hard_close"

    while True:
        pt_target_dollars = params.profit_target_pct * cur_entry_credit * 100
        sl_threshold_dollars = -params.stop_loss_mult * cur_entry_credit * 100

        leg_exit_ts: pd.Timestamp | None = None
        leg_exit_cost: float | None = None
        leg_exit_reason = "hard_close"

        forward = cur_short_bars.index[
            (cur_short_bars.index > cur_entry_ts)
            & (cur_short_bars.index <= hard_close_ts)
        ]
        for ts in forward:
            short_mid = _leg_price(cur_short_bars, ts, "close")
            long_mid = _leg_price(cur_long_bars, ts, "close")
            if short_mid is None or long_mid is None:
                continue
            short_ask_close = short_mid + h
            long_bid_close = max(long_mid - h, 0.0)
            cost_to_close = short_ask_close - long_bid_close
            pnl_per_spread = (cur_entry_credit - cost_to_close) * 100
            if pnl_per_spread >= pt_target_dollars:
                leg_exit_ts, leg_exit_cost, leg_exit_reason = ts, cost_to_close, "profit"
                break
            if pnl_per_spread <= sl_threshold_dollars:
                leg_exit_ts, leg_exit_cost, leg_exit_reason = ts, cost_to_close, "stop"
                break

        if leg_exit_ts is None:
            # hard_close
            cand = cur_short_bars.index[cur_short_bars.index <= hard_close_ts]
            if len(cand) == 0:
                return _empty_result(day, params, balance, "no_data")
            leg_exit_ts = cand[-1]
            short_mid = _leg_price(cur_short_bars, leg_exit_ts, "close")
            long_mid = _leg_price(cur_long_bars, leg_exit_ts, "close")
            if short_mid is None or long_mid is None:
                return _empty_result(day, params, balance, "no_data")
            short_ask_close = short_mid + h
            long_bid_close = max(long_mid - h, 0.0)
            leg_exit_cost = short_ask_close - long_bid_close
            leg_exit_reason = "hard_close"

        leg_pnl_per_spread = (cur_entry_credit - leg_exit_cost) * 100
        chain_gross_pnl += leg_pnl_per_spread * qty
        chain_fees += commission_round_trip * qty
        final_exit_ts = leg_exit_ts
        final_exit_cost = leg_exit_cost
        final_exit_reason = leg_exit_reason

        # Roll decision: only on stop, only if room left in chain & in the day.
        if not params.rolling_enabled or leg_exit_reason != "stop":
            break
        if roll_count >= params.max_rolls:
            break
        if (hard_close_ts - leg_exit_ts).total_seconds() < 30 * 60:
            break

        # Pick new strikes from current spot.
        spot_now = _leg_price(et, leg_exit_ts, "close")
        if spot_now is None:
            break
        new_short_strike = _nearest_strike(
            spot_now * (1 - params.short_otm_pct), params.strike_step, put_strikes
        )
        if new_short_strike is None:
            break
        new_long_strike = _nearest_strike(
            new_short_strike - params.spread_width, params.strike_step, put_strikes
        )
        if new_long_strike is None or new_long_strike >= new_short_strike:
            break

        new_short_ticker = build_option_ticker(UNDERLYING, day, "P", new_short_strike)
        new_long_ticker = build_option_ticker(UNDERLYING, day, "P", new_long_strike)
        new_short_bars = _reindex_option_bars(
            client.get_option_minute_bars(new_short_ticker, day), day
        )
        new_long_bars = _reindex_option_bars(
            client.get_option_minute_bars(new_long_ticker, day), day
        )
        if new_short_bars.empty or new_long_bars.empty:
            break

        next_mins = new_short_bars.index[new_short_bars.index > leg_exit_ts]
        if len(next_mins) == 0:
            break
        new_entry_ts = next_mins[0]
        new_short_mid = _leg_price(new_short_bars, new_entry_ts, "open")
        new_long_mid = _leg_price(new_long_bars, new_entry_ts, "open")
        if new_short_mid is None or new_long_mid is None:
            break
        new_short_bid = max(new_short_mid - h, 0.0)
        new_long_ask = new_long_mid + h
        new_entry_credit = new_short_bid - new_long_ask
        if new_entry_credit <= 0:
            break

        roll_count += 1
        cur_short_bars = new_short_bars
        cur_long_bars = new_long_bars
        cur_entry_ts = new_entry_ts
        cur_entry_credit = new_entry_credit
        log.debug(
            "%s ROLL #%d at %s: spot=%.2f new_short=%.0fP new_long=%.0fP credit=%.3f",
            day, roll_count, leg_exit_ts, spot_now,
            new_short_strike, new_long_strike, new_entry_credit,
        )

    gross = chain_gross_pnl
    fees = chain_fees
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
        exit_cost_per_share=final_exit_cost,
        exit_time=final_exit_ts.to_pydatetime() if final_exit_ts is not None else None,
        minutes_held=(final_exit_ts - entry_ts).total_seconds() / 60.0
            if final_exit_ts is not None else None,
        exit_reason=final_exit_reason,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        balance_after=balance + net,
        roll_count=roll_count,
    )


def _filter_mode_label(params: StrategyParams) -> str:
    o, p = params.overnight_filter_enabled, params.premarket_filter_enabled
    if not o and not p:
        return "none"
    if o and not p:
        return "overnight"
    if p and not o:
        return "premarket"
    return "either" if params.filter_combine == "any" else "both"


def _roll_mode_label(params: StrategyParams) -> str:
    if not params.rolling_enabled or params.max_rolls <= 0:
        return "noroll"
    return f"roll{params.max_rolls}"


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
        f"{_filter_mode_label(params)}|{_roll_mode_label(params)}"
        f"|otm{int(params.short_otm_pct * 1000)/10:.1f}|w{int(params.spread_width)}"
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
    filter_modes: Iterable[tuple[str, bool, bool, str]] = FILTER_MODES,
    roll_modes: Iterable[tuple[str, bool, int]] = ROLL_MODES,
    base_params: StrategyParams | None = None,
    client: PolygonClient | None = None,
    checkpoint_dir: "Path | None" = None,
) -> pd.DataFrame:
    """Run the full sweep. If `checkpoint_dir` is given, write sweep_trades.csv
    and sweep_summary.csv after EACH config completes, so an interrupted run
    leaves partial results on disk."""
    from .metrics import summarize_sweep

    client = client or PolygonClient()
    base = base_params or StrategyParams()
    all_rows: list[pd.DataFrame] = []

    combos = [
        (mode_label, on_off, roll_label, roll_enabled, roll_max, otm, w, pt, sl)
        for mode_label, *on_off in filter_modes
        for roll_label, roll_enabled, roll_max in roll_modes
        for otm in short_otm_pcts
        for w in spread_widths
        for pt in profit_targets
        for sl in stop_loss_mults
    ]
    for idx, (
        mode_label, on_off, roll_label, roll_enabled, roll_max, otm, w, pt, sl
    ) in enumerate(combos, 1):
        overnight_on, premarket_on, combine = on_off
        params = StrategyParams(
            entry_time=base.entry_time,
            hard_close=base.hard_close,
            short_otm_pct=otm,
            spread_width=w,
            strike_step=base.strike_step,
            vix_filter_enabled=base.vix_filter_enabled,
            vix_max=base.vix_max,
            overnight_filter_enabled=overnight_on,
            overnight_min_pct=base.overnight_min_pct,
            premarket_filter_enabled=premarket_on,
            premarket_min_pct=base.premarket_min_pct,
            filter_combine=combine,  # type: ignore[arg-type]
            profit_target_pct=pt,
            stop_loss_mult=sl,
            rolling_enabled=roll_enabled,
            max_rolls=roll_max,
            commission_per_contract=base.commission_per_contract,
            leg_half_spread=base.leg_half_spread,
            starting_balance=base.starting_balance,
            max_capital_per_trade=base.max_capital_per_trade,
        )
        df = run_backtest(params, start, end, client=client)
        df["config"] = (
            f"{mode_label}|{roll_label}|otm{int(otm * 1000)/10:.1f}|w{int(w)}"
            f"|pt{int(pt * 100)}|sl{sl:g}x"
        )
        all_rows.append(df)

        if checkpoint_dir is not None:
            partial = pd.concat(all_rows, ignore_index=True)
            partial.to_csv(checkpoint_dir / "sweep_trades.csv", index=False)
            summarize_sweep(partial, base.starting_balance).to_csv(
                checkpoint_dir / "sweep_summary.csv", index=False
            )
            log.info(
                "Checkpoint after config %d/%d (%s)",
                idx, len(combos), df["config"].iloc[0],
            )
    return pd.concat(all_rows, ignore_index=True)
