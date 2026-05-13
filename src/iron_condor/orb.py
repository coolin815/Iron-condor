"""SPY 0DTE breakout + reversal strategy.

Single trade per day. Two scans run in parallel; whichever fires first wins.
No trades on Fridays.

Indicators:
- VWAP: cumulative volume-weighted typical price from 9:30 ET (resets daily)
- EMA9 / EMA21: cross-day exponential moving averages on 1-min closes
- 5-min cross-day RSI(14): RSI on 5-min closes, continuous across days
- 5-min intraday RSI(14): RSI on 5-min closes, resets at 9:30 each day
  (needs ~70 min of warmup before it has a value)

Breakout signal:
- 2 consecutive 1-min closes outside the opening range (above ORH = call,
  below ORL = put)
- Filter: price aligned with VWAP, EMA9, EMA21 (above for call, below for put)
- Filter: cross-day RSI > 50 (call) or < 50 (put)
- Skip if cross-day RSI was > 70 or < 30 at the previous 5-min bar

Reversal signal:
- A 1-min candle closes outside the OR, the NEXT 1-min candle closes back inside
- Direction determined by current VWAP/EMA9 alignment (above = call, below = put)
- Filter: VWAP + EMA9 aligned (no EMA21)
- Filter: BOTH cross-day RSI and intraday RSI > 50 (call) or < 50 (put)
- Skip call reversals if cross-day RSI in [60, 65] (puts have no skip zone)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from .config import StrategyParams
from .polygon_client import build_option_ticker


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Levels:
    orh: float | None = None
    orl: float | None = None


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    signal_type: Literal["breakout", "reversal"]
    direction: Literal["call", "put"]
    spot: float
    orh: float
    orl: float
    vwap: float
    ema9: float
    ema21: float
    rsi_cross_day: float
    rsi_intraday: float | None


# ---------------------------------------------------------------------------
# Time-window helpers
# ---------------------------------------------------------------------------


def _between_time(bars: pd.DataFrame, start: time, end: time) -> pd.DataFrame:
    if bars.empty:
        return bars
    idx = bars.index.tz_convert("America/New_York")
    mask = (idx.time >= start) & (idx.time < end)
    return bars[mask]


def _to_et(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    out = bars.copy()
    out.index = bars.index.tz_convert("America/New_York")
    return out


# ---------------------------------------------------------------------------
# Opening range
# ---------------------------------------------------------------------------


def opening_range(today_bars: pd.DataFrame, or_window_min: int) -> Levels:
    """High and low of TOUCHED prices in the first `or_window_min` minutes."""
    if today_bars.empty:
        return Levels()
    end_dt = datetime.combine(date.today(), time(9, 30)) + timedelta(minutes=or_window_min)
    or_bars = _between_time(today_bars, time(9, 30), end_dt.time())
    if or_bars.empty:
        return Levels()
    return Levels(
        orh=float(or_bars["high"].max()),
        orl=float(or_bars["low"].min()),
    )


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def session_vwap(today_bars: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP across the regular session, indexed by minute."""
    reg = _to_et(_between_time(today_bars, time(9, 30), time(16, 0)))
    if reg.empty:
        return pd.Series(dtype=float)
    typ = (reg["high"] + reg["low"] + reg["close"]) / 3.0
    vol = reg["volume"].astype(float).fillna(0.0)
    cum_pv = (typ * vol).cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).ffill()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA with adjust=False so it starts from the first value."""
    return series.ewm(span=period, adjust=False).mean()


def _wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI on a closes series; uses simple-average seed then EWM alpha=1/period."""
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def aggregate_to_5min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV to 5-min, regular session only, ET-indexed."""
    reg = _to_et(_between_time(bars_1min, time(9, 30), time(16, 0)))
    if reg.empty:
        return reg
    agg = reg.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    return agg


def cross_day_5min_rsi(
    today_1min: pd.DataFrame, yesterday_1min: pd.DataFrame, period: int = 14
) -> pd.Series:
    """RSI(14) on 5-min closes spanning yesterday + today. Indexed by 5-min ET bar start."""
    today_5 = aggregate_to_5min(today_1min)
    yest_5 = aggregate_to_5min(yesterday_1min)
    if today_5.empty:
        return pd.Series(dtype=float)
    if yest_5.empty:
        combined = today_5
    else:
        combined = pd.concat([yest_5, today_5])
    return _wilder_rsi(combined["close"], period=period)


def intraday_5min_rsi(today_1min: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI(14) on today's 5-min closes only (resets daily)."""
    today_5 = aggregate_to_5min(today_1min)
    if today_5.empty:
        return pd.Series(dtype=float)
    return _wilder_rsi(today_5["close"], period=period)


def _value_at(series: pd.Series, ts: pd.Timestamp) -> float | None:
    """Most recent value in `series` at-or-before `ts`. None if no value."""
    if series.empty:
        return None
    sub = series[series.index <= ts]
    if sub.empty or pd.isna(sub.iloc[-1]):
        return None
    return float(sub.iloc[-1])


def _previous_5min_rsi_value(rsi_5min: pd.Series, ts: pd.Timestamp) -> float | None:
    """The 5-min RSI value at the BAR BEFORE the bar containing `ts`."""
    if rsi_5min.empty:
        return None
    sub = rsi_5min[rsi_5min.index < ts.floor("5min")]
    if sub.empty or pd.isna(sub.iloc[-1]):
        return None
    return float(sub.iloc[-1])


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def _is_friday(day: date) -> bool:
    return day.weekday() == 4


def find_signal(
    today_1min: pd.DataFrame,
    yesterday_1min: pd.DataFrame,
    params: StrategyParams,
) -> Signal | None:
    """Find the first breakout or reversal signal in the entry window.

    Two-stage logic. The candle PATTERN opens a 'scanning state' that
    persists across subsequent bars; on each bar in the scanning state, the
    bot tries to enter when the INDICATORS align.

    Breakout scanning state: count of consecutive closes outside OR (above
    or below) is >= 2. A close back inside OR resets the count to 0 (which
    exits the scanning state).

    Reversal scanning state: triggered when a bar closes inside OR after
    the previous bar closed outside. Persists across subsequent inside
    closes. Cancelled when price closes back outside OR (a new breakout
    attempt invalidates the prior reversal pattern).
    """
    if today_1min.empty:
        return None
    if params.skip_fridays:
        day = today_1min.index[0].tz_convert("America/New_York").date()
        if _is_friday(day):
            return None

    levels = opening_range(today_1min, params.or_window_min)
    if levels.orh is None or levels.orl is None:
        return None

    today_et = _to_et(today_1min)
    reg = _between_time(today_et, time(9, 30), time(16, 0))
    if reg.empty:
        return None

    vwap = session_vwap(today_1min)
    ema9 = ema(reg["close"], 9)
    ema21 = ema(reg["close"], 21)
    rsi_xd = cross_day_5min_rsi(today_1min, yesterday_1min)
    rsi_intra = intraday_5min_rsi(today_1min)

    bars = reg[(reg.index.time >= params.earliest_entry) & (reg.index.time <= params.latest_entry)]
    if bars.empty:
        return None

    try_breakout = params.signal_mode in ("both", "breakout")
    try_reversal = params.signal_mode in ("both", "reversal")

    # Scanning state
    consec_above_orh = 0
    consec_below_orl = 0
    reversal_call_scan = False     # came back inside from below (dip and recover)
    reversal_put_scan = False      # came back inside from above (spike and fade)

    for ts, row in bars.iterrows():
        cur_close = float(row["close"])

        prev_above_count = consec_above_orh
        prev_below_count = consec_below_orl

        if cur_close > levels.orh:
            consec_above_orh += 1
            consec_below_orl = 0
            # Closing outside cancels any reversal scan in progress
            reversal_call_scan = False
            reversal_put_scan = False
        elif cur_close < levels.orl:
            consec_below_orl += 1
            consec_above_orh = 0
            reversal_call_scan = False
            reversal_put_scan = False
        else:
            # Closed back inside OR — count resets
            consec_above_orh = 0
            consec_below_orl = 0
            # New reversal scans are triggered when prior bar was outside
            if prev_below_count >= 1:
                reversal_call_scan = True
                reversal_put_scan = False
            elif prev_above_count >= 1:
                reversal_put_scan = True
                reversal_call_scan = False
            # else: simply still inside — keep any prior reversal scan alive

        # First-to-fire: check breakout(s) first, then reversal(s)
        if try_breakout:
            if consec_above_orh >= 2:
                sig = _check_breakout_indicators(
                    "call", cur_close, ts, levels,
                    vwap, ema9, ema21, rsi_xd, params,
                )
                if sig is not None:
                    return sig
            elif consec_below_orl >= 2:
                sig = _check_breakout_indicators(
                    "put", cur_close, ts, levels,
                    vwap, ema9, ema21, rsi_xd, params,
                )
                if sig is not None:
                    return sig

        if try_reversal:
            if reversal_call_scan:
                sig = _check_reversal_indicators(
                    "call", cur_close, ts, levels,
                    vwap, ema9, rsi_xd, rsi_intra, params,
                )
                if sig is not None:
                    return sig
            elif reversal_put_scan:
                sig = _check_reversal_indicators(
                    "put", cur_close, ts, levels,
                    vwap, ema9, rsi_xd, rsi_intra, params,
                )
                if sig is not None:
                    return sig

    return None


def _check_breakout_indicators(
    direction: str, cur_close: float, ts, levels,
    vwap, ema9, ema21, rsi_xd, params,
) -> Signal | None:
    """Indicator alignment check for a breakout-direction scan."""
    vwap_now = _value_at(vwap, ts)
    ema9_now = _value_at(ema9, ts)
    ema21_now = _value_at(ema21, ts)
    rsi_now = _value_at(rsi_xd, ts)
    rsi_prev = _previous_5min_rsi_value(rsi_xd, ts)
    if None in (vwap_now, ema9_now, ema21_now, rsi_now):
        return None
    # Skip if the prior 5-min RSI bar was extreme
    if rsi_prev is not None and (
        rsi_prev > params.rsi_extreme_high or rsi_prev < params.rsi_extreme_low
    ):
        return None

    if direction == "call":
        if not (cur_close > vwap_now and cur_close > ema9_now and cur_close > ema21_now):
            return None
        if not (rsi_now > params.rsi_long_thresh):
            return None
    else:
        if not (cur_close < vwap_now and cur_close < ema9_now and cur_close < ema21_now):
            return None
        if not (rsi_now < params.rsi_short_thresh):
            return None

    return Signal(
        timestamp=ts, signal_type="breakout", direction=direction,
        spot=cur_close, orh=levels.orh, orl=levels.orl,
        vwap=vwap_now, ema9=ema9_now, ema21=ema21_now,
        rsi_cross_day=rsi_now, rsi_intraday=None,
    )


def _check_reversal_indicators(
    direction: str, cur_close: float, ts, levels,
    vwap, ema9, rsi_xd, rsi_intra, params,
) -> Signal | None:
    """Indicator alignment check for a reversal-direction scan."""
    vwap_now = _value_at(vwap, ts)
    ema9_now = _value_at(ema9, ts)
    rsi_xd_now = _value_at(rsi_xd, ts)
    rsi_intra_now = _value_at(rsi_intra, ts)
    if None in (vwap_now, ema9_now, rsi_xd_now, rsi_intra_now):
        return None

    if direction == "call":
        if not (cur_close > vwap_now and cur_close > ema9_now):
            return None
        if not (rsi_xd_now > params.rsi_long_thresh and rsi_intra_now > params.rsi_long_thresh):
            return None
        if params.reversal_call_skip_lo <= rsi_xd_now <= params.reversal_call_skip_hi:
            return None
    else:
        if not (cur_close < vwap_now and cur_close < ema9_now):
            return None
        if not (rsi_xd_now < params.rsi_short_thresh and rsi_intra_now < params.rsi_short_thresh):
            return None

    return Signal(
        timestamp=ts, signal_type="reversal", direction=direction,
        spot=cur_close, orh=levels.orh, orl=levels.orl,
        vwap=vwap_now, ema9=ema9_now, ema21=float("nan"),
        rsi_cross_day=rsi_xd_now, rsi_intraday=rsi_intra_now,
    )


# ---------------------------------------------------------------------------
# ATM contract picker
# ---------------------------------------------------------------------------


def pick_atm_contract(
    signal: Signal,
    expiry: date,
    contracts: list[dict],
    underlying: str = "SPY",
) -> tuple[str, float] | None:
    """Pick the call (long signal) or put (short signal) closest to spot."""
    want_type = "call" if signal.direction == "call" else "put"
    right = "C" if signal.direction == "call" else "P"
    candidates = [
        c for c in contracts
        if c.get("contract_type", "").lower() == want_type
        and c.get("strike_price") is not None
    ]
    if not candidates:
        return None
    best = min(
        candidates, key=lambda c: abs(float(c["strike_price"]) - signal.spot)
    )
    strike = float(best["strike_price"])
    ticker = build_option_ticker(underlying, expiry, right, strike)
    return ticker, strike
