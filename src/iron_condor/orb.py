"""SPY 0DTE candle-pattern strategy.

Scans 10 candle patterns on 5-min bars. Whichever fires first inside the
entry window, with all four indicators confirming, wins. One trade/day.

Patterns (10 total, all 3 candles):
  Bullish (call):
    1. three_white_soldiers — 3 consecutive higher-closing bullish candles
    2. morning_star — big bear, small body, big bull closing past c1 midpoint
    3. bullish_engulfing — bear, bullish engulfing, bullish continuation
    4. hammer — bear, hammer (long lower wick), bullish continuation
    5. piercing — bear, bullish closing past 50% of c1 body, continuation
  Bearish (put):
    6. three_black_crows
    7. evening_star
    8. bearish_engulfing
    9. shooting_star
    10. dark_cloud_cover

Confirmation required on the detection bar's close (all four):
  Calls:  close > VWAP  AND  EMA9 > EMA21  AND  close > EMA9  AND  RSI > 50
  Puts:   close < VWAP  AND  EMA9 < EMA21  AND  close < EMA9  AND  RSI < 50

Indicators are all computed on the 5-min bar series, cross-day (EMA, RSI),
so opening bars already have a warmed-up value.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

import numpy as np
import pandas as pd

from .config import PATTERN_NAMES, StrategyParams
from .polygon_client import build_option_ticker


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp          # close of the 3rd (most recent) detection bar
    pattern: str
    direction: Literal["call", "put"]
    spot: float
    vwap: float
    ema9: float
    ema21: float
    rsi: float


# ---------------------------------------------------------------------------
# Time helpers
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
# Indicators
# ---------------------------------------------------------------------------


def session_vwap(bars_1min: pd.DataFrame) -> pd.Series:
    reg = _to_et(_between_time(bars_1min, time(9, 30), time(16, 0)))
    if reg.empty:
        return pd.Series(dtype=float)
    typ = (reg["high"] + reg["low"] + reg["close"]) / 3.0
    vol = reg["volume"].astype(float).fillna(0.0)
    cum_pv = (typ * vol).cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).ffill()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
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
    """1-min OHLCV -> 5-min OHLCV, regular session only, ET-indexed."""
    reg = _to_et(_between_time(bars_1min, time(9, 30), time(16, 0)))
    if reg.empty:
        return reg
    agg = reg.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    return agg


# ---------------------------------------------------------------------------
# Candle pattern detectors
# Each takes 3 consecutive 5-min bars (c1 oldest, c3 most recent) and returns
# True if the pattern is present. Body-size thresholds are relative to the
# candle's total range so they scale with intraday volatility.
# ---------------------------------------------------------------------------


def _body(c) -> float:
    return abs(float(c["close"]) - float(c["open"]))


def _is_bullish(c) -> bool:
    return float(c["close"]) > float(c["open"])


def _is_bearish(c) -> bool:
    return float(c["close"]) < float(c["open"])


def _upper_wick(c) -> float:
    return float(c["high"]) - max(float(c["close"]), float(c["open"]))


def _lower_wick(c) -> float:
    return min(float(c["close"]), float(c["open"])) - float(c["low"])


def _total_range(c) -> float:
    return float(c["high"]) - float(c["low"])


# 1. Three White Soldiers
def is_three_white_soldiers(c1, c2, c3) -> bool:
    return (
        _is_bullish(c1) and _is_bullish(c2) and _is_bullish(c3)
        and float(c2["close"]) > float(c1["close"])
        and float(c3["close"]) > float(c2["close"])
        # each candle has a meaningful body
        and _body(c1) > 0.5 * _total_range(c1)
        and _body(c2) > 0.5 * _total_range(c2)
        and _body(c3) > 0.5 * _total_range(c3)
    )


# 2. Three Black Crows
def is_three_black_crows(c1, c2, c3) -> bool:
    return (
        _is_bearish(c1) and _is_bearish(c2) and _is_bearish(c3)
        and float(c2["close"]) < float(c1["close"])
        and float(c3["close"]) < float(c2["close"])
        and _body(c1) > 0.5 * _total_range(c1)
        and _body(c2) > 0.5 * _total_range(c2)
        and _body(c3) > 0.5 * _total_range(c3)
    )


# 3. Morning Star (bull reversal)
def is_morning_star(c1, c2, c3) -> bool:
    c1_body = _body(c1)
    if c1_body == 0:
        return False
    midpoint = (float(c1["open"]) + float(c1["close"])) / 2.0
    return (
        _is_bearish(c1)
        and c1_body > 0.5 * _total_range(c1)
        and _body(c2) < 0.5 * c1_body       # small middle candle
        and _is_bullish(c3)
        and _body(c3) > 0.5 * _total_range(c3)
        and float(c3["close"]) > midpoint   # closes back past c1 midpoint
    )


# 4. Evening Star (bear reversal)
def is_evening_star(c1, c2, c3) -> bool:
    c1_body = _body(c1)
    if c1_body == 0:
        return False
    midpoint = (float(c1["open"]) + float(c1["close"])) / 2.0
    return (
        _is_bullish(c1)
        and c1_body > 0.5 * _total_range(c1)
        and _body(c2) < 0.5 * c1_body
        and _is_bearish(c3)
        and _body(c3) > 0.5 * _total_range(c3)
        and float(c3["close"]) < midpoint
    )


# 5. Bullish Engulfing + confirmation
def is_bullish_engulfing(c1, c2, c3) -> bool:
    return (
        _is_bearish(c1)
        and _is_bullish(c2)
        and float(c2["open"]) <= float(c1["close"])
        and float(c2["close"]) >= float(c1["open"])
        and _body(c2) > _body(c1)
        and _is_bullish(c3)
        and float(c3["close"]) > float(c2["close"])
    )


# 6. Bearish Engulfing + confirmation
def is_bearish_engulfing(c1, c2, c3) -> bool:
    return (
        _is_bullish(c1)
        and _is_bearish(c2)
        and float(c2["open"]) >= float(c1["close"])
        and float(c2["close"]) <= float(c1["open"])
        and _body(c2) > _body(c1)
        and _is_bearish(c3)
        and float(c3["close"]) < float(c2["close"])
    )


# 7. Hammer + confirmation (bull reversal)
def is_hammer(c1, c2, c3) -> bool:
    c2_body = _body(c2)
    c2_range = _total_range(c2)
    if c2_body == 0 or c2_range == 0:
        return False
    return (
        _is_bearish(c1)
        and _body(c1) > 0.5 * _total_range(c1)
        and _lower_wick(c2) >= 2.0 * c2_body
        and _upper_wick(c2) <= 0.15 * c2_range
        and c2_body <= 0.35 * c2_range
        and _is_bullish(c3)
        and float(c3["close"]) > float(c2["close"])
    )


# 8. Shooting Star + confirmation (bear reversal)
def is_shooting_star(c1, c2, c3) -> bool:
    c2_body = _body(c2)
    c2_range = _total_range(c2)
    if c2_body == 0 or c2_range == 0:
        return False
    return (
        _is_bullish(c1)
        and _body(c1) > 0.5 * _total_range(c1)
        and _upper_wick(c2) >= 2.0 * c2_body
        and _lower_wick(c2) <= 0.15 * c2_range
        and c2_body <= 0.35 * c2_range
        and _is_bearish(c3)
        and float(c3["close"]) < float(c2["close"])
    )


# 9. Piercing pattern + confirmation (bull reversal)
def is_piercing(c1, c2, c3) -> bool:
    c1_body = _body(c1)
    if c1_body == 0:
        return False
    midpoint = (float(c1["open"]) + float(c1["close"])) / 2.0
    return (
        _is_bearish(c1)
        and c1_body > 0.5 * _total_range(c1)
        and _is_bullish(c2)
        and float(c2["open"]) < float(c1["close"])     # opens below c1 close
        and float(c2["close"]) > midpoint               # closes past midpoint
        and float(c2["close"]) < float(c1["open"])     # but below c1 open (not engulfing)
        and _is_bullish(c3)
        and float(c3["close"]) > float(c2["close"])
    )


# 10. Dark Cloud Cover + confirmation (bear reversal)
def is_dark_cloud(c1, c2, c3) -> bool:
    c1_body = _body(c1)
    if c1_body == 0:
        return False
    midpoint = (float(c1["open"]) + float(c1["close"])) / 2.0
    return (
        _is_bullish(c1)
        and c1_body > 0.5 * _total_range(c1)
        and _is_bearish(c2)
        and float(c2["open"]) > float(c1["close"])
        and float(c2["close"]) < midpoint
        and float(c2["close"]) > float(c1["open"])
        and _is_bearish(c3)
        and float(c3["close"]) < float(c2["close"])
    )


PATTERN_DETECTORS = (
    ("three_white_soldiers", is_three_white_soldiers, "call"),
    ("three_black_crows",    is_three_black_crows,    "put"),
    ("morning_star",         is_morning_star,         "call"),
    ("evening_star",         is_evening_star,         "put"),
    ("bullish_engulfing",    is_bullish_engulfing,    "call"),
    ("bearish_engulfing",    is_bearish_engulfing,    "put"),
    ("hammer",               is_hammer,               "call"),
    ("shooting_star",        is_shooting_star,        "put"),
    ("piercing",             is_piercing,             "call"),
    ("dark_cloud",           is_dark_cloud,           "put"),
)


def detect_first_pattern(c1, c2, c3, enabled: tuple[str, ...] | None = None):
    """Run pattern detectors in priority order. If `enabled` is given, only
    those pattern names are considered. Returns (name, direction) or None."""
    for name, detector, direction in PATTERN_DETECTORS:
        if enabled is not None and name not in enabled:
            continue
        if detector(c1, c2, c3):
            return name, direction
    return None


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


def _confirm(direction: str, close: float, vwap: float, ema9: float, ema21: float, rsi: float, params) -> bool:
    if direction == "call":
        return (
            close > vwap
            and ema9 > ema21
            and close > ema9
            and rsi > params.rsi_long_thresh
        )
    return (
        close < vwap
        and ema9 < ema21
        and close < ema9
        and rsi < params.rsi_short_thresh
    )


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def _is_friday(day: date) -> bool:
    return day.weekday() == 4


def _vwap_at(vwap: pd.Series, ts: pd.Timestamp) -> float | None:
    """VWAP value at-or-before `ts`."""
    if vwap.empty:
        return None
    sub = vwap[vwap.index <= ts]
    if sub.empty or pd.isna(sub.iloc[-1]):
        return None
    return float(sub.iloc[-1])


def find_signal(
    today_1min: pd.DataFrame,
    yesterday_1min: pd.DataFrame,
    params: StrategyParams,
) -> Signal | None:
    """First pattern-with-confirmation match inside the entry window."""
    if today_1min.empty:
        return None
    if params.skip_fridays:
        day = today_1min.index[0].tz_convert("America/New_York").date()
        if _is_friday(day):
            return None

    # Aggregate to 5-min bars, cross-day for EMA + RSI warmup
    today_5 = aggregate_to_5min(today_1min)
    if today_5.empty:
        return None
    yest_5 = aggregate_to_5min(yesterday_1min) if not yesterday_1min.empty else pd.DataFrame()
    combined_close = pd.concat([yest_5["close"], today_5["close"]]) if not yest_5.empty else today_5["close"]
    ema9 = ema(combined_close, 9).loc[today_5.index]
    ema21 = ema(combined_close, 21).loc[today_5.index]
    rsi = _wilder_rsi(combined_close, period=14).loc[today_5.index]

    # VWAP from 1-min data (intraday cumulative)
    vwap = session_vwap(today_1min)

    # Walk 5-min bars and look for patterns
    bars = today_5.copy()
    timestamps = list(bars.index)
    if len(timestamps) < 3:
        return None

    bar_minutes = params.bar_timeframe_min

    for i in range(2, len(timestamps)):
        ts = timestamps[i]
        # The 5-min bar labeled `ts` covers [ts, ts + bar_minutes). The bar's
        # close + indicator values are not knowable in real life until that
        # END timestamp — so the signal fires at `ts + bar_minutes`. Entering
        # at the bar's start would be lookahead bias.
        signal_ts = ts + pd.Timedelta(minutes=bar_minutes)
        signal_time = signal_ts.time()
        if signal_time < params.earliest_entry:
            continue
        if signal_time > params.latest_entry:
            break

        c1 = bars.iloc[i - 2]
        c2 = bars.iloc[i - 1]
        c3 = bars.iloc[i]
        match = detect_first_pattern(c1, c2, c3, enabled=params.enabled_patterns)
        if match is None:
            continue
        pattern_name, direction = match

        # Confirmations on this bar's close. EMAs and RSI are indexed by the
        # 5-min bar's left label and the value AT that label includes the
        # full bar (so they're already the right value at signal_ts).
        # VWAP is indexed by 1-min timestamps; vwap.loc[T] = VWAP through end
        # of the 1-min bar at T (= VWAP at time T+1). At signal_ts we want
        # the VWAP through the last 1-min bar before signal_ts:00.
        close = float(c3["close"])
        vwap_now = _vwap_at(vwap, signal_ts - pd.Timedelta(minutes=1))
        ema9_now = float(ema9.loc[ts]) if ts in ema9.index and pd.notna(ema9.loc[ts]) else None
        ema21_now = float(ema21.loc[ts]) if ts in ema21.index and pd.notna(ema21.loc[ts]) else None
        rsi_now = float(rsi.loc[ts]) if ts in rsi.index and pd.notna(rsi.loc[ts]) else None

        if None in (vwap_now, ema9_now, ema21_now, rsi_now):
            continue
        if not _confirm(direction, close, vwap_now, ema9_now, ema21_now, rsi_now, params):
            continue

        return Signal(
            timestamp=signal_ts,        # bar END, not start
            pattern=pattern_name,
            direction=direction,
            spot=close,
            vwap=vwap_now,
            ema9=ema9_now,
            ema21=ema21_now,
            rsi=rsi_now,
        )

    return None


# ---------------------------------------------------------------------------
# ATM contract picker
# ---------------------------------------------------------------------------


def pick_atm_contract(
    signal: Signal,
    expiry: date,
    contracts: list[dict],
    underlying: str = "SPY",
) -> tuple[str, float] | None:
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
