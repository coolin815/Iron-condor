"""SPY consecutive-20min-candle entry trigger.

Each trading day:
  - Resample SPY 1-min bars into 20-min OHLC candles starting at 9:30 ET.
  - Walk candles in order; track each candle's direction (green / red / doji).
  - On the FIRST occurrence of two consecutive same-direction candles, fire:
      - 2 greens -> ATM call
      - 2 reds   -> ATM put
  - Entry fills at the OPEN of the next 1-min bar after the 2nd 20-min candle
    closes (e.g. 2nd candle [10:10, 10:30) -> entry minute = 10:30).
  - If the entry minute is past params.latest_entry, skip the day.

Position:
  - ATM (nearest $1 strike, round half up).
  - DTE: today (0DTE) or today + 2 business days (2DTE), per params.dte.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

import pandas as pd

from .config import StrategyParams
from .polygon_client import PolygonClient, build_option_ticker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp           # entry minute (1-min bar open = fill)
    contract: str
    strike: float
    right: Literal["C", "P"]
    expiry: date
    spot_at_entry: float
    trigger_open: float               # 2nd candle open
    trigger_close: float              # 2nd candle close
    direction: Literal["green", "red"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_et(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    out = bars.copy()
    out.index = bars.index.tz_convert("America/New_York")
    return out


def _aggregate_candles(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample ET-indexed 1-min bars into N-min OHLC anchored at 9:30 ET.
    Pre- and post-market bars are dropped so the buckets align to the
    regular session: [9:30, 9:50), [9:50, 10:10), ..."""
    if bars.empty:
        return bars
    day = bars.index[0].date()
    rth_start = pd.Timestamp(
        datetime.combine(day, time(9, 30))
    ).tz_localize("America/New_York")
    rth_end = pd.Timestamp(
        datetime.combine(day, time(16, 0))
    ).tz_localize("America/New_York")
    rth = bars[(bars.index >= rth_start) & (bars.index < rth_end)]
    if rth.empty:
        return rth
    agg = rth.resample(
        f"{minutes}min", origin=rth_start, label="left", closed="left",
    ).agg({"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"})
    return agg.dropna(subset=["open", "close"])


def _nearest_atm_strike(spot: float, step: float = 1.0) -> float:
    """Nearest strike multiple of step. Round half up."""
    return math.floor(spot / step + 0.5) * step


def _expiry_for_dte(today: date, dte: int) -> date:
    """0 DTE = today; >0 DTE = today + dte business days (weekends skipped)."""
    if dte <= 0:
        return today
    return (pd.Timestamp(today) + pd.tseries.offsets.BusinessDay(dte)).date()


def _candle_direction(open_: float, close: float) -> str | None:
    if close > open_:
        return "green"
    if close < open_:
        return "red"
    return None  # doji — breaks the streak


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI on a close-price series (EMA smoothing, alpha = 1/period)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def find_signal(
    today_1min_spy: pd.DataFrame,
    contracts: list[dict],
    client: PolygonClient,
    params: StrategyParams,
    underlying: str = "SPY",
) -> Signal | None:
    """Scan today's SPY tape for the first 2-consecutive-direction 20-min candle pattern."""
    if today_1min_spy.empty:
        return None
    et = _to_et(today_1min_spy)
    day = et.index[0].date()

    candles = _aggregate_candles(et, params.candle_minutes)
    if len(candles) < 2:
        return None

    latest_entry_ts = pd.Timestamp(
        datetime.combine(day, params.latest_entry)
    ).tz_localize("America/New_York")
    expiry = _expiry_for_dte(day, params.dte)

    # Set of strikes available for the chosen expiry — fall back to nearest
    # available if the literal round-half-up strike isn't listed.
    available_strikes: dict[str, set[float]] = {"C": set(), "P": set()}
    for c in contracts:
        try:
            strike = float(c["strike_price"])
        except (TypeError, ValueError, KeyError):
            continue
        ctype = c.get("contract_type", "").lower()
        if ctype == "call":
            available_strikes["C"].add(strike)
        elif ctype == "put":
            available_strikes["P"].add(strike)

    prev_dir: str | None = None
    for label_ts in candles.index:
        bar = candles.loc[label_ts]
        open_ = float(bar["open"])
        close_ = float(bar["close"])
        this_dir = _candle_direction(open_, close_)

        if this_dir is not None and this_dir == prev_dir:
            # Two in a row. Entry minute = 2nd candle's close-time.
            entry_ts = label_ts + pd.Timedelta(minutes=params.candle_minutes)
            if entry_ts > latest_entry_ts:
                log.debug(
                    "%s: 2nd %s candle at %s closes too late (latest %s)",
                    day, this_dir, label_ts.time(), params.latest_entry,
                )
                return None
            try:
                entry_bar = et.loc[entry_ts]
            except KeyError:
                log.debug("%s: entry 1-min bar at %s missing", day, entry_ts.time())
                # Need a bar to fill on — bail rather than skip ahead
                return None
            entry_open = entry_bar.get("open")
            if entry_open is None or pd.isna(entry_open):
                return None
            spot = float(entry_open)

            # RSI extreme filter — check RSI at the close of the 2nd 20-min
            # candle (= entry_ts). Use rsi_candle_minutes-bar same-day closes.
            if params.rsi_filter_enabled:
                rsi_bars = _aggregate_candles(et, params.rsi_candle_minutes)
                if not rsi_bars.empty:
                    rsi_series = _wilder_rsi(rsi_bars["close"], period=params.rsi_period)
                    rsi_label = entry_ts - pd.Timedelta(minutes=params.rsi_candle_minutes)
                    try:
                        rsi_val = float(rsi_series.loc[rsi_label])
                    except KeyError:
                        rsi_val = float("nan")
                    if not pd.isna(rsi_val):
                        if rsi_val > params.rsi_max or rsi_val < params.rsi_min:
                            log.debug(
                                "%s SKIP @ %s: RSI(%d,%dm)=%.1f outside [%.0f, %.0f]",
                                day, entry_ts.time(),
                                params.rsi_period, params.rsi_candle_minutes,
                                rsi_val, params.rsi_min, params.rsi_max,
                            )
                            return None

            right = "C" if this_dir == "green" else "P"
            strike = _nearest_atm_strike(spot, step=params.strike_step)

            # Verify strike trades; if not, walk outward
            if available_strikes.get(right):
                tries = 0
                while strike not in available_strikes[right] and tries < 4:
                    # Alternate +/- 1 strike step outward
                    delta = params.strike_step * (1 if tries % 2 == 0 else -1) * ((tries // 2) + 1)
                    strike = _nearest_atm_strike(spot, step=params.strike_step) + delta
                    tries += 1
                if strike not in available_strikes[right]:
                    log.debug("%s: no ATM strike near %.2f for %s on %s",
                              day, spot, right, expiry)
                    return None

            ticker = build_option_ticker(underlying, expiry, right, strike)
            log.debug(
                "%s ENTRY @ %s: 2x%s 20-min candles "
                "(prev close %.3f -> open %.3f -> close %.3f, "
                "spot %.3f, strike %.0f%s, dte=%d) -> %s",
                day, entry_ts.time(), this_dir,
                float(candles.iloc[candles.index.get_loc(label_ts) - 1]["close"])
                if candles.index.get_loc(label_ts) > 0 else float("nan"),
                open_, close_, spot, strike, right, params.dte, ticker,
            )
            return Signal(
                timestamp=entry_ts,
                contract=ticker,
                strike=strike,
                right=right,
                expiry=expiry,
                spot_at_entry=spot,
                trigger_open=open_,
                trigger_close=close_,
                direction=this_dir,  # type: ignore[arg-type]
            )

        prev_dir = this_dir

    return None
