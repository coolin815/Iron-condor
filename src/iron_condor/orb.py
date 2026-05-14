"""SPY 0DTE credit-spread strategy with ORB direction.

Single trade per day. The opening range (first 30 min) defines ORH/ORL.
The first 1-min close outside the OR between earliest_entry and
latest_entry triggers a trade in the direction of the break:

  Break above ORH (continuation up):  sell a BULL PUT spread
      short put = nearest strike >= spot - short_strike_offset
      long put  = short put strike - spread_width
    Profits if SPY stays above the short put through expiry.

  Break below ORL (continuation down): sell a BEAR CALL spread
      short call = nearest strike <= spot + short_strike_offset
      long call  = short call strike + spread_width
    Profits if SPY stays below the short call through expiry.

No Fridays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal

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
    timestamp: pd.Timestamp     # the 1-min bar whose close broke OR
    direction: Literal["bull_put", "bear_call"]
    spot: float                 # close of the breaking bar
    orh: float
    orl: float


@dataclass(frozen=True)
class SpreadLegs:
    short_strike: float
    long_strike: float
    right: Literal["P", "C"]     # both legs are the same type
    expiry: date

    def tickers(self, underlying: str) -> dict[str, str]:
        return {
            "short": build_option_ticker(underlying, self.expiry, self.right, self.short_strike),
            "long":  build_option_ticker(underlying, self.expiry, self.right, self.long_strike),
        }


# ---------------------------------------------------------------------------
# Helpers
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


def _is_friday(day: date) -> bool:
    return day.weekday() == 4


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
# Signal detection
# ---------------------------------------------------------------------------


def find_signal(
    today_1min: pd.DataFrame,
    params: StrategyParams,
) -> Signal | None:
    """First 1-min close outside the OR inside the entry window. Direction
    interpretation depends on params.direction_mode (continuation vs reversion).
    None on no signal, no data, or Friday-skip."""
    if today_1min.empty:
        return None
    if params.skip_fridays:
        day = today_1min.index[0].tz_convert("America/New_York").date()
        if _is_friday(day):
            return None

    levels = opening_range(today_1min, params.or_window_min)
    if levels.orh is None or levels.orl is None:
        return None

    et = _to_et(_between_time(today_1min, time(9, 30), time(16, 0)))
    if et.empty:
        return None

    revert = params.direction_mode == "reversion"

    mask = (et.index.time >= params.earliest_entry) & (et.index.time <= params.latest_entry)
    window = et[mask]
    for ts, row in window.iterrows():
        close = float(row["close"])
        if close > levels.orh:
            # Break above ORH:
            #   continuation: bull put (bet stays high)
            #   reversion: bear call (bet fades back down)
            direction = "bear_call" if revert else "bull_put"
            return Signal(
                timestamp=ts, direction=direction,
                spot=close, orh=levels.orh, orl=levels.orl,
            )
        if close < levels.orl:
            # Break below ORL:
            #   continuation: bear call (bet stays low)
            #   reversion: bull put (bet bounces back up)
            direction = "bull_put" if revert else "bear_call"
            return Signal(
                timestamp=ts, direction=direction,
                spot=close, orh=levels.orh, orl=levels.orl,
            )
    return None


# ---------------------------------------------------------------------------
# Strike picker
# ---------------------------------------------------------------------------


def _available_strikes(contracts: list[dict], right: str) -> set[float]:
    want = "call" if right == "C" else "put"
    out = set()
    for c in contracts:
        if c.get("contract_type", "").lower() != want:
            continue
        try:
            out.add(float(c["strike_price"]))
        except (TypeError, ValueError):
            continue
    return out


def pick_spread_legs(
    signal: Signal,
    expiry: date,
    contracts: list[dict],
    params: StrategyParams,
) -> SpreadLegs | None:
    """Pick the two strikes for the credit spread, snapping to listed strikes."""
    if signal.direction == "bull_put":
        right = "P"
        target_short = signal.spot - params.short_strike_offset
        target_long = target_short - params.spread_width
    else:  # bear_call
        right = "C"
        target_short = signal.spot + params.short_strike_offset
        target_long = target_short + params.spread_width

    strikes = _available_strikes(contracts, right)
    if not strikes:
        return None
    short = min(strikes, key=lambda s: abs(s - target_short))
    long_ = min(strikes, key=lambda s: abs(s - target_long))

    # Sanity: long must be on the OTM side of short by approx spread_width
    if signal.direction == "bull_put" and long_ >= short:
        return None
    if signal.direction == "bear_call" and long_ <= short:
        return None

    return SpreadLegs(
        short_strike=short, long_strike=long_, right=right, expiry=expiry,
    )


def spread_width_dollars(legs: SpreadLegs) -> float:
    return abs(legs.short_strike - legs.long_strike)
