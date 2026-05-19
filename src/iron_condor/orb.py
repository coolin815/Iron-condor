"""SPY 2DTE momentum-trigger strategy — entry detection.

Each trading day we look for one entry signal:
  - Starting at params.entry_start (default 10:34 ET / 7:34 PT)
  - For each subsequent 1-min candle (up to params.max_attempts):
      - Read the previous minute's SPY close
      - If close >= 9:30 open + threshold  -> bias CALL
      - If close <= 9:30 open - threshold  -> bias PUT
      - Otherwise: not a trigger, but still counts as an attempt
      - Compute RSI(14) at the previous minute; must be in [rsi_min, rsi_max]
  - First minute that passes the trigger AND the RSI gate fires the signal.
  - Entry is at the OPEN of that candle.
  - Strike: 1-step ITM (CALL = first strike strictly < spot; PUT = first strike strictly > spot)
  - Expiry: today + params.dte business days (skipping weekends; no holiday handling).
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
    timestamp: pd.Timestamp           # entry minute (we buy at this candle's open)
    contract: str                     # Polygon option ticker
    strike: float
    right: Literal["C", "P"]
    expiry: date
    spot_at_signal: float             # SPY close from previous minute
    rsi_at_signal: float
    market_open: float                # SPY 9:30 open used as the anchor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_et(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    out = bars.copy()
    out.index = bars.index.tz_convert("America/New_York")
    return out


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI on a close-price series."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # Wilder smoothing — EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _two_dte_expiry(today: date, dte: int = 2) -> date:
    """Return the date `dte` business days after `today` (weekends skipped)."""
    return (pd.Timestamp(today) + pd.tseries.offsets.BusinessDay(dte)).date()


def _one_step_itm_strike(spot: float, right: str, step: float = 1.0) -> float:
    """First strike strictly ITM relative to spot.
    Call: strike < spot. Put: strike > spot. Strikes are multiples of `step`."""
    if right == "C":
        below = math.floor(spot / step) * step
        if below < spot:
            return float(below)
        return float(below - step)
    else:  # "P"
        above = math.ceil(spot / step) * step
        if above > spot:
            return float(above)
        return float(above + step)


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
    """Scan today's SPY 1-min bars for a momentum-trigger entry.

    `contracts` is the list of option contracts expiring on the 2DTE
    expiration. We use it to confirm the chosen strike actually trades.
    """
    if today_1min_spy.empty:
        return None
    et = _to_et(today_1min_spy)
    day = et.index[0].date()

    # 9:30 ET open
    open_bar = et[et.index.time == time(9, 30)]
    if open_bar.empty:
        return None
    market_open = float(open_bar.iloc[0]["open"])

    # Same-day RSI on close
    rsi = _wilder_rsi(et["close"], period=params.rsi_period)

    expiry = _two_dte_expiry(day, params.dte)

    # Build a set of strikes available for the chosen expiry so we can verify
    # the 1-step ITM strike actually exists. Most ATM strikes do, but better
    # safe than sorry — we'll fall back to the next strike if missing.
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

    minute_index = pd.Timestamp(
        datetime.combine(day, params.entry_start)
    ).tz_localize("America/New_York")

    for attempt in range(params.max_attempts):
        current = minute_index + pd.Timedelta(minutes=attempt)
        prev = current - pd.Timedelta(minutes=1)

        # Previous candle's close
        try:
            prev_close = float(et.loc[prev, "close"])
        except KeyError:
            log.debug("%s attempt %d: prev bar %s missing", day, attempt + 1, prev.time())
            continue
        if pd.isna(prev_close):
            continue

        diff = prev_close - market_open
        if diff >= params.price_move_threshold:
            right = "C"
        elif diff <= -params.price_move_threshold:
            right = "P"
        else:
            log.debug(
                "%s attempt %d (%s): diff=%+.3f below threshold $%.2f",
                day, attempt + 1, current.time(), diff, params.price_move_threshold,
            )
            continue

        # RSI gate (evaluated at the previous candle's timestamp)
        try:
            rsi_val = float(rsi.loc[prev])
        except KeyError:
            continue
        if pd.isna(rsi_val):
            continue
        if rsi_val < params.rsi_min or rsi_val > params.rsi_max:
            log.debug(
                "%s attempt %d (%s): RSI=%.1f outside [%.0f, %.0f]",
                day, attempt + 1, current.time(),
                rsi_val, params.rsi_min, params.rsi_max,
            )
            continue

        # Both gates passed. Need entry bar to exist (we'll fill at its open).
        try:
            entry_bar = et.loc[current]
        except KeyError:
            log.debug("%s attempt %d: entry bar %s missing", day, attempt + 1, current.time())
            continue
        if pd.isna(entry_bar.get("open")):
            continue

        # 1-step ITM strike, with fallback if the exact one isn't listed.
        strike = _one_step_itm_strike(prev_close, right, step=params.strike_step)
        if available_strikes.get(right):
            tries = 0
            while strike not in available_strikes[right] and tries < 4:
                # Push one further ITM
                strike = strike - params.strike_step if right == "C" else strike + params.strike_step
                tries += 1
            if strike not in available_strikes[right]:
                log.debug("%s attempt %d: no ITM strike near %.2f for %s",
                          day, attempt + 1, prev_close, right)
                continue

        ticker = build_option_ticker(underlying, expiry, right, strike)
        log.debug(
            "%s attempt %d ENTRY at %s: prev_close=%.3f open=%.3f diff=%+.3f "
            "right=%s strike=%.0f rsi=%.1f -> %s",
            day, attempt + 1, current.time(), prev_close, market_open, diff,
            right, strike, rsi_val, ticker,
        )
        return Signal(
            timestamp=current,
            contract=ticker,
            strike=strike,
            right=right,
            expiry=expiry,
            spot_at_signal=prev_close,
            rsi_at_signal=rsi_val,
            market_open=market_open,
        )

    log.debug("%s: no signal after %d attempts", day, params.max_attempts)
    return None
