"""SPY 0DTE short put credit spread — strike selection only.

No price-action trigger. At params.entry_time we pick the short and long
put strikes by percent-OTM-of-spot, verify both trade, and emit a Signal.
The backtest engine handles the actual fill / exit walk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

import pandas as pd

from .config import StrategyParams
from .polygon_client import PolygonClient, build_option_ticker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp           # entry minute (we fill at this bar's open)
    short_ticker: str
    long_ticker: str
    short_strike: float
    long_strike: float
    expiry: date
    spot_at_entry: float


def _to_et(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    out = bars.copy()
    out.index = bars.index.tz_convert("America/New_York")
    return out


def _nearest_strike(target: float, step: float, available: set[float]) -> float | None:
    """Return the closest available strike to `target`, snapping to step."""
    if not available:
        return None
    candidate = round(target / step) * step
    if candidate in available:
        return float(candidate)
    return float(min(available, key=lambda s: abs(s - target)))


def find_signal(
    today_1min_spy: pd.DataFrame,
    contracts: list[dict],
    client: PolygonClient,
    params: StrategyParams,
    underlying: str = "SPY",
) -> Signal | None:
    if today_1min_spy.empty:
        return None
    et = _to_et(today_1min_spy)
    day = et.index[0].date()

    entry_ts = pd.Timestamp(
        datetime.combine(day, params.entry_time)
    ).tz_localize("America/New_York")
    try:
        entry_bar = et.loc[entry_ts]
    except KeyError:
        log.debug("%s: no 1-min bar at %s", day, params.entry_time)
        return None
    spot = entry_bar.get("open")
    if spot is None or pd.isna(spot):
        return None
    spot = float(spot)

    # Build the set of available put strikes for today's expiry.
    put_strikes: set[float] = set()
    for c in contracts:
        if c.get("contract_type", "").lower() != "put":
            continue
        try:
            put_strikes.add(float(c["strike_price"]))
        except (TypeError, ValueError, KeyError):
            continue
    if not put_strikes:
        return None

    target_short = spot * (1 - params.short_otm_pct)
    short_strike = _nearest_strike(target_short, params.strike_step, put_strikes)
    if short_strike is None:
        return None
    target_long = short_strike - params.spread_width
    long_strike = _nearest_strike(target_long, params.strike_step, put_strikes)
    if long_strike is None:
        return None
    if long_strike >= short_strike:
        log.debug(
            "%s: degenerate spread (short=%.0f, long=%.0f), skipping",
            day, short_strike, long_strike,
        )
        return None

    expiry = day  # 0DTE
    short_ticker = build_option_ticker(underlying, expiry, "P", short_strike)
    long_ticker = build_option_ticker(underlying, expiry, "P", long_strike)

    log.debug(
        "%s ENTRY @ %s: spot=%.2f short=%.0fP long=%.0fP (width=$%.0f, otm=%.2f%%)",
        day, params.entry_time, spot, short_strike, long_strike,
        short_strike - long_strike, params.short_otm_pct * 100,
    )
    return Signal(
        timestamp=entry_ts,
        short_ticker=short_ticker,
        long_ticker=long_ticker,
        short_strike=short_strike,
        long_strike=long_strike,
        expiry=expiry,
        spot_at_entry=spot,
    )
