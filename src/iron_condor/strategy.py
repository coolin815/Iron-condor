"""Entry signal detection and four-leg strike selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import pandas as pd

from .config import RISK_FREE_RATE, StrategyParams, StrikeRule
from .indicators import wilder_rsi
from .polygon_client import PolygonClient, build_option_ticker
from .pricing import bs_delta, implied_vol, time_to_expiry_years

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    timestamp: datetime           # tz-aware ET
    spot: float                   # SPY 1-min close at the signal bar
    direction: Literal["over", "under"]
    rsi: float


@dataclass(frozen=True)
class CondorLegs:
    """A long iron condor (debit, breakout payoff).

    Sign convention for `qty` is per-contract: +1 for long, -1 for short.
    The condor is: long inner put + short outer put + long inner call + short outer call.
    """

    long_put_strike: float
    short_put_strike: float       # further OTM (lower)
    long_call_strike: float
    short_call_strike: float      # further OTM (higher)
    expiry: date

    def all_strikes(self) -> tuple[float, float, float, float]:
        return (
            self.short_put_strike,
            self.long_put_strike,
            self.long_call_strike,
            self.short_call_strike,
        )

    def tickers(self, underlying: str) -> dict[str, str]:
        return {
            "long_put": build_option_ticker(
                underlying, self.expiry, "P", self.long_put_strike
            ),
            "short_put": build_option_ticker(
                underlying, self.expiry, "P", self.short_put_strike
            ),
            "long_call": build_option_ticker(
                underlying, self.expiry, "C", self.long_call_strike
            ),
            "short_call": build_option_ticker(
                underlying, self.expiry, "C", self.short_call_strike
            ),
        }


# ---------------------------------------------------------------------------
# Entry signal
# ---------------------------------------------------------------------------


def find_first_signal(
    spy_bars: pd.DataFrame, params: StrategyParams
) -> Signal | None:
    """Return the first qualifying RSI signal in the entry window, or None."""
    if spy_bars.empty:
        return None

    rsi = wilder_rsi(spy_bars["close"], period=params.rsi_period)
    df = spy_bars.copy()
    df["rsi"] = rsi

    # Restrict to the entry window.
    et_index = df.index.tz_convert("America/New_York")
    mask = (et_index.time >= params.earliest_entry) & (
        et_index.time <= params.latest_entry
    )
    window = df[mask]

    for ts, row in window.iterrows():
        r = row["rsi"]
        if pd.isna(r):
            continue
        if r > params.rsi_upper:
            return Signal(
                timestamp=ts, spot=float(row["close"]), direction="over", rsi=float(r)
            )
        if r < params.rsi_lower:
            return Signal(
                timestamp=ts, spot=float(row["close"]), direction="under", rsi=float(r)
            )
    return None


# ---------------------------------------------------------------------------
# Strike selection
# ---------------------------------------------------------------------------


def _round_strike(x: float, step: float = 1.0) -> float:
    return round(x / step) * step


def _available_strikes(contracts: list[dict]) -> tuple[set[float], set[float]]:
    """Split contract listing into available call and put strikes."""
    calls: set[float] = set()
    puts: set[float] = set()
    for c in contracts:
        try:
            strike = float(c.get("strike_price"))
        except (TypeError, ValueError):
            continue
        ctype = c.get("contract_type", "").lower()
        if ctype == "call":
            calls.add(strike)
        elif ctype == "put":
            puts.add(strike)
    return calls, puts


def _nearest_strike(target: float, available: set[float]) -> float:
    return min(available, key=lambda s: abs(s - target))


def _pick_fixed_legs(
    signal: Signal,
    expiry: date,
    rule: StrikeRule,
    call_strikes: set[float],
    put_strikes: set[float],
) -> CondorLegs:
    spot = signal.spot
    assert rule.long_inner_offset is not None and rule.wing_width is not None
    long_put_target = spot - rule.long_inner_offset
    short_put_target = long_put_target - rule.wing_width
    long_call_target = spot + rule.long_inner_offset
    short_call_target = long_call_target + rule.wing_width

    long_put = _nearest_strike(long_put_target, put_strikes)
    short_put = _nearest_strike(short_put_target, put_strikes)
    long_call = _nearest_strike(long_call_target, call_strikes)
    short_call = _nearest_strike(short_call_target, call_strikes)

    return CondorLegs(
        long_put_strike=long_put,
        short_put_strike=short_put,
        long_call_strike=long_call,
        short_call_strike=short_call,
        expiry=expiry,
    )


def _pick_delta_legs(
    signal: Signal,
    expiry: date,
    rule: StrikeRule,
    contracts: list[dict],
    client: PolygonClient,
    underlying: str,
) -> CondorLegs | None:
    """Pick strikes whose BS delta is closest to the targets.

    Reads the entry-minute mid price for each candidate strike, solves IV,
    computes delta. Considers strikes within ±$8 of spot to keep API calls bounded.
    """
    assert rule.inner_delta is not None and rule.outer_delta is not None
    call_strikes, put_strikes = _available_strikes(contracts)

    spot = signal.spot
    window = 8.0
    candidate_calls = sorted(s for s in call_strikes if abs(s - spot) <= window)
    candidate_puts = sorted(s for s in put_strikes if abs(s - spot) <= window)

    t = time_to_expiry_years(signal.timestamp, expiry)
    r = RISK_FREE_RATE
    entry_minute = signal.timestamp.floor("min")

    trade_day = signal.timestamp.date()

    def _delta_for(strike: float, right: Literal["C", "P"]) -> float | None:
        ticker = build_option_ticker(underlying, expiry, right, strike)
        bars = client.get_option_minute_bars(ticker, trade_day)
        if bars.empty:
            return None
        # Use the bar at the entry minute (or the most recent bar <= entry).
        bars_et = bars.index.tz_convert("America/New_York")
        idx = bars[bars_et <= entry_minute]
        if idx.empty:
            return None
        row = idx.iloc[-1]
        mid = float(row["close"])  # 1-min close as a proxy for mid
        if mid <= 0:
            return None
        iv = implied_vol(mid, spot, strike, t, r, right)
        if iv is None:
            return None
        return bs_delta(spot, strike, t, r, iv, right)

    # Calls: positive deltas. Inner = target inner_delta; outer = target outer_delta (smaller).
    call_deltas = {s: _delta_for(s, "C") for s in candidate_calls}
    call_deltas = {s: d for s, d in call_deltas.items() if d is not None}
    if not call_deltas:
        log.warning("No call delta data for %s", signal.timestamp.date())
        return None

    # Puts: negative deltas; compare absolute values to targets.
    put_deltas = {s: _delta_for(s, "P") for s in candidate_puts}
    put_deltas = {s: d for s, d in put_deltas.items() if d is not None}
    if not put_deltas:
        log.warning("No put delta data for %s", signal.timestamp.date())
        return None

    long_call = min(call_deltas, key=lambda s: abs(call_deltas[s] - rule.inner_delta))
    short_call = min(call_deltas, key=lambda s: abs(call_deltas[s] - rule.outer_delta))
    long_put = min(put_deltas, key=lambda s: abs(-put_deltas[s] - rule.inner_delta))
    short_put = min(put_deltas, key=lambda s: abs(-put_deltas[s] - rule.outer_delta))

    # Enforce ordering: short legs are further OTM than long legs.
    if short_call <= long_call:
        short_call = max(candidate_calls)
    if short_put >= long_put:
        short_put = min(candidate_puts)

    return CondorLegs(
        long_put_strike=long_put,
        short_put_strike=short_put,
        long_call_strike=long_call,
        short_call_strike=short_call,
        expiry=expiry,
    )


def pick_legs(
    signal: Signal,
    expiry: date,
    rule: StrikeRule,
    contracts: list[dict],
    client: PolygonClient,
    underlying: str = "SPY",
) -> CondorLegs | None:
    call_strikes, put_strikes = _available_strikes(contracts)
    if not call_strikes or not put_strikes:
        return None
    if rule.mode == "fixed":
        return _pick_fixed_legs(signal, expiry, rule, call_strikes, put_strikes)
    return _pick_delta_legs(signal, expiry, rule, contracts, client, underlying)
