"""Opening Range Breakout signal + level computation + ATM contract picker.

Strategy (one paragraph):
- Build the opening range (ORH / ORL) from the first N minutes of the regular
  session (9:30 ET onward).
- Compute overnight reference levels: previous-day high/low (PDH/PDL),
  premarket high/low (PMH/PML; 4:00-9:29 ET today), overnight high/low
  (ONH/ONL; yesterday after-hours + today premarket).
- After the OR window closes, watch for the first bar where SPY high > ORH
  (long signal) or low < ORL (short signal) inside the entry window
  [earliest_entry, latest_entry].
- Optional confluence filter requires the break to also clear PDH/PMH/ONH
  (long side) or PDL/PML/ONL (short side).
- On signal: buy ATM 0DTE call (long) or put (short).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal

import pandas as pd

from .config import ConfluenceLevel, StrategyParams
from .polygon_client import build_option_ticker


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Levels:
    """All the price levels we use for ORB + confluence."""
    pdh: float | None = None
    pdl: float | None = None
    pmh: float | None = None
    pml: float | None = None
    onh: float | None = None
    onl: float | None = None
    orh: float | None = None
    orl: float | None = None


@dataclass(frozen=True)
class ORBSignal:
    timestamp: pd.Timestamp     # tz-aware ET, the bar that broke the level
    direction: Literal["long", "short"]
    spot: float                 # close of the breaking bar
    levels: Levels
    break_price: float          # the high (long) or low (short) that broke
    level_broken: str           # e.g. "ORH+PDH", "ORL"


# ---------------------------------------------------------------------------
# Level computation
# ---------------------------------------------------------------------------


def _between_time(bars: pd.DataFrame, start: time, end: time) -> pd.DataFrame:
    """Return rows where the ET clock-time is in [start, end). Empty-safe."""
    if bars.empty:
        return bars
    idx = bars.index.tz_convert("America/New_York")
    mask = (idx.time >= start) & (idx.time < end)
    return bars[mask]


def compute_levels(
    today_bars: pd.DataFrame,        # raw SPY bars for today (inc. pre/post)
    yesterday_bars: pd.DataFrame,    # raw SPY bars for previous trading day
    or_window_min: int,
) -> Levels:
    """Compute PDH/PDL, PMH/PML, ONH/ONL, ORH/ORL."""
    pdh = pdl = pmh = pml = onh = onl = orh = orl = None

    # --- yesterday's regular session for PDH/PDL ---
    if not yesterday_bars.empty:
        y_reg = _between_time(yesterday_bars, time(9, 30), time(16, 0))
        if not y_reg.empty:
            pdh = float(y_reg["high"].max())
            pdl = float(y_reg["low"].min())

    # --- today's premarket for PMH/PML ---
    if not today_bars.empty:
        t_pm = _between_time(today_bars, time(4, 0), time(9, 30))
        if not t_pm.empty:
            pmh = float(t_pm["high"].max())
            pml = float(t_pm["low"].min())

    # --- overnight: yesterday after-hours + today premarket ---
    on_pieces = []
    if not yesterday_bars.empty:
        y_ah = _between_time(yesterday_bars, time(16, 0), time(20, 0))
        if not y_ah.empty:
            on_pieces.append(y_ah)
    if not today_bars.empty:
        t_pm = _between_time(today_bars, time(4, 0), time(9, 30))
        if not t_pm.empty:
            on_pieces.append(t_pm)
    if on_pieces:
        combined = pd.concat(on_pieces)
        onh = float(combined["high"].max())
        onl = float(combined["low"].min())

    # --- ORH/ORL: first `or_window_min` minutes of today's regular session ---
    if not today_bars.empty:
        or_end_dt = datetime.combine(date.today(), time(9, 30)) + timedelta(minutes=or_window_min)
        or_end = or_end_dt.time()
        t_or = _between_time(today_bars, time(9, 30), or_end)
        if not t_or.empty:
            orh = float(t_or["high"].max())
            orl = float(t_or["low"].min())

    return Levels(pdh=pdh, pdl=pdl, pmh=pmh, pml=pml, onh=onh, onl=onl, orh=orh, orl=orl)


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def _confluence_label(direction: str, break_price: float, levels: Levels) -> str:
    """Build a string describing which confluence levels were cleared."""
    parts = []
    if direction == "long":
        parts.append("ORH")
        for tag, lv in (("PDH", levels.pdh), ("PMH", levels.pmh), ("ONH", levels.onh)):
            if lv is not None and break_price > lv:
                parts.append(tag)
    else:
        parts.append("ORL")
        for tag, lv in (("PDL", levels.pdl), ("PML", levels.pml), ("ONL", levels.onl)):
            if lv is not None and break_price < lv:
                parts.append(tag)
    return "+".join(parts)


def _passes_confluence(
    direction: str, break_price: float, levels: Levels, confluence: ConfluenceLevel
) -> bool:
    if confluence == "none":
        return True
    if direction == "long":
        cands = {
            "pdh_pdl": [levels.pdh],
            "pmh_pml": [levels.pmh],
            "onh_onl": [levels.onh],
            "any": [levels.pdh, levels.pmh, levels.onh],
        }[confluence]
        return any(lv is not None and break_price > lv for lv in cands)
    else:
        cands = {
            "pdh_pdl": [levels.pdl],
            "pmh_pml": [levels.pml],
            "onh_onl": [levels.onl],
            "any": [levels.pdl, levels.pml, levels.onl],
        }[confluence]
        return any(lv is not None and break_price < lv for lv in cands)


def find_orb_signal(
    today_bars: pd.DataFrame,
    levels: Levels,
    params: StrategyParams,
) -> ORBSignal | None:
    """First ORH/ORL break inside the entry window. None if no break."""
    if levels.orh is None or levels.orl is None or today_bars.empty:
        return None

    idx = today_bars.index.tz_convert("America/New_York")
    bars = today_bars.copy()
    bars.index = idx
    mask = (idx.time >= params.earliest_entry) & (idx.time <= params.latest_entry)
    window = bars[mask]

    for ts, row in window.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        if hi > levels.orh:
            if _passes_confluence("long", hi, levels, params.confluence):
                return ORBSignal(
                    timestamp=ts,
                    direction="long",
                    spot=cl,
                    levels=levels,
                    break_price=hi,
                    level_broken=_confluence_label("long", hi, levels),
                )
        if lo < levels.orl:
            if _passes_confluence("short", lo, levels, params.confluence):
                return ORBSignal(
                    timestamp=ts,
                    direction="short",
                    spot=cl,
                    levels=levels,
                    break_price=lo,
                    level_broken=_confluence_label("short", lo, levels),
                )
    return None


# ---------------------------------------------------------------------------
# Contract picker
# ---------------------------------------------------------------------------


def pick_atm_contract(
    signal: ORBSignal,
    expiry: date,
    contracts: list[dict],
    underlying: str = "SPY",
) -> tuple[str, float] | None:
    """Pick the call (long signal) or put (short signal) closest to spot.

    Returns (option_ticker, strike) or None if no matching contracts.
    """
    want_type = "call" if signal.direction == "long" else "put"
    right = "C" if signal.direction == "long" else "P"
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
