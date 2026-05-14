"""Tests for the credit-spread strategy."""
from __future__ import annotations

from datetime import date, time

import pandas as pd

from iron_condor.config import StrategyParams
from iron_condor.orb import (
    find_signal,
    opening_range,
    pick_spread_legs,
    spread_width_dollars,
)


def _bars(start_iso: str, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Each row is (open, high, low, close). Index = consecutive 1-min bars."""
    idx = pd.date_range(start_iso, periods=len(rows), freq="1min").tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "open":  [r[0] for r in rows],
            "high":  [r[1] for r in rows],
            "low":   [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000] * len(rows),
        },
        index=idx,
    )


def test_opening_range_high_low() -> None:
    rows = [(500, 502, 499, 500) for _ in range(30)]
    rows[10] = (500, 505, 498, 500)  # 9:40 spike high
    rows[15] = (500, 502, 495, 500)  # 9:45 dip low
    df = _bars("2026-05-12 09:30", rows)
    levels = opening_range(df, or_window_min=30)
    assert levels.orh == 505
    assert levels.orl == 495


def test_signal_above_orh_is_bull_put() -> None:
    rows = [(500, 501, 499, 500) for _ in range(30)]   # OR: 9:30-9:59
    rows.append((500, 503, 500, 502.5))                # 10:00: first break above ORH=501
    df = _bars("2026-05-12 09:30", rows)
    params = StrategyParams(skip_fridays=False)
    sig = find_signal(df, params)
    assert sig is not None
    assert sig.direction == "bull_put"
    assert sig.spot == 502.5


def test_signal_below_orl_is_bear_call() -> None:
    rows = [(500, 501, 499, 500) for _ in range(30)]
    rows.append((500, 500, 497, 498))    # 10:00: first break below ORL=499
    df = _bars("2026-05-12 09:30", rows)
    params = StrategyParams(skip_fridays=False)
    sig = find_signal(df, params)
    assert sig is not None
    assert sig.direction == "bear_call"


def test_no_signal_on_fridays_when_skip_enabled() -> None:
    # 2026-05-15 is a Friday
    rows = [(500, 501, 499, 500) for _ in range(30)]
    rows.append((500, 503, 500, 502))
    df = _bars("2026-05-15 09:30", rows)
    params = StrategyParams(skip_fridays=True)
    assert find_signal(df, params) is None


def test_pick_bull_put_spread() -> None:
    from iron_condor.orb import Signal
    sig = Signal(
        timestamp=pd.Timestamp("2026-05-12 10:00", tz="America/New_York"),
        direction="bull_put", spot=500.5, orh=501, orl=499,
    )
    contracts = [
        {"contract_type": "put", "strike_price": k} for k in [497, 498, 499, 500, 501, 502]
    ]
    params = StrategyParams(short_strike_offset=1.0, spread_width=1.0)
    legs = pick_spread_legs(sig, date(2026, 5, 12), contracts, params)
    assert legs is not None
    # spot 500.5 - 1.0 = 499.5 → nearest is 499 or 500. min(abs) - either tied; min() picks first.
    # short should be at 499 or 500; long is short - 1.0 = lower
    assert legs.right == "P"
    assert legs.long_strike < legs.short_strike
    assert spread_width_dollars(legs) == 1.0


def test_pick_bear_call_spread() -> None:
    from iron_condor.orb import Signal
    sig = Signal(
        timestamp=pd.Timestamp("2026-05-12 10:00", tz="America/New_York"),
        direction="bear_call", spot=500.5, orh=501, orl=499,
    )
    contracts = [
        {"contract_type": "call", "strike_price": k} for k in [499, 500, 501, 502, 503, 504]
    ]
    params = StrategyParams(short_strike_offset=1.0, spread_width=1.0)
    legs = pick_spread_legs(sig, date(2026, 5, 12), contracts, params)
    assert legs is not None
    assert legs.right == "C"
    assert legs.long_strike > legs.short_strike
    assert spread_width_dollars(legs) == 1.0


def test_signal_outside_entry_window_returns_none() -> None:
    """Signal at 13:30 ET (past 13:00 cutoff) must not fire."""
    rows = [(500, 501, 499, 500) for _ in range(30)]    # OR 9:30-9:59
    rows += [(500, 501, 499, 500) for _ in range(210)]  # quiet through 13:29
    # Bar 240 = 9:30 + 240 min = 13:30 ET → past 13:00 cutoff
    rows.append((500, 505, 500, 504))                   # would break ORH
    df = _bars("2026-05-12 09:30", rows)
    params = StrategyParams(skip_fridays=False, latest_entry=time(13, 0))
    assert find_signal(df, params) is None
