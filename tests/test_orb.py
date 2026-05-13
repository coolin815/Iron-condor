"""Tests for the candle-pattern strategy."""
from __future__ import annotations

import pandas as pd
import pytest

from iron_condor.orb import (
    detect_first_pattern,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_dark_cloud,
    is_evening_star,
    is_hammer,
    is_morning_star,
    is_piercing,
    is_shooting_star,
    is_three_black_crows,
    is_three_white_soldiers,
)


def C(o, h, l, c):
    """Build a single-bar dict-like accessible via [\"open\"], etc."""
    return pd.Series({"open": float(o), "high": float(h), "low": float(l), "close": float(c)})


# ---------------------------------------------------------------------------
# 1-2. Three White Soldiers / Three Black Crows
# ---------------------------------------------------------------------------


def test_three_white_soldiers_fires_on_clean_ramp() -> None:
    c1 = C(100, 101, 99.8, 101)
    c2 = C(101, 102, 100.8, 102)
    c3 = C(102, 103, 101.8, 103)
    assert is_three_white_soldiers(c1, c2, c3)


def test_three_white_soldiers_rejects_a_red_candle() -> None:
    c1 = C(100, 101, 99.8, 101)
    c2 = C(101, 101.5, 100.5, 100.8)   # red
    c3 = C(101, 102, 100.8, 102)
    assert not is_three_white_soldiers(c1, c2, c3)


def test_three_black_crows_fires_on_clean_drop() -> None:
    c1 = C(103, 103.2, 102, 102)
    c2 = C(102, 102.2, 101, 101)
    c3 = C(101, 101.2, 100, 100)
    assert is_three_black_crows(c1, c2, c3)


# ---------------------------------------------------------------------------
# 3-4. Morning Star / Evening Star
# ---------------------------------------------------------------------------


def test_morning_star_fires() -> None:
    # Big bear, tiny middle, big bull closing past c1 midpoint
    c1 = C(105, 105.1, 99.9, 100)      # big bearish, midpoint 102.5
    c2 = C(100, 100.2, 99.5, 99.8)     # small body
    c3 = C(100, 105, 99.9, 104)        # big bullish, closes 104 > 102.5
    assert is_morning_star(c1, c2, c3)


def test_evening_star_fires() -> None:
    c1 = C(100, 105.1, 99.9, 105)      # big bullish, midpoint 102.5
    c2 = C(105, 105.5, 104.8, 105.2)   # small body
    c3 = C(105, 105.1, 100, 101)       # big bearish, closes 101 < 102.5
    assert is_evening_star(c1, c2, c3)


# ---------------------------------------------------------------------------
# 5-6. Engulfing + confirmation
# ---------------------------------------------------------------------------


def test_bullish_engulfing_with_confirmation() -> None:
    c1 = C(101, 101.2, 99.8, 100)      # bearish
    c2 = C(99.8, 102, 99.7, 101.5)     # bullish, engulfs c1 body
    c3 = C(101.5, 102.5, 101.4, 102.2) # bullish continuation
    assert is_bullish_engulfing(c1, c2, c3)


def test_bearish_engulfing_with_confirmation() -> None:
    c1 = C(100, 101.2, 99.9, 101)
    c2 = C(101.2, 101.3, 99.5, 99.8)
    c3 = C(99.8, 99.9, 98.5, 98.8)
    assert is_bearish_engulfing(c1, c2, c3)


# ---------------------------------------------------------------------------
# 7-8. Hammer / Shooting Star + confirmation
# ---------------------------------------------------------------------------


def test_hammer_with_confirmation() -> None:
    c1 = C(102, 102.1, 100.9, 101)     # bearish
    # Hammer: tiny body near top, long lower wick
    c2 = C(101.1, 101.2, 99.5, 101.0)  # body 0.1, lower wick 1.5
    c3 = C(101.0, 102.5, 100.9, 102.0) # bullish, close > c2 close
    assert is_hammer(c1, c2, c3)


def test_shooting_star_with_confirmation() -> None:
    c1 = C(100, 101.1, 99.9, 101)
    c2 = C(100.9, 102.5, 100.8, 101.0) # body 0.1, upper wick 1.5
    c3 = C(101.0, 101.1, 99.5, 99.8)
    assert is_shooting_star(c1, c2, c3)


# ---------------------------------------------------------------------------
# 9-10. Piercing / Dark Cloud + confirmation
# ---------------------------------------------------------------------------


def test_piercing_with_confirmation() -> None:
    c1 = C(105, 105.1, 99.9, 100)      # big bear, midpoint 102.5
    c2 = C(99.5, 103.5, 99.4, 103)     # opens below c1 close, closes > midpoint, < c1 open
    c3 = C(103, 104, 102.9, 103.8)
    assert is_piercing(c1, c2, c3)


def test_dark_cloud_with_confirmation() -> None:
    c1 = C(100, 105.1, 99.9, 105)      # big bull, midpoint 102.5
    c2 = C(105.5, 105.6, 101.5, 102)   # opens above c1 close, closes < midpoint, > c1 open
    c3 = C(102, 102.1, 100.9, 101)
    assert is_dark_cloud(c1, c2, c3)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_dispatch_returns_first_match() -> None:
    # Build a 3 White Soldiers setup, dispatch should pick it first.
    c1 = C(100, 101, 99.8, 101)
    c2 = C(101, 102, 100.8, 102)
    c3 = C(102, 103, 101.8, 103)
    match = detect_first_pattern(c1, c2, c3)
    assert match is not None
    assert match[0] == "three_white_soldiers"
    assert match[1] == "call"


def test_dispatch_returns_none_when_no_pattern() -> None:
    # Three flat doji-like candles
    c1 = C(100, 100.1, 99.9, 100)
    c2 = C(100, 100.1, 99.9, 100)
    c3 = C(100, 100.1, 99.9, 100)
    assert detect_first_pattern(c1, c2, c3) is None
