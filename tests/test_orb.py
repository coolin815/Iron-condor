"""Tests for the ORB level computation and signal detection."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from iron_condor.config import StrategyParams
from iron_condor.orb import compute_levels, find_orb_signal


def _make_bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """Build a DataFrame from a list of (time_iso_et, open, high, low, close)."""
    times = pd.to_datetime([r[0] for r in rows]).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low":  [r[3] for r in rows],
            "close":[r[4] for r in rows],
        },
        index=times,
    )


def test_levels_computed_from_extended_hours_bars() -> None:
    # Yesterday: regular session high 500, low 495
    yest = _make_bars([
        ("2026-05-11 09:30", 498, 500, 497, 499),
        ("2026-05-11 15:59", 498, 499, 495, 498),
        # after hours: 16:00-20:00
        ("2026-05-11 18:00", 498, 502, 498, 501),
    ])
    # Today: premarket then regular session
    today = _make_bars([
        ("2026-05-12 07:00", 501, 503, 500, 502),  # premarket
        ("2026-05-12 09:30", 502, 504, 501, 503),  # OR start
        ("2026-05-12 09:34", 503, 505, 502, 504),  # still in OR (5-min)
        ("2026-05-12 09:35", 504, 506, 503, 505),  # outside 5-min OR
    ])
    levels = compute_levels(today, yest, or_window_min=5)
    assert levels.pdh == 500.0
    assert levels.pdl == 495.0
    assert levels.pmh == 503.0   # 502 ask high; max premarket high
    assert levels.pml == 500.0
    # Overnight = yesterday after-hours (high 502) + today premarket (high 503)
    assert levels.onh == 503.0
    assert levels.onl == 498.0
    # OR (first 5 min, 9:30 and 9:34 - both before 09:35)
    assert levels.orh == 505.0
    assert levels.orl == 501.0


def test_orb_long_signal_fires_on_first_break_of_orh() -> None:
    today = _make_bars([
        ("2026-05-12 09:30", 500, 501, 499, 500),
        ("2026-05-12 09:31", 500, 502, 500, 501),  # OR high=502
        ("2026-05-12 09:34", 501, 502, 500, 501),
        ("2026-05-12 09:50", 501, 502, 500, 501),  # not yet broken
        ("2026-05-12 09:51", 501, 503, 501, 503),  # break: high 503 > 502
        ("2026-05-12 10:30", 503, 505, 503, 504),
    ])
    yest = _make_bars([
        ("2026-05-11 09:30", 499, 500, 498, 499),
        ("2026-05-11 15:59", 499, 500, 498, 499),
    ])
    levels = compute_levels(today, yest, or_window_min=5)
    params = StrategyParams(or_window_min=5)
    sig = find_orb_signal(today, levels, params)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.timestamp.time() == time(9, 51)
    assert sig.break_price == 503.0
    assert "ORH" in sig.level_broken


def test_orb_short_signal_fires_on_break_of_orl() -> None:
    today = _make_bars([
        ("2026-05-12 09:30", 500, 502, 498, 500),  # OR low=498
        ("2026-05-12 09:34", 500, 501, 498, 500),
        ("2026-05-12 09:51", 500, 500, 497, 497),  # break: low 497 < 498
    ])
    levels = compute_levels(today, _make_bars([]), or_window_min=5)
    params = StrategyParams(or_window_min=5)
    sig = find_orb_signal(today, levels, params)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.break_price == 497.0


def test_orb_signal_skipped_if_outside_window() -> None:
    today = _make_bars([
        ("2026-05-12 09:30", 500, 502, 498, 500),
        ("2026-05-12 09:34", 500, 501, 498, 500),
        # Break in the OR is ignored; need to wait until earliest_entry
        ("2026-05-12 09:35", 500, 503, 500, 502),  # past OR but pre-window
    ])
    levels = compute_levels(today, _make_bars([]), or_window_min=5)
    # Set earliest_entry to 9:45 so the 9:35 break is ignored
    params = StrategyParams(or_window_min=5, earliest_entry=time(9, 45))
    sig = find_orb_signal(today, levels, params)
    assert sig is None


def test_min_break_pct_blocks_weak_break() -> None:
    """A break that barely clears ORH should be filtered out when min_break_pct > 0."""
    today = _make_bars([
        ("2026-05-12 09:30", 500, 502, 498, 500),   # ORH = 502
        ("2026-05-12 09:34", 500, 502, 498, 500),
        ("2026-05-12 09:51", 500, 502.10, 500, 502.05),  # break by 0.05 (~0.01%)
    ])
    levels = compute_levels(today, _make_bars([]), or_window_min=5)
    # No filter -> signal
    p_loose = StrategyParams(or_window_min=5)
    assert find_orb_signal(today, levels, p_loose) is not None
    # 0.05% required (~0.25 on 502) -> filtered
    p_tight = StrategyParams(or_window_min=5, min_break_pct=0.0005)
    assert find_orb_signal(today, levels, p_tight) is None


def test_vol_mult_blocks_low_volume_break() -> None:
    """Break-bar volume below threshold should fail the filter."""
    rows = []
    # Build OR bars with reference volume 1000
    for i, t in enumerate(["09:30", "09:31", "09:32", "09:33", "09:34"]):
        rows.append((f"2026-05-12 {t}", 500, 502, 498, 500))
    # Quiet bars after OR (low vol baseline)
    rows.extend([
        ("2026-05-12 09:35", 500, 501, 499, 500),
        ("2026-05-12 09:50", 500, 501, 499, 500),
        ("2026-05-12 09:51", 500, 503, 501, 503),   # break, but we'll set its volume
    ])
    df = _make_bars(rows)
    # Set volumes: baseline avg = 1000, breakout bar volume = 800 (below 1.5x)
    df["volume"] = [1000] * (len(rows) - 1) + [800]
    levels = compute_levels(df, _make_bars([]), or_window_min=5)
    p = StrategyParams(or_window_min=5, vol_mult=1.5)
    assert find_orb_signal(df, levels, p) is None
    # If the breakout bar has 2000 volume (2x avg), it passes
    df.loc[df.index[-1], "volume"] = 2000
    assert find_orb_signal(df, levels, p) is not None


def test_vwap_filter_blocks_long_when_close_below_vwap() -> None:
    """Long signal must close above session VWAP. The break-bar itself has a
    long upper wick + heavy volume that drives VWAP above its close."""
    rows = [
        ("2026-05-12 09:30", 500, 502, 498, 500),
        ("2026-05-12 09:34", 500, 502, 498, 500),
        # Break: high 510 > ORH=502 but close 502.5 below the bar's own
        # typical (~504) which heavy volume pulls VWAP to ~503.5.
        ("2026-05-12 09:51", 500, 510, 500, 502.5),
    ]
    df = _make_bars(rows)
    df["volume"] = [1000, 1000, 10000]
    levels = compute_levels(df, _make_bars([]), or_window_min=5)
    # Loose signal fires (high 510 > ORH 502)
    assert find_orb_signal(df, levels, StrategyParams(or_window_min=5)) is not None
    # VWAP filter blocks: VWAP ~ 503.5 but close 502.5
    p = StrategyParams(or_window_min=5, vwap_filter=True)
    assert find_orb_signal(df, levels, p) is None


def test_premarket_bias_blocks_long_when_premarket_down() -> None:
    """With premarket_bias=True, long signal blocked if premarket trended down."""
    today = _make_bars([
        # Premarket: down trend (501 -> 499)
        ("2026-05-12 06:00", 501, 501, 500, 500.5),
        ("2026-05-12 09:00", 500, 500, 499, 499),
        # Regular session
        ("2026-05-12 09:30", 500, 502, 498, 500),
        ("2026-05-12 09:34", 500, 502, 498, 500),
        ("2026-05-12 09:51", 500, 503, 500, 503),
    ])
    levels = compute_levels(today, _make_bars([]), or_window_min=5)
    p = StrategyParams(or_window_min=5, premarket_bias=True)
    assert find_orb_signal(today, levels, p) is None
    # Without filter, signal would fire
    p_loose = StrategyParams(or_window_min=5)
    assert find_orb_signal(today, levels, p_loose) is not None


def test_orb_confluence_pdh_blocks_weak_breaks() -> None:
    """Long break must clear PDH=510 for confluence='any' to pass."""
    yest = _make_bars([
        ("2026-05-11 12:00", 505, 510, 505, 510),  # PDH = 510
    ])
    today = _make_bars([
        ("2026-05-12 09:30", 500, 502, 499, 500),  # OR high = 502
        ("2026-05-12 09:51", 500, 503, 500, 503),  # break ORH but NOT PDH
    ])
    levels = compute_levels(today, yest, or_window_min=5)
    params = StrategyParams(or_window_min=5, confluence="any")
    assert find_orb_signal(today, levels, params) is None  # PDH blocked
    # With confluence='none', the same bars would fire.
    params2 = StrategyParams(or_window_min=5, confluence="none")
    assert find_orb_signal(today, levels, params2) is not None
