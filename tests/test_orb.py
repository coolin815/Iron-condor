"""Tests for the SPY 20-min-candle trigger strategy."""
from __future__ import annotations

from datetime import date

import pandas as pd

from iron_condor.config import (
    DTE_VALUES,
    PROFIT_SCENARIOS,
    STOP_SCENARIOS,
    StrategyParams,
)
from iron_condor.orb import (
    _aggregate_candles,
    _candle_direction,
    _expiry_for_dte,
    _nearest_atm_strike,
    _wilder_rsi,
)


def test_default_params() -> None:
    p = StrategyParams()
    assert p.candle_minutes == 20
    assert p.latest_entry.hour == 12
    assert p.latest_entry.minute == 30
    assert p.dte == 0
    assert p.strike_step == 1.0


def test_expiry_zero_dte_is_today() -> None:
    assert _expiry_for_dte(date(2026, 5, 11), 0) == date(2026, 5, 11)


def test_expiry_two_dte_skips_weekends() -> None:
    # Monday + 2 BD -> Wednesday
    assert _expiry_for_dte(date(2026, 5, 11), 2) == date(2026, 5, 13)
    # Thursday + 2 BD -> Monday
    assert _expiry_for_dte(date(2026, 5, 14), 2) == date(2026, 5, 18)
    # Friday + 2 BD -> Tuesday
    assert _expiry_for_dte(date(2026, 5, 15), 2) == date(2026, 5, 19)


def test_nearest_atm_strike_rounds_half_up() -> None:
    assert _nearest_atm_strike(590.55) == 591.0
    assert _nearest_atm_strike(590.49) == 590.0
    # Exactly on half rounds up
    assert _nearest_atm_strike(590.5) == 591.0
    assert _nearest_atm_strike(590.0) == 590.0


def test_candle_direction() -> None:
    assert _candle_direction(100.0, 100.5) == "green"
    assert _candle_direction(100.0, 99.5) == "red"
    assert _candle_direction(100.0, 100.0) is None  # doji


def test_aggregate_candles_into_20min_buckets() -> None:
    # Build 60 1-min bars starting at 9:30 ET on a single day.
    idx = pd.date_range("2026-05-11 09:30", periods=60, freq="1min", tz="America/New_York")
    df = pd.DataFrame({
        "open": [float(i) for i in range(60)],
        "high": [float(i) + 0.5 for i in range(60)],
        "low": [float(i) - 0.5 for i in range(60)],
        "close": [float(i) + 0.1 for i in range(60)],
        "volume": [100] * 60,
    }, index=idx)
    agg = _aggregate_candles(df, 20)
    # 60 min / 20 = 3 candles
    assert len(agg) == 3
    # First candle: open = bar 0's open (0.0), close = bar 19's close (19.1)
    assert agg.iloc[0]["open"] == 0.0
    assert agg.iloc[0]["close"] == 19.1
    # Labels at 9:30, 9:50, 10:10
    assert agg.index[0].hour == 9 and agg.index[0].minute == 30
    assert agg.index[1].hour == 9 and agg.index[1].minute == 50
    assert agg.index[2].hour == 10 and agg.index[2].minute == 10


def test_aggregate_candles_drops_premarket_bars() -> None:
    # Pre-market 1-min bars from 4:00 AM, plus a few regular-session bars.
    pre = pd.date_range("2026-05-11 04:00", "2026-05-11 09:29", freq="1min", tz="America/New_York")
    rth = pd.date_range("2026-05-11 09:30", "2026-05-11 10:09", freq="1min", tz="America/New_York")
    idx = pre.append(rth)
    df = pd.DataFrame({
        "open": [99.0] * len(idx),
        "high": [99.5] * len(idx),
        "low": [98.5] * len(idx),
        "close": [99.0] * len(idx),
        "volume": [10] * len(idx),
    }, index=idx)
    agg = _aggregate_candles(df, 20)
    # Two RTH candles (9:30-9:50 and 9:50-10:10), no pre-market candles.
    assert len(agg) == 2
    assert agg.index[0].hour == 9 and agg.index[0].minute == 30
    assert agg.index[1].hour == 9 and agg.index[1].minute == 50


def test_wilder_rsi_extremes() -> None:
    # Strictly increasing -> RSI -> 100
    up = pd.Series([float(i) for i in range(1, 30)])
    assert _wilder_rsi(up, period=14).iloc[-1] > 90.0
    # Strictly decreasing -> RSI -> 0
    down = pd.Series([float(30 - i) for i in range(30)])
    assert _wilder_rsi(down, period=14).iloc[-1] < 10.0


def test_default_rsi_filter_settings() -> None:
    p = StrategyParams()
    assert p.rsi_filter_enabled is True
    assert p.rsi_period == 14
    assert p.rsi_lookback_minutes == 5
    assert p.rsi_min == 30.0
    assert p.rsi_max == 70.0


def test_sweep_grid_dimensions() -> None:
    # 2 DTE x 5 PT x 4 SL = 40 configs
    assert len(DTE_VALUES) == 2
    assert len(PROFIT_SCENARIOS) == 5
    assert len(STOP_SCENARIOS) == 4
    assert len(DTE_VALUES) * len(PROFIT_SCENARIOS) * len(STOP_SCENARIOS) == 40
