"""Tests for the SPY 2DTE momentum-trigger strategy."""
from __future__ import annotations

from datetime import date

import pandas as pd

from iron_condor.config import (
    PROFIT_SCENARIOS,
    STOP_SCENARIOS,
    StrategyParams,
)
from iron_condor.orb import _one_step_itm_strike, _two_dte_expiry, _wilder_rsi


def test_default_params() -> None:
    p = StrategyParams()
    assert p.price_move_threshold == 0.15
    assert p.rsi_period == 14
    assert p.rsi_min == 30.0
    assert p.rsi_max == 70.0
    assert p.max_attempts == 5
    assert p.dte == 2
    assert p.entry_start.hour == 10
    assert p.entry_start.minute == 34


def test_two_dte_skips_weekends() -> None:
    # Monday 2026-05-11 + 2 business days = Wednesday 2026-05-13
    assert _two_dte_expiry(date(2026, 5, 11)) == date(2026, 5, 13)
    # Tuesday -> Thursday
    assert _two_dte_expiry(date(2026, 5, 12)) == date(2026, 5, 14)
    # Wednesday -> Friday
    assert _two_dte_expiry(date(2026, 5, 13)) == date(2026, 5, 15)
    # Thursday -> Monday (skip weekend)
    assert _two_dte_expiry(date(2026, 5, 14)) == date(2026, 5, 18)
    # Friday -> Tuesday (skip weekend)
    assert _two_dte_expiry(date(2026, 5, 15)) == date(2026, 5, 19)


def test_one_step_itm_strike_call_below_spot() -> None:
    # Spot 590.55 -> 590 call (first strike below)
    assert _one_step_itm_strike(590.55, "C") == 590.0
    # Spot 590.01 -> 590 call
    assert _one_step_itm_strike(590.01, "C") == 590.0


def test_one_step_itm_strike_call_when_spot_on_strike() -> None:
    # Spot exactly 590.00 -> 589 (590 is ATM, not ITM)
    assert _one_step_itm_strike(590.0, "C") == 589.0


def test_one_step_itm_strike_put_above_spot() -> None:
    # Spot 590.55 -> 591 put
    assert _one_step_itm_strike(590.55, "P") == 591.0
    # Spot 590.99 -> 591 put
    assert _one_step_itm_strike(590.99, "P") == 591.0


def test_one_step_itm_strike_put_when_spot_on_strike() -> None:
    # Spot exactly 590.00 -> 591 (590 is ATM, not ITM)
    assert _one_step_itm_strike(590.0, "P") == 591.0


def test_wilder_rsi_known_values() -> None:
    # Monotonically increasing prices -> RSI should approach 100
    close = pd.Series([float(i) for i in range(1, 30)])
    rsi = _wilder_rsi(close, period=14)
    assert rsi.iloc[-1] > 90.0
    # Monotonically decreasing prices -> RSI should approach 0
    close_down = pd.Series([float(30 - i) for i in range(30)])
    rsi_down = _wilder_rsi(close_down, period=14)
    assert rsi_down.iloc[-1] < 10.0


def test_sweep_grid_dimensions() -> None:
    # 3 profit scenarios x 10 stop scenarios = 30 configs
    assert len(PROFIT_SCENARIOS) == 3
    assert len(STOP_SCENARIOS) == 10
    # exactly one of (pct, minutes) is non-zero per stop scenario
    for sl_pct, sl_min in STOP_SCENARIOS:
        assert (sl_pct > 0 and sl_min == 0) or (sl_pct == 0 and sl_min > 0)
