"""Tests for the SPY 0DTE short put credit spread strategy."""
from __future__ import annotations

from iron_condor.config import (
    PROFIT_TARGETS,
    SHORT_OTM_PCTS,
    SPREAD_WIDTHS,
    STOP_LOSS_MULTS,
    StrategyParams,
)
from iron_condor.orb import _nearest_strike


def test_default_params() -> None:
    p = StrategyParams()
    assert p.entry_time.hour == 9 and p.entry_time.minute == 35
    assert p.short_otm_pct == 0.01
    assert p.spread_width == 2.0
    assert p.profit_target_pct == 0.50
    assert p.stop_loss_mult == 2.0


def test_nearest_strike_snaps_to_step() -> None:
    avail = {580.0, 581.0, 582.0, 583.0, 584.0, 585.0}
    # Target 582.4 -> nearest is 582
    assert _nearest_strike(582.4, 1.0, avail) == 582.0
    # Target 582.6 -> nearest is 583
    assert _nearest_strike(582.6, 1.0, avail) == 583.0


def test_nearest_strike_falls_back_when_snap_missing() -> None:
    # Snap would give 590 but it's not in chain — pick closest available.
    avail = {585.0, 588.0, 591.0}
    assert _nearest_strike(590.0, 1.0, avail) == 591.0
    assert _nearest_strike(586.0, 1.0, avail) == 585.0


def test_nearest_strike_empty_returns_none() -> None:
    assert _nearest_strike(590.0, 1.0, set()) is None


def test_sweep_grid_dimensions() -> None:
    assert len(SHORT_OTM_PCTS) == 4
    assert len(SPREAD_WIDTHS) == 2
    assert len(PROFIT_TARGETS) == 3
    assert len(STOP_LOSS_MULTS) == 3
    total = (
        len(SHORT_OTM_PCTS)
        * len(SPREAD_WIDTHS)
        * len(PROFIT_TARGETS)
        * len(STOP_LOSS_MULTS)
    )
    assert total == 72
