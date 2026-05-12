"""Smoke tests for the RSI implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from iron_condor.indicators import wilder_rsi


def test_rsi_canonical_series() -> None:
    # Classic Wilder example series (from his 1978 book), period 14.
    closes = pd.Series(
        [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
            46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45,
            45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66,
            43.13,
        ]
    )
    rsi = wilder_rsi(closes, period=14)
    # The hand-computed RSI value for the last point in this canonical series
    # is ~37.3 (well-known reference).
    assert np.isfinite(rsi.iloc[-1])
    assert 35.0 < rsi.iloc[-1] < 40.0


def test_rsi_warmup_is_nan() -> None:
    closes = pd.Series([1.0, 2.0, 3.0, 4.0])
    rsi = wilder_rsi(closes, period=14)
    assert rsi.isna().all()


def test_rsi_flat_series_is_50() -> None:
    closes = pd.Series([100.0] * 30)
    rsi = wilder_rsi(closes, period=14).dropna()
    assert (rsi == 50.0).all()


def test_rsi_monotonic_up_is_100() -> None:
    closes = pd.Series(np.linspace(100, 200, 30))
    rsi = wilder_rsi(closes, period=14).dropna()
    # All gains, no losses -> RSI saturates at 100.
    assert rsi.iloc[-1] == 100.0
