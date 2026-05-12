"""Entry-signal integration tests (no network)."""
from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd
import pytz

from iron_condor.config import StrategyParams
from iron_condor.strategy import find_first_signal

ET = pytz.timezone("America/New_York")


def _bars(closes: list[float], start_ts: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(start_ts, periods=len(closes), freq="1min")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=idx)


def test_no_signal_when_flat() -> None:
    start = pd.Timestamp("2024-05-17 09:30", tz=ET)
    bars = _bars([500.0] * 200, start)
    sig = find_first_signal(bars, StrategyParams(rsi_period=14))
    assert sig is None


def test_signal_after_uptrend_in_window() -> None:
    # Flat for 20 mins (covers RSI warmup), then a steady uptrend that pushes
    # RSI above 70.
    start = pd.Timestamp("2024-05-17 09:30", tz=ET)
    closes = [500.0] * 20 + list(np.linspace(500, 510, 60))
    bars = _bars(closes, start)
    sig = find_first_signal(bars, StrategyParams(rsi_period=9))
    assert sig is not None
    assert sig.direction == "over"
    assert sig.timestamp.time() >= time(9, 50)


def test_signal_before_window_is_ignored() -> None:
    # Sharp move BEFORE 9:50 should not trigger, but a subsequent move after
    # 9:50 should.
    start = pd.Timestamp("2024-05-17 09:30", tz=ET)
    closes = list(np.linspace(500, 510, 18)) + [510.0] * 5 + list(np.linspace(510, 500, 60))
    bars = _bars(closes, start)
    sig = find_first_signal(bars, StrategyParams(rsi_period=9))
    if sig is not None:
        assert sig.timestamp.time() >= time(9, 50)


def test_signal_after_cutoff_is_ignored() -> None:
    # Spike happens at 15:00 ET — well past 14:00 cutoff.
    start = pd.Timestamp("2024-05-17 09:30", tz=ET)
    closes = [500.0] * 330 + list(np.linspace(500, 520, 30))
    bars = _bars(closes, start)
    sig = find_first_signal(bars, StrategyParams(rsi_period=9))
    assert sig is None
