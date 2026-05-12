"""Technical indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on a Series of closes.

    Uses the original RMA (exponentially weighted with alpha = 1/period,
    seeded from the simple average of the first `period` gains/losses).
    Returns a Series aligned to `closes` with NaNs for the warmup window.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    delta = closes.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss is 0 and avg_gain > 0, RSI is 100.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # When both are 0 (flat), RSI is undefined; pandas-friendly default = 50.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder RMA: SMA seed then alpha = 1/period EMA."""
    s = series.copy().astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(s) < period:
        return out
    initial = s.iloc[:period].mean()
    out.iloc[period - 1] = initial
    alpha = 1.0 / period
    prev = initial
    for i in range(period, len(s)):
        prev = prev + alpha * (s.iat[i] - prev)
        out.iat[i] = prev
    return out
