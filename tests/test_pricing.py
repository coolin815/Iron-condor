"""Smoke tests for the pricing module."""
from __future__ import annotations

import math
from datetime import datetime, date

import pytz

from iron_condor.pricing import (
    bs_delta,
    bs_price,
    implied_vol,
    time_to_expiry_years,
)


ET = pytz.timezone("America/New_York")


def test_atm_call_put_parity() -> None:
    s, k, t, r, sigma = 100.0, 100.0, 0.25, 0.045, 0.2
    c = bs_price(s, k, t, r, sigma, "C")
    p = bs_price(s, k, t, r, sigma, "P")
    # Call - Put = S - K*exp(-rT)
    assert math.isclose(c - p, s - k * math.exp(-r * t), abs_tol=1e-6)


def test_iv_roundtrip() -> None:
    s, k, t, r, sigma = 500.0, 500.0, 5 / (365 * 24), 0.045, 0.18
    price = bs_price(s, k, t, r, sigma, "C")
    iv = implied_vol(price, s, k, t, r, "C")
    assert iv is not None
    assert abs(iv - sigma) < 1e-3


def test_delta_signs() -> None:
    s, k, t, r, sigma = 500.0, 500.0, 0.01, 0.045, 0.2
    assert 0 < bs_delta(s, k, t, r, sigma, "C") < 1
    assert -1 < bs_delta(s, k, t, r, sigma, "P") < 0
    # Deep ITM call delta -> 1, deep OTM call delta -> 0
    assert bs_delta(600, 500, t, r, sigma, "C") > 0.9
    assert bs_delta(400, 500, t, r, sigma, "C") < 0.1


def test_time_to_expiry_intraday() -> None:
    now = ET.localize(datetime(2024, 5, 17, 10, 0))  # 10:00 AM ET
    t = time_to_expiry_years(now, date(2024, 5, 17))
    # 6 hours until 4 PM expiry
    expected = 6 * 3600 / (365 * 24 * 3600)
    assert abs(t - expected) < 1e-9
