"""Black-Scholes pricing, IV solve, and delta — used for delta-targeted strike
selection on 0DTE.

We treat the option as European on a non-dividend underlying (SPY actually pays
dividends but their effect on 0DTE deltas is negligible). All times-to-expiry
are in calendar years using a 365-day year.
"""
from __future__ import annotations

import math
from datetime import datetime, time
from typing import Literal

from scipy.stats import norm

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
EXPIRY_TIME_ET = time(16, 0)  # SPY 0DTE options expire at 4:00 PM ET


def time_to_expiry_years(now_et: datetime, expiry_date) -> float:
    """Time from `now_et` (tz-aware ET) until 4:00 PM ET on `expiry_date`, in years.

    Clamped to a small positive value so BS doesn't blow up at exact expiry.
    """
    expiry_dt = datetime.combine(expiry_date, EXPIRY_TIME_ET).replace(
        tzinfo=now_et.tzinfo
    )
    seconds = (expiry_dt - now_et).total_seconds()
    return max(seconds / SECONDS_PER_YEAR, 1e-6)


def bs_price(
    s: float, k: float, t: float, r: float, sigma: float, right: Literal["C", "P"]
) -> float:
    """Black-Scholes European price for one share."""
    if sigma <= 0 or t <= 0:
        intrinsic = max(s - k, 0.0) if right == "C" else max(k - s, 0.0)
        return intrinsic
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if right == "C":
        return s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def bs_delta(
    s: float, k: float, t: float, r: float, sigma: float, right: Literal["C", "P"]
) -> float:
    if sigma <= 0 or t <= 0:
        if right == "C":
            return 1.0 if s > k else 0.0
        return -1.0 if s < k else 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    return norm.cdf(d1) if right == "C" else norm.cdf(d1) - 1.0


def implied_vol(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    right: Literal["C", "P"],
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float | None:
    """Solve BS implied vol via bisection. Returns None if no solution in [0.01, 5]."""
    if price <= 0 or t <= 0:
        return None
    intrinsic = max(s - k, 0.0) if right == "C" else max(k - s, 0.0)
    if price < intrinsic - 0.01:
        return None  # price below intrinsic — bad data

    lo, hi = 0.01, 5.0
    p_lo = bs_price(s, k, t, r, lo, right)
    p_hi = bs_price(s, k, t, r, hi, right)
    if (p_lo - price) * (p_hi - price) > 0:
        # Price outside bracketed range.
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(s, k, t, r, mid, right)
        if abs(p_mid - price) < tol:
            return mid
        if (p_lo - price) * (p_mid - price) < 0:
            hi = mid
            p_hi = p_mid
        else:
            lo = mid
            p_lo = p_mid
    return 0.5 * (lo + hi)
