"""Strategy parameters for the SPY 0DTE candle-pattern strategy.

Scans 10 candle patterns on 5-min bars in parallel; first to fire (with all
indicator confirmations aligned) wins. One trade per day, no Fridays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"

# The 10 patterns we scan, in priority order (first match wins on a tie).
PATTERN_NAMES: tuple[str, ...] = (
    "three_white_soldiers",
    "three_black_crows",
    "morning_star",
    "evening_star",
    "bullish_engulfing",
    "bearish_engulfing",
    "hammer",
    "shooting_star",
    "piercing",
    "dark_cloud",
)


@dataclass(frozen=True)
class StrategyParams:
    # Signal config
    bar_timeframe_min: int = 5             # detection timeframe
    earliest_entry: time = time(9, 45)     # need at least 3 5-min bars before entry
    latest_entry: time = time(13, 0)       # 10 AM PT — no new trades after this
    time_stop_min: int = 60                # cap holding period
    hard_close: time = time(15, 55)        # safety net

    # Confirmation indicator thresholds
    rsi_long_thresh: float = 50.0
    rsi_short_thresh: float = 50.0
    skip_fridays: bool = True

    # Which patterns to scan. Default = all 10. CLI can narrow via --pattern.
    enabled_patterns: tuple[str, ...] = PATTERN_NAMES

    # P&L measurement: "gross" (option mid-to-mid) or "net" (after fees)
    pnl_mode: Literal["gross", "net"] = "gross"

    # Exits
    profit_target_pct: float = 0.10
    stop_loss_pct: float = 0.20

    # Execution
    commission_per_contract: float = 0.85
    leg_half_spread: float = 0.005

    # Account
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# Default sweep dimensions
TIME_STOPS: tuple[int, ...] = (30, 60, 120)
ENTRY_CUTOFFS: tuple[time, ...] = (
    time(11, 30),
    time(13, 0),
    time(15, 0),
)
