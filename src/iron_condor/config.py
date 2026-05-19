"""Strategy parameters for the SPY consecutive-20min-candle trigger.

Entry trigger (each trading day, anchored at 9:30 ET):
  - Build 20-min OHLC candles: [9:30, 9:50), [9:50, 10:10), [10:10, 10:30), ...
  - A candle is "green" if close > open, "red" if close < open. Doji breaks streak.
  - On the first occurrence of two consecutive same-direction candles, fire signal.
  - Entry fills at the OPEN of the NEXT 1-min candle after the 2nd 20-min candle closes.
  - Latest possible entry time: params.latest_entry (default 12:30 ET == 9:30 PT).

Position:
  - ATM strike (nearest $1 to spot, round half up).
  - 0 DTE or 2 DTE — swept as a dimension to compare.

Exit:
  - Profit target on mid-to-mid (gross) or net-after-fees (per pnl_mode).
  - Stop loss as a % of capital (matches pnl_mode).
  - Hard close at 3:55 PM ET fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"


@dataclass(frozen=True)
class StrategyParams:
    # -- Entry trigger --
    candle_minutes: int = 20
    latest_entry: time = time(12, 30)      # 12:30 ET == 9:30 PT

    # -- RSI extreme filter --
    # If RSI(rsi_period) on rsi_candle_minutes-bar SPY closes (same-day) is
    # outside [rsi_min, rsi_max] AT the close of the 2nd 20-min trigger candle,
    # skip the trade. Filter is silently bypassed if RSI hasn't warmed up yet.
    rsi_filter_enabled: bool = True
    rsi_period: int = 14
    rsi_candle_minutes: int = 5
    rsi_min: float = 30.0
    rsi_max: float = 70.0

    # -- Contract selection --
    dte: int = 0                           # 0 (today's expiry) or 2 (today + 2 BD)
    strike_step: float = 1.0               # SPY $1 strikes at ATM

    # -- Exits --
    hard_close: time = time(15, 55)
    profit_target_pct: float = 0.05
    stop_loss_pct: float = 0.20
    pnl_mode: Literal["gross", "net"] = "gross"

    # -- Execution --
    commission_per_contract: float = 0.85
    leg_half_spread: float = 0.01

    # -- Account --
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# Sweep grid — (profit_target_pct, pnl_mode) pairs
PROFIT_SCENARIOS: tuple[tuple[float, str], ...] = (
    (0.03, "gross"),
    (0.03, "net"),
    (0.05, "gross"),
    (0.05, "net"),
    (0.10, "gross"),
)

# Sweep grid — stop_loss_pct values; mode is paired with PT's mode
STOP_SCENARIOS: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)

# Sweep grid — DTE values to compare
DTE_VALUES: tuple[int, ...] = (0, 2)
