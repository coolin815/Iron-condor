"""Strategy parameters for the SPY 0DTE 'follow the flow' strategy.

Copies large BUY-aggressor option prints. Single trade/day, no Fridays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"


@dataclass(frozen=True)
class StrategyParams:
    # Signal
    earliest_entry: time = time(9, 35)     # right after the open
    latest_entry: time = time(15, 0)       # 12:00 PT — no new entries this late
    time_stop_min: int = 30
    hard_close: time = time(15, 55)
    skip_fridays: bool = True

    # Flow-detection threshold (single print size in contracts)
    size_threshold: int = 1500

    # Strike scope: only look at contracts within ±this many dollars of spot
    strike_window: float = 5.0

    # P&L measurement: "gross" (option mid-to-mid) or "net" (after fills)
    pnl_mode: Literal["gross", "net"] = "gross"

    # Entry timing:
    #   "instant"        — fill at print_price + full bid-ask spread (models a
    #                      WebSocket-driven bot reacting in ~1 sec, paying the
    #                      ask just above the print)
    #   "next_bar_open"  — fill at the OPEN of the bar AFTER the print, +
    #                      half-spread. More conservative (~60s reaction delay).
    entry_mode: Literal["instant", "next_bar_open"] = "instant"

    # Exits on the option price (single-leg long):
    profit_target_pct: float = 0.30    # +30% on option mid
    stop_loss_pct: float = 0.30        # -30% on option mid

    # Execution
    commission_per_contract: float = 0.85
    leg_half_spread: float = 0.01      # single-leg ATM bid-ask half

    # Account
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# Sweep grids
SIZE_THRESHOLDS: tuple[int, ...] = (1000, 1500, 2000, 2500)
PROFIT_TARGETS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50, 1.00)
STOP_LOSSES: tuple[float, ...] = (0.20, 0.30, 0.50)
TIME_STOPS: tuple[int, ...] = (15, 30, 60)
ENTRY_MODES: tuple[str, ...] = ("instant", "next_bar_open")
