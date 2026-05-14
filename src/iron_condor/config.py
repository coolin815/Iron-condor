"""Strategy parameters for the SPY 0DTE credit-spread strategy.

ORB-direction credit spreads. Single trade/day, no Fridays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"

DirectionMode = Literal["continuation", "reversion"]


@dataclass(frozen=True)
class StrategyParams:
    # Signal
    or_window_min: int = 30                # opening range = first 30 min
    earliest_entry: time = time(10, 0)     # immediately after OR closes
    latest_entry: time = time(13, 0)       # 10:00 AM PT
    time_stop_min: int = 60                # cap holding period
    hard_close: time = time(15, 55)
    skip_fridays: bool = True

    # ORB direction interpretation:
    #   "continuation" — break above ORH -> bull put; below ORL -> bear call
    #   "reversion"   — break above ORH -> bear call; below ORL -> bull put
    direction_mode: DirectionMode = "reversion"

    # Strike selection (per spread)
    short_strike_offset: float = 1.0       # short strike $X OTM from spot
    spread_width: float = 1.0              # long strike $X further OTM

    # P&L measurement: "gross" (mid-to-mid spread value) or "net" (after fills)
    pnl_mode: Literal["gross", "net"] = "gross"

    # Exits (as fraction of CREDIT COLLECTED, TastyTrade-style):
    #   PT X% -> exit when spread value <= (1 - X) * entry_credit
    #   SL X% -> exit when spread value >= (1 + X) * entry_credit
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 0.30

    # Execution
    commission_per_contract: float = 0.85
    leg_half_spread: float = 0.005

    # Account
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# Sweep grids (per the user's spec)
PROFIT_TARGETS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
STOP_LOSSES: tuple[float, ...] = (0.10, 0.20, 0.30)
TIME_STOPS: tuple[int, ...] = (30, 60, 120)
