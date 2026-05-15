"""Strategy parameters for the SPY 0DTE 'follow the flow' strategy.

Copies large BUY-aggressor option prints. Single trade/day, no Fridays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"


# Polygon/OPRA condition codes that indicate a print is part of a
# multi-leg / complex / spread / stock-tied order. A "buy" print with any
# of these codes is almost certainly one leg of a vertical, condor,
# risk-reversal, or stock+option combo — not a directional bet — so we
# skip it. This list is based on the commonly documented OPRA reference;
# if the filter is over- or under-aggressive you can adjust the set.
MULTI_LEG_CONDITION_CODES: frozenset[int] = frozenset({
    14,  # Multi-Leg Auto-Electronic Trade Against Single-Leg(s) with Stock
    15,  # Multi-Leg Auto-Electronic Trade
    17,  # Multi-Leg Cross
    19,  # Multi-Leg Floor Trade
    21,  # Multi-Leg Trade
    22,  # Multi-Leg with Stock
    27,  # Stock Options Auto-Electronic
    33,  # Stock Options Trade
    41,  # Multi-Leg Floor Trade of Proprietary Products
    44,  # Multi-Leg Auto-Electronic Trade of Proprietary Products
})


@dataclass(frozen=True)
class StrategyParams:
    # Signal
    earliest_entry: time = time(9, 35)     # right after the open
    latest_entry: time = time(15, 0)       # 12:00 PT — no new entries this late
    hard_close: time = time(15, 55)
    skip_fridays: bool = True

    # Flow-detection threshold (single print size in contracts)
    size_threshold: int = 1500

    # Strike scope: only look at contracts within ±this many dollars of spot
    strike_window: float = 5.0

    # Skip prints whose conditions indicate they're part of a multi-leg /
    # spread / stock-tied order (not a directional single-leg buy).
    exclude_multi_leg: bool = True

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
SIZE_THRESHOLDS: tuple[int, ...] = (3000, 3500, 4000, 4500, 5000)
PROFIT_TARGETS: tuple[float, ...] = (0.05, 0.10, 0.15)
STOP_LOSSES: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
ENTRY_MODES: tuple[str, ...] = ("instant", "next_bar_open")
