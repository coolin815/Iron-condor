"""Strategy parameters and sweep grids for the SPY 0DTE ORB strategy.

All times are NY-market local (US/Eastern).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"
RISK_FREE_RATE: float = 0.045  # kept for any greek calcs we add later

# ---------------------------------------------------------------------------
# Confluence levels
# ---------------------------------------------------------------------------

ConfluenceLevel = Literal["none", "pdh_pdl", "pmh_pml", "onh_onl", "any"]
# "none"    = take any ORH/ORL break
# "pdh_pdl" = long break must also clear PDH; short break must also clear PDL
# "pmh_pml" = same with premarket high/low
# "onh_onl" = same with overnight high/low (post yesterday's close, pre today's open)
# "any"     = clear any of PDH/PMH/ONH on long side (or PDL/PML/ONL on short)

# ---------------------------------------------------------------------------
# Single-run strategy parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyParams:
    # Signal
    or_window_min: int = 15                # opening range length in minutes
    confluence: ConfluenceLevel = "none"
    earliest_entry: time = time(9, 45)     # don't trade in the OR window itself
    latest_entry: time = time(11, 0)       # 8:00 AM PT — no new trades after this
    time_stop_min: int = 30                # max minutes to hold a position
    hard_close: time = time(15, 55)        # safety net before 4 PM ET

    # Filters (default = off / pass-through)
    min_break_pct: float = 0.0             # require break by >= this % past ORH/ORL
    vol_mult: float = 0.0                  # require break-bar vol >= vol_mult * 20-bar avg
    vwap_filter: bool = False              # long: close > VWAP; short: close < VWAP
    premarket_bias: bool = False           # long: premarket up; short: premarket down

    # Net-of-fees exits
    profit_target_pct: float = 0.05        # +5% on capital deployed (after fees)
    stop_loss_pct: float = 0.10            # -10% on capital deployed

    # Execution
    commission_per_contract: float = 0.85
    leg_half_spread: float = 0.005         # half of single-leg bid-ask

    # Account
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------

OR_WINDOWS: tuple[int, ...] = (5, 15, 30)

CONFLUENCE_LEVELS: tuple[ConfluenceLevel, ...] = ("none", "any")

PROFIT_TARGETS: tuple[float, ...] = (0.03, 0.05, 0.07, 0.10)

STOP_LOSSES: tuple[float, ...] = (0.05, 0.10, 0.15)

TIME_STOPS: tuple[int, ...] = (15, 30, 60)

# Filter sweep grids (default is just "off" so the base sweep is unchanged).
MIN_BREAK_PCTS: tuple[float, ...] = (0.0,)
VOL_MULTS: tuple[float, ...] = (0.0,)
VWAP_FILTERS: tuple[bool, ...] = (False,)
PREMARKET_BIASES: tuple[bool, ...] = (False,)

# Entry cutoffs in NY (ET) time. PT in parens:
#   10:30 ET = 7:30 PT, 11:00 ET = 8:00 PT, 11:30 ET = 8:30 PT
ENTRY_CUTOFFS: tuple[time, ...] = (
    time(10, 30),
    time(11, 0),
    time(11, 30),
)
