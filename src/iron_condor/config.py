"""Strategy parameters for SPY 0DTE short put credit spread.

Each trading day at params.entry_time:
  1. Pick a target short-put strike = spot * (1 - short_otm_pct), rounded to $1.
  2. Long-put strike = short - spread_width.
  3. SELL the short put at bid, BUY the long put at ask. Net credit per share.
  4. Hold until:
       - PT: spread P&L >= profit_target_pct * credit (i.e. captured N% of credit)
       - SL: spread P&L <= -stop_loss_mult * credit (lost N× the credit received)
       - Hard close at 3:55 PM ET (avoid expiry pin risk)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"


@dataclass(frozen=True)
class StrategyParams:
    # -- Entry --
    entry_time: time = time(9, 35)
    hard_close: time = time(15, 55)

    # -- Strike selection --
    short_otm_pct: float = 0.01            # 1% OTM short strike (proxy for ~0.15 delta on 0DTE)
    spread_width: float = 2.0              # $ between short and long strike
    strike_step: float = 1.0               # SPY $1 strikes at ATM

    # -- Regime filters --
    # VIX: skip the day if prior trading day's VIX close was above vix_max.
    vix_filter_enabled: bool = True
    vix_max: float = 22.0
    # Overnight gap filter: skip if (today_9:30_open - prior_16:00_close) /
    # prior_close is below overnight_min_pct. Captures big overnight selloffs.
    overnight_filter_enabled: bool = False
    overnight_min_pct: float = -0.005
    # Premarket filter: skip if SPY's % change from 9:00 ET (6:00 PT) to 9:30 ET
    # (6:30 PT) is below premarket_min_pct. Captures last-30-min-of-premarket
    # weakness specifically.
    premarket_filter_enabled: bool = False
    premarket_min_pct: float = -0.005
    # When BOTH overnight and premarket filters are enabled, this controls the
    # combine logic: 'any' = skip if either fires; 'all' = skip only if both fire.
    filter_combine: Literal["any", "all"] = "any"

    # -- Exits --
    profit_target_pct: float = 0.50        # % of credit captured to take profit
    stop_loss_mult: float = 2.0            # SL fires when unrealized loss = N × credit

    # -- Execution --
    commission_per_contract: float = 0.85  # per leg per side — round trip = 4× this
    leg_half_spread: float = 0.01

    # -- Account --
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# Sweep grids
SHORT_OTM_PCTS: tuple[float, ...] = (0.005, 0.010, 0.015)
SPREAD_WIDTHS: tuple[float, ...] = (2.0, 5.0)
PROFIT_TARGETS: tuple[float, ...] = (0.25, 0.50)
STOP_LOSS_MULTS: tuple[float, ...] = (1.5, 2.0, 3.0)

# Regime filter modes — (label, overnight_on, premarket_on, combine_logic)
# Used as a sweep dimension to compare overnight vs premarket vs combined.
FILTER_MODES: tuple[tuple[str, bool, bool, str], ...] = (
    ("overnight", True,  False, "any"),
    ("premarket", False, True,  "any"),
    ("either",    True,  True,  "any"),
    ("both",      True,  True,  "all"),
)
