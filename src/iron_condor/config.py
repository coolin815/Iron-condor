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
    # SPY gap: skip the day if (today_open - prior_close) / prior_close is below
    # gap_min_pct. Negative number = down-gap threshold (e.g. -0.005 = -0.5%).
    gap_filter_enabled: bool = True
    gap_min_pct: float = -0.005

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
SHORT_OTM_PCTS: tuple[float, ...] = (0.005, 0.010, 0.015, 0.020)
SPREAD_WIDTHS: tuple[float, ...] = (2.0, 5.0)
PROFIT_TARGETS: tuple[float, ...] = (0.25, 0.50, 0.75)
STOP_LOSS_MULTS: tuple[float, ...] = (1.5, 2.0, 3.0)
