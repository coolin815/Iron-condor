"""Strategy parameters for the SPY 2DTE momentum-trigger strategy.

Entry trigger (around 10:34 ET / 7:34 PT):
  - SPY 1-min close vs today's 9:30 ET regular-session open
  - If close >= open + $0.15  -> 1-step ITM 2DTE CALL
  - If close <= open - $0.15  -> 1-step ITM 2DTE PUT
  - RSI(14) on same-day SPY 1-min closes must be 30-70
  - Try each subsequent minute up to `max_attempts` (default 5) if no entry yet

Exit:
  - Profit target (PT) hit on mid-to-mid or net-after-fees (per pnl_mode)
  - Stop loss — either a % loss (gross/net per pnl_mode) OR an N-minute time stop
  - Hard close at 3:55 PM ET to avoid overnight gap risk
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"


@dataclass(frozen=True)
class StrategyParams:
    # -- Entry trigger --
    entry_start: time = time(10, 34)       # 10:34 ET == 7:34 PT
    max_attempts: int = 5                  # consecutive 1-min checks; give up after
    price_move_threshold: float = 0.15     # required move vs 9:30 open ($)

    # -- RSI gate --
    rsi_period: int = 14                   # Wilder, same-day
    rsi_min: float = 30.0
    rsi_max: float = 70.0

    # -- Contract selection --
    dte: int = 2                           # 2 trading days to expiration
    strike_step: float = 1.0               # SPY $1 strikes at ATM

    # -- Exits --
    hard_close: time = time(15, 55)
    profit_target_pct: float = 0.10        # 10% gross by default
    stop_loss_pct: float = 0.30            # 30% (0 if using time stop)
    stop_loss_minutes: int = 0             # >0 means time stop, ignore stop_loss_pct
    pnl_mode: Literal["gross", "net"] = "gross"

    # -- Execution --
    commission_per_contract: float = 0.85
    leg_half_spread: float = 0.01

    # -- Account --
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# Sweep grid — (profit_target_pct, pnl_mode) pairs
PROFIT_SCENARIOS: tuple[tuple[float, str], ...] = (
    (0.05, "gross"),
    (0.05, "net"),
    (0.10, "gross"),
)

# Sweep grid — (stop_loss_pct, stop_loss_minutes) pairs. Exactly one is non-zero per row.
STOP_SCENARIOS: tuple[tuple[float, int], ...] = (
    (0.20, 0),
    (0.25, 0),
    (0.30, 0),
    (0.35, 0),
    (0.40, 0),
    (0.45, 0),
    (0.50, 0),
    (0.0, 60),
    (0.0, 90),
    (0.0, 120),
)
