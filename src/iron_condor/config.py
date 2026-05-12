"""Strategy parameters and sweep grids.

All times are NY-market local (US/Eastern). Convert from PT by adding 3h.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Literal


# ---------------------------------------------------------------------------
# Single-run strategy parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrikeRule:
    """How to pick the four iron-condor strikes at entry.

    `mode="fixed"` uses dollar offsets from the signal-bar SPY price:
      - long_inner_offset:  distance from spot to the long inner strikes
      - wing_width:         distance between long inner and short wing
    `mode="delta"` picks strikes whose Black-Scholes delta is closest to the
    target on each side.
    """

    mode: Literal["fixed", "delta"]
    name: str
    long_inner_offset: float | None = None  # used when mode="fixed"
    wing_width: float | None = None         # used when mode="fixed"
    inner_delta: float | None = None        # used when mode="delta", e.g. 0.25
    outer_delta: float | None = None        # used when mode="delta", e.g. 0.10


@dataclass(frozen=True)
class StrategyParams:
    rsi_period: int = 14
    rsi_upper: float = 70.0
    rsi_lower: float = 30.0

    # Entry window, NY time
    earliest_entry: time = time(9, 50)   # 6:50 AM PT
    latest_entry: time = time(14, 0)     # 11:00 AM PT
    time_stop: time = time(14, 30)       # 11:30 AM PT
    hard_close: time = time(15, 55)      # safety net before 4 PM ET

    profit_target_pct: float = 0.25      # 25% of debit paid
    stop_loss_pct: float = 0.35          # lose 35% of debit -> exit

    strike_rule: StrikeRule = field(
        default_factory=lambda: StrikeRule(
            mode="fixed", name="fixed_1.5x4", long_inner_offset=1.5, wing_width=4.0
        )
    )

    # Execution assumptions
    # Default = IBKR Pro Fixed ($0.65/contract, plus negligible regulatory).
    # IBKR Pro Tiered users can drop this to ~0.30; Robinhood ~0.04;
    # Tastytrade caps at ~$0.50 round-trip-averaged.
    commission_per_contract: float = 0.65
    # Slippage is applied at the COMBO net price, once per side (open + close).
    # 4-leg ICs are submitted as a single combo order at IBKR / Tastytrade /
    # Schwab, so you cross the bid-ask once on the package, not four times.
    # $0.05/share per side ≈ $10 round-trip friction on one IC.
    combo_slippage_per_share: float = 0.05

    # Account
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 20000.0


# ---------------------------------------------------------------------------
# Sweep grid (used by backtest.run_sweep)
# ---------------------------------------------------------------------------


RSI_PERIODS: tuple[int, ...] = (9, 14)

PROFIT_TARGETS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)

STOP_LOSSES: tuple[float, ...] = (0.25, 0.35, 0.50)

# Entry-window cutoffs in NY (ET) time. PT in parens for sanity:
#   12:30 ET = 9:30 PT,  13:00 ET = 10:00 PT,
#   13:30 ET = 10:30 PT, 14:00 ET = 11:00 PT
ENTRY_CUTOFFS: tuple[time, ...] = (
    time(12, 30),
    time(13, 0),
    time(13, 30),
    time(14, 0),
)

STRIKE_RULES: tuple[StrikeRule, ...] = (
    # Fixed offsets from the brief: $1.50–$2 inner, $3–$4 wings.
    StrikeRule(mode="fixed", name="fixed_1.0x3", long_inner_offset=1.0, wing_width=3.0),
    StrikeRule(mode="fixed", name="fixed_1.5x3", long_inner_offset=1.5, wing_width=3.0),
    StrikeRule(mode="fixed", name="fixed_1.5x4", long_inner_offset=1.5, wing_width=4.0),
    StrikeRule(mode="fixed", name="fixed_2.0x4", long_inner_offset=2.0, wing_width=4.0),
    StrikeRule(mode="fixed", name="fixed_2.0x5", long_inner_offset=2.0, wing_width=5.0),
    # Delta-targeted (volatility-aware).
    StrikeRule(mode="delta", name="delta_25_10", inner_delta=0.25, outer_delta=0.10),
    StrikeRule(mode="delta", name="delta_30_15", inner_delta=0.30, outer_delta=0.15),
)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

RISK_FREE_RATE: float = 0.045  # 4.5% — close enough for 0DTE delta math
UNDERLYING: str = "SPY"
