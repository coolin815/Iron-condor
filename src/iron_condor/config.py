"""Strategy parameters for the SPY 0DTE breakout-and-reversal strategy.

Single trade per day, whichever signal (breakout or reversal) fires first.
All times are NY-market local (US/Eastern).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Literal

UNDERLYING: str = "SPY"
RISK_FREE_RATE: float = 0.045

SignalMode = Literal["both", "breakout", "reversal"]


# ---------------------------------------------------------------------------
# Single-run strategy parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyParams:
    # Signal config
    or_window_min: int = 30                # opening range = first 30 min
    earliest_entry: time = time(10, 0)     # immediately after OR window closes
    # Per-signal cutoffs (per user spec). 12:30 ET = 9:30 PT; 13:00 ET = 10:00 PT
    reversal_latest_entry: time = time(12, 30)
    breakout_latest_entry: time = time(13, 0)
    time_stop_min: int = 60                # cap holding period
    hard_close: time = time(15, 55)        # safety net

    # Filter values (per the strategy spec)
    rsi_long_thresh: float = 50.0          # cross-day RSI must be > this for calls
    rsi_short_thresh: float = 50.0         # cross-day RSI must be < this for puts
    rsi_extreme_high: float = 70.0         # breakout skip if RSI > this in last 5 min
    rsi_extreme_low: float = 30.0          # breakout skip if RSI < this in last 5 min
    reversal_call_skip_lo: float = 60.0    # reversal calls skip if RSI in [lo, hi]
    reversal_call_skip_hi: float = 65.0
    skip_fridays: bool = True
    signal_mode: SignalMode = "both"       # which signal type(s) to enable

    # P&L measurement mode:
    #   "gross"  - exits trigger on option mid-price change (matches simple
    #              backtest tools that ignore the bid/ask spread + commissions)
    #   "net"    - exits trigger on net P&L / capital deployed (after fees +
    #              paying ask / receiving bid). More realistic but stricter.
    pnl_mode: Literal["gross", "net"] = "gross"

    # Net-of-fees exits
    profit_target_pct: float = 0.10        # +10%
    stop_loss_pct: float = 0.20            # -20%

    # Execution
    commission_per_contract: float = 0.85
    leg_half_spread: float = 0.005

    # Account
    starting_balance: float = 1500.0
    max_capital_per_trade: float = 50000.0


# ---------------------------------------------------------------------------
# Sweep grids (small — strategy is mostly fixed per spec)
# ---------------------------------------------------------------------------

SIGNAL_MODES: tuple[SignalMode, ...] = ("both", "breakout", "reversal")
TIME_STOPS: tuple[int, ...] = (30, 60, 120)

# Entry cutoffs — when do we stop opening new positions for the day
ENTRY_CUTOFFS: tuple[time, ...] = (
    time(11, 30),
    time(13, 0),
    time(15, 30),
)
