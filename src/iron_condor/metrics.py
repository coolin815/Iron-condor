"""Performance metrics for a trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_run(trades: pd.DataFrame, starting_balance: float) -> dict:
    """Compute a per-run summary dict from a trade-log DataFrame."""
    taken = trades[trades["exit_reason"].isin(["profit", "stop", "time_stop"])]
    n_days = len(trades)
    n_trades = len(taken)
    if n_trades == 0:
        return {
            "starting_balance": starting_balance,
            "ending_balance": starting_balance,
            "total_return_pct": 0.0,
            "n_days": n_days,
            "n_trades": 0,
            "win_rate": float("nan"),
            "avg_net_pnl": float("nan"),
            "median_net_pnl": float("nan"),
            "max_drawdown_pct": 0.0,
            "profit_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    ending = float(taken["balance_after"].iloc[-1])
    wins = taken[taken["net_pnl"] > 0]
    equity = trades["balance_after"].ffill().fillna(starting_balance)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    return {
        "starting_balance": starting_balance,
        "ending_balance": ending,
        "total_return_pct": (ending / starting_balance - 1.0) * 100,
        "n_days": n_days,
        "n_trades": n_trades,
        "win_rate": len(wins) / n_trades,
        "avg_net_pnl": float(taken["net_pnl"].mean()),
        "median_net_pnl": float(taken["net_pnl"].median()),
        "max_drawdown_pct": max_dd * 100,
        "profit_exits": int((taken["exit_reason"] == "profit").sum()),
        "stop_exits": int((taken["exit_reason"] == "stop").sum()),
        "time_exits": int((taken["exit_reason"] == "time_stop").sum()),
    }


def summarize_sweep(sweep: pd.DataFrame, starting_balance: float) -> pd.DataFrame:
    rows = []
    for cfg, group in sweep.groupby("config", sort=False):
        s = summarize_run(group, starting_balance)
        s["config"] = cfg
        rows.append(s)
    df = pd.DataFrame(rows)
    cols = [
        "config",
        "ending_balance",
        "total_return_pct",
        "n_trades",
        "win_rate",
        "avg_net_pnl",
        "median_net_pnl",
        "max_drawdown_pct",
        "profit_exits",
        "stop_exits",
        "time_exits",
        "n_days",
    ]
    return df[cols].sort_values("total_return_pct", ascending=False)
