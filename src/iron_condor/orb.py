"""SPY 0DTE 'follow the flow' strategy.

Scans 0DTE option trade tape for large BUY-aggressor prints. The first
qualifying print of the day triggers a copy trade: buy the same contract,
manage with PT / SL / time stop.

Notes / honest caveats:
- Aggressor classification is a proxy. We compare each trade's price to
  the contract's 1-min bar OPEN for that minute:
    trade_price >= bar_open  -> classify as BUY aggressor
    trade_price <  bar_open  -> classify as SELL aggressor (we skip those)
  A more accurate classification needs NBBO quote data, which is much
  heavier to fetch.
- Multi-leg / hedge detection is NOT done. A 1000-contract call print
  might be one leg of a calendar spread or a covered call; we'll trade
  it as if it were a directional bet. Known noise source.
- No Fridays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal

import pandas as pd

from .config import StrategyParams
from .polygon_client import PolygonClient, build_option_ticker


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp           # the minute we'd enter (next bar after the print)
    print_timestamp: pd.Timestamp     # when the institutional print hit the tape
    contract: str                     # option ticker we're copying
    strike: float
    right: Literal["C", "P"]
    size: int                         # how big the print was
    trade_price: float                # price the print hit
    bar_open: float                   # for the aggressor classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_et(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    out = bars.copy()
    out.index = bars.index.tz_convert("America/New_York")
    return out


def _is_friday(day: date) -> bool:
    return day.weekday() == 4


def _strikes_near_spot(
    spot: float,
    contracts: list[dict],
    window: float = 5.0,
) -> list[dict]:
    """Filter the option chain to contracts whose strikes are within ±`window`
    dollars of `spot`. Returns the original contract dicts (with strike_price,
    contract_type, ...)."""
    out = []
    for c in contracts:
        try:
            k = float(c["strike_price"])
        except (TypeError, ValueError, KeyError):
            continue
        if abs(k - spot) <= window:
            out.append(c)
    return out


def _classify_aggressor(
    trade_price: float, bar_open: float | None
) -> Literal["buy", "sell", "unknown"]:
    if bar_open is None or pd.isna(bar_open):
        return "unknown"
    if trade_price >= bar_open:
        return "buy"
    return "sell"


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def find_signal(
    today_1min_spy: pd.DataFrame,
    contracts: list[dict],
    client: PolygonClient,
    params: StrategyParams,
    underlying: str = "SPY",
) -> Signal | None:
    """Scan ATM-area 0DTE trade tape for the first large BUY-aggressor print
    inside the entry window."""
    if today_1min_spy.empty:
        return None
    if params.skip_fridays:
        day = today_1min_spy.index[0].tz_convert("America/New_York").date()
        if _is_friday(day):
            return None
    day = today_1min_spy.index[0].tz_convert("America/New_York").date()

    # Use SPY's opening 9:30 bar as the anchor for picking nearby strikes.
    et = _to_et(today_1min_spy)
    open_bar = et[et.index.time == time(9, 30)]
    if open_bar.empty:
        return None
    spot_at_open = float(open_bar.iloc[0]["close"])

    nearby = _strikes_near_spot(spot_at_open, contracts, window=5.0)
    if not nearby:
        return None

    # Fetch trade tape + 1-min bars for every nearby contract on this day.
    # Then for each contract, find the first large buy-aggressor print
    # whose ET timestamp is inside the entry window.
    earliest = params.earliest_entry
    latest = params.latest_entry
    threshold = params.size_threshold

    best: tuple[pd.Timestamp, Signal] | None = None  # (timestamp, signal)
    for c in nearby:
        try:
            strike = float(c["strike_price"])
        except (TypeError, ValueError, KeyError):
            continue
        ctype = c.get("contract_type", "").lower()
        if ctype not in ("call", "put"):
            continue
        right = "C" if ctype == "call" else "P"
        ticker = build_option_ticker(underlying, day, right, strike)

        trades = client.get_option_trades(ticker, day)
        if trades.empty:
            continue
        big = trades[trades["size"] >= threshold]
        if big.empty:
            continue

        bars = client.get_option_minute_bars(ticker, day)
        if bars.empty:
            continue
        bars = _to_et(bars)

        for _, row in big.iterrows():
            ts_ns = int(row["sip_timestamp_ns"])
            ts = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")
            tt = ts.time()
            if tt < earliest or tt > latest:
                continue
            minute = ts.floor("min")
            try:
                bar = bars.loc[minute]
            except KeyError:
                continue
            bar_open = float(bar["open"]) if not pd.isna(bar["open"]) else None
            trade_price = float(row["price"])
            if _classify_aggressor(trade_price, bar_open) != "buy":
                continue
            # Entry happens at the NEXT 1-min bar's open (we react after the
            # print fully fires). signal.timestamp is the entry minute.
            entry_minute = (minute + pd.Timedelta(minutes=1))
            sig = Signal(
                timestamp=entry_minute,
                print_timestamp=ts,
                contract=ticker,
                strike=strike,
                right=right,
                size=int(row["size"]),
                trade_price=trade_price,
                bar_open=bar_open if bar_open is not None else float("nan"),
            )
            if best is None or ts < best[0]:
                best = (ts, sig)
            break  # first qualifying print per contract is enough

    if best is None:
        return None
    return best[1]
