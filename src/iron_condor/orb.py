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
- Multi-leg filtering uses Polygon's OPRA condition codes. Prints flagged
  as part of a multi-leg / spread / stock-tied order are skipped when
  params.exclude_multi_leg is True. The condition-code set lives in
  config.MULTI_LEG_CONDITION_CODES.
- No Fridays.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal

import pandas as pd

from .config import MULTI_LEG_CONDITION_CODES, StrategyParams
from .polygon_client import PolygonClient, build_option_ticker

log = logging.getLogger(__name__)


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


def _parse_conditions(raw: object) -> set[int]:
    """Parse Polygon's trade-conditions field into a set of int codes.

    Polygon returns conditions as a list[int]. After parquet caching we
    stored it as a string repr like "[1, 152]" — handle both.
    """
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        try:
            return {int(c) for c in raw}
        except (TypeError, ValueError):
            return set()
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s in ("nan", "[]", "None"):
            return set()
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return set()
        if isinstance(parsed, (list, tuple)):
            try:
                return {int(c) for c in parsed}
            except (TypeError, ValueError):
                return set()
    return set()


def _is_multi_leg(conditions_raw: object) -> bool:
    return bool(_parse_conditions(conditions_raw) & MULTI_LEG_CONDITION_CODES)


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
    """Dispatch to the single_print or clustered scanner per params.signal_mode."""
    if params.signal_mode == "clustered":
        return _find_clustered_signal(
            today_1min_spy, contracts, client, params, underlying
        )
    return _find_single_print_signal(
        today_1min_spy, contracts, client, params, underlying
    )


def _find_single_print_signal(
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
    n_large = 0
    n_multi_leg = 0
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
            n_large += 1
            if params.exclude_multi_leg and _is_multi_leg(row.get("conditions")):
                n_multi_leg += 1
                continue
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

    if n_large:
        log.debug(
            "%s: %d large prints, %d filtered as multi-leg",
            day, n_large, n_multi_leg,
        )
    if best is None:
        return None
    return best[1]


def _find_clustered_signal(
    today_1min_spy: pd.DataFrame,
    contracts: list[dict],
    client: PolygonClient,
    params: StrategyParams,
    underlying: str = "SPY",
) -> Signal | None:
    """Scan ATM-area 0DTE trade tape for the first 1-min candle on a single
    contract containing >= params.cluster_min_trades buy-aggressor prints of
    size >= params.size_threshold."""
    if today_1min_spy.empty:
        return None
    if params.skip_fridays:
        day = today_1min_spy.index[0].tz_convert("America/New_York").date()
        if _is_friday(day):
            return None
    day = today_1min_spy.index[0].tz_convert("America/New_York").date()

    et = _to_et(today_1min_spy)
    open_bar = et[et.index.time == time(9, 30)]
    if open_bar.empty:
        return None
    spot_at_open = float(open_bar.iloc[0]["close"])

    nearby = _strikes_near_spot(spot_at_open, contracts, window=5.0)
    if not nearby:
        return None

    earliest = params.earliest_entry
    latest = params.latest_entry
    threshold = params.size_threshold
    min_count = max(1, int(params.cluster_min_trades))

    best: tuple[pd.Timestamp, Signal] | None = None
    n_clusters_found = 0
    n_large_total = 0
    n_multi_leg_total = 0
    n_sell_total = 0
    n_buy_kept_total = 0
    max_per_minute_seen = 0
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

        # Build per-minute groups of qualifying buy-aggressor prints.
        by_minute: dict[pd.Timestamp, list[dict]] = {}
        for _, row in big.iterrows():
            n_large_total += 1
            if params.exclude_multi_leg and _is_multi_leg(row.get("conditions")):
                n_multi_leg_total += 1
                continue
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
            aggressor = _classify_aggressor(trade_price, bar_open)
            if aggressor != "buy":
                if aggressor == "sell":
                    n_sell_total += 1
                continue
            n_buy_kept_total += 1
            by_minute.setdefault(minute, []).append({
                "ts": ts,
                "price": trade_price,
                "size": int(row["size"]),
                "bar_open": bar_open,
            })
        if by_minute:
            local_max = max(len(rs) for rs in by_minute.values())
            if local_max > max_per_minute_seen:
                max_per_minute_seen = local_max

        # Earliest minute on this contract with >= min_count qualifying prints.
        qualifying = sorted(
            (m for m, rs in by_minute.items() if len(rs) >= min_count)
        )
        if not qualifying:
            continue
        minute = qualifying[0]
        rows_sorted = sorted(by_minute[minute], key=lambda r: r["ts"])
        trigger = rows_sorted[-1]  # last print in the cluster
        n_clusters_found += 1

        total_size = sum(r["size"] for r in by_minute[minute])
        entry_minute = minute + pd.Timedelta(minutes=1)
        sig = Signal(
            timestamp=entry_minute,
            print_timestamp=trigger["ts"],
            contract=ticker,
            strike=strike,
            right=right,
            size=total_size,  # combined size across the cluster
            trade_price=trigger["price"],
            bar_open=(
                trigger["bar_open"]
                if trigger["bar_open"] is not None
                else float("nan")
            ),
        )
        if best is None or trigger["ts"] < best[0]:
            best = (trigger["ts"], sig)

    log.debug(
        "%s [clustered S>=%d N>=%d]: large=%d multi_leg=%d sell=%d buy_kept=%d "
        "max_per_minute=%d clusters_found=%d",
        day, threshold, min_count,
        n_large_total, n_multi_leg_total, n_sell_total, n_buy_kept_total,
        max_per_minute_seen, n_clusters_found,
    )
    if best is None:
        return None
    return best[1]
