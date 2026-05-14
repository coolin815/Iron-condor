"""Thin Polygon REST wrapper with on-disk caching and throttling.

Designed for the Polygon "Options Developer" plan: unlimited historical, but
hold ~3 requests/second to be polite. All cached responses live under
`data/cache/` and are keyed by endpoint + parameters so reruns are cheap.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time as _time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
RATE_LIMIT_PER_SEC = 10           # Conservative default. Bump in .env via
                                  # POLYGON_RATE_LIMIT_PER_SEC if your plan has
                                  # real headroom and you stop seeing 429s.
MAX_RETRIES = 10


# ---------------------------------------------------------------------------
# Rate limiter — simple sliding window
# ---------------------------------------------------------------------------


class _Throttle:
    def __init__(self, rate_per_sec: int) -> None:
        self.rate = rate_per_sec
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = _time.monotonic()
            while self._times and now - self._times[0] >= 1.0:
                self._times.popleft()
            if len(self._times) >= self.rate:
                sleep_for = 1.0 - (now - self._times[0])
                if sleep_for > 0:
                    _time.sleep(sleep_for)
                now = _time.monotonic()
                while self._times and now - self._times[0] >= 1.0:
                    self._times.popleft()
            self._times.append(_time.monotonic())


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PolygonClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        rate_per_sec: int | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "POLYGON_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if rate_per_sec is None:
            rate_per_sec = int(
                os.environ.get("POLYGON_RATE_LIMIT_PER_SEC", RATE_LIMIT_PER_SEC)
            )
        log.info("PolygonClient throttle = %d req/s", rate_per_sec)
        self.throttle = _Throttle(rate_per_sec)

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _request(
        self, url: str, params: dict[str, Any], label: str
    ) -> dict[str, Any]:
        for attempt in range(MAX_RETRIES):
            self.throttle.wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                wait = min(60.0, 2.0 ** attempt)
                log.warning(
                    "Polygon %s -> network error (%s); sleeping %.0fs (attempt %d/%d)",
                    label, type(e).__name__, wait, attempt + 1, MAX_RETRIES,
                )
                _time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 60.0
                log.warning(
                    "Polygon %s -> 429 rate-limited, sleeping %.0fs (attempt %d/%d)",
                    label, wait, attempt + 1, MAX_RETRIES,
                )
                _time.sleep(wait)
                continue
            if resp.status_code in (502, 503, 504):
                wait = min(60.0, 2.0 ** attempt)
                log.warning(
                    "Polygon %s -> %s, sleeping %.0fs (attempt %d/%d)",
                    label, resp.status_code, wait, attempt + 1, MAX_RETRIES,
                )
                _time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Polygon {label} failed after {MAX_RETRIES} retries")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        return self._request(f"{BASE_URL}{path}", params, path)

    def _paginate(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        data = self._get(path, params)
        out.extend(data.get("results") or [])
        next_url = data.get("next_url")
        while next_url:
            data = self._request(
                next_url, {"apiKey": self.api_key}, f"{path} (next_url)"
            )
            out.extend(data.get("results") or [])
            next_url = data.get("next_url")
        return out

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, namespace: str, key: str, ext: str) -> Path:
        d = self.cache_dir / namespace
        d.mkdir(parents=True, exist_ok=True)
        safe = hashlib.sha1(key.encode()).hexdigest()[:16] if "/" in key else key
        return d / f"{safe}.{ext}"

    # ------------------------------------------------------------------
    # Stocks: SPY 1-min bars
    # ------------------------------------------------------------------

    def get_minute_bars(
        self, ticker: str, day: date, force: bool = False
    ) -> pd.DataFrame:
        """1-minute aggregates for `ticker` on a single calendar day.

        Returns a DataFrame indexed by ET timestamp with columns
        [open, high, low, close, volume, vwap, n]. Returns an empty frame if
        Polygon doesn't have the data (e.g. 403 on today's incomplete date).
        """
        cache = self._cache_path(f"bars_{ticker}", day.isoformat(), "parquet")
        if cache.exists() and not force:
            return pd.read_parquet(cache)

        path = f"/v2/aggs/ticker/{ticker}/range/1/minute/{day}/{day}"
        try:
            data = self._get(
                path, {"adjusted": "true", "sort": "asc", "limit": 50000}
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                log.warning(
                    "Polygon 403 for %s on %s — no permission/data; skipping day",
                    ticker, day,
                )
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume", "vwap", "n"]
                )
            raise
        results = data.get("results") or []
        if not results:
            df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap", "n"]
            )
        else:
            df = pd.DataFrame(results).rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "n",
                    "t": "ts_ms",
                }
            )
            df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(
                "America/New_York"
            )
            df = df.drop(columns=["ts_ms"])
            keep = ["open", "high", "low", "close", "volume", "vwap", "n"]
            df = df[[c for c in keep if c in df.columns]]
        df.to_parquet(cache)
        return df

    # ------------------------------------------------------------------
    # Options: contracts for an expiration date
    # ------------------------------------------------------------------

    def get_option_contracts(
        self, underlying: str, expiration_date: date, force: bool = False
    ) -> list[dict[str, Any]]:
        """Return every option contract for `underlying` expiring on the given date.

        Includes expired contracts (which is what we want for historical days).
        """
        ns = f"contracts_{underlying}"
        cache = self._cache_path(ns, expiration_date.isoformat(), "json")
        if cache.exists() and not force:
            return json.loads(cache.read_text())

        results = self._paginate(
            "/v3/reference/options/contracts",
            {
                "underlying_ticker": underlying,
                "expiration_date": expiration_date.isoformat(),
                "expired": "true",
                "limit": 1000,
            },
        )
        cache.write_text(json.dumps(results))
        return results

    # ------------------------------------------------------------------
    # Options: 1-minute aggregates for a single contract
    # ------------------------------------------------------------------

    def get_option_minute_bars(
        self, option_ticker: str, day: date, force: bool = False
    ) -> pd.DataFrame:
        cache = self._cache_path(
            f"opt_bars/{option_ticker}", day.isoformat(), "parquet"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists() and not force:
            return pd.read_parquet(cache)

        path = f"/v2/aggs/ticker/{option_ticker}/range/1/minute/{day}/{day}"
        try:
            data = self._get(
                path, {"adjusted": "true", "sort": "asc", "limit": 50000}
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                log.warning(
                    "Polygon 403 for %s on %s — no permission/data; skipping",
                    option_ticker, day,
                )
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume", "vwap", "n"]
                )
            raise
        results = data.get("results") or []
        if not results:
            df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap", "n"]
            )
        else:
            df = pd.DataFrame(results).rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "n",
                    "t": "ts_ms",
                }
            )
            df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(
                "America/New_York"
            )
            df = df.drop(columns=["ts_ms"])
            keep = ["open", "high", "low", "close", "volume", "vwap", "n"]
            df = df[[c for c in keep if c in df.columns]]
        df.to_parquet(cache)
        return df

    # ------------------------------------------------------------------
    # Options: tick-by-tick trades for one contract on one day
    # ------------------------------------------------------------------

    def get_option_trades(
        self, option_ticker: str, day: date, force: bool = False
    ) -> pd.DataFrame:
        """All trade prints for an option contract on a single calendar day.

        Returns a DataFrame with columns [sip_timestamp_ns, price, size,
        exchange, conditions] sorted ascending by sip_timestamp_ns. Empty
        frame if no trades or no permission.
        """
        cache = self._cache_path(
            f"opt_trades/{option_ticker}", day.isoformat(), "parquet"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists() and not force:
            return pd.read_parquet(cache)

        # Trades endpoint expects timestamps in nanoseconds or RFC3339.
        # Use UTC ISO date to cover the full calendar day.
        start_iso = f"{day.isoformat()}T00:00:00Z"
        end_iso = f"{day.isoformat()}T23:59:59Z"
        path = f"/v3/trades/{option_ticker}"
        params = {
            "timestamp.gte": start_iso,
            "timestamp.lte": end_iso,
            "order": "asc",
            "sort": "timestamp",
            "limit": 50000,
        }
        try:
            results = self._paginate(path, params)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                log.warning(
                    "Polygon 403 trades %s on %s — no permission; skipping",
                    option_ticker, day,
                )
                return pd.DataFrame(
                    columns=["sip_timestamp_ns", "price", "size", "exchange", "conditions"]
                )
            raise
        if not results:
            df = pd.DataFrame(
                columns=["sip_timestamp_ns", "price", "size", "exchange", "conditions"]
            )
        else:
            df = pd.DataFrame(results)
            # Polygon's trade objects use "sip_timestamp" (nanoseconds) and
            # "price"/"size"/"exchange"/"conditions" (list). Standardize.
            keep = {
                "sip_timestamp": "sip_timestamp_ns",
                "price": "price",
                "size": "size",
                "exchange": "exchange",
                "conditions": "conditions",
            }
            df = df.rename(columns=keep)
            cols = [c for c in keep.values() if c in df.columns]
            df = df[cols]
            # Conditions is a list[int]; coerce to string for parquet stability.
            if "conditions" in df.columns:
                df["conditions"] = df["conditions"].astype(str)
        df.to_parquet(cache)
        return df


# ---------------------------------------------------------------------------
# Polygon option-ticker helpers
# ---------------------------------------------------------------------------


def build_option_ticker(
    underlying: str, expiry: date, right: str, strike: float
) -> str:
    """Build a Polygon option ticker, e.g. O:SPY240517C00525000."""
    right = right.upper()
    assert right in ("C", "P"), right
    yymmdd = expiry.strftime("%y%m%d")
    strike_int = int(round(strike * 1000))
    return f"O:{underlying}{yymmdd}{right}{strike_int:08d}"


def parse_option_ticker(ticker: str) -> tuple[str, date, str, float]:
    """Inverse of build_option_ticker."""
    body = ticker.removeprefix("O:")
    # Underlying is the leading alpha block (1–6 chars for our purposes).
    i = 0
    while i < len(body) and body[i].isalpha():
        i += 1
    underlying = body[:i]
    rest = body[i:]
    yymmdd, right, strike_str = rest[:6], rest[6], rest[7:]
    expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    strike = int(strike_str) / 1000.0
    return underlying, expiry, right, strike
