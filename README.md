# SPY 0DTE Long Iron Condor Backtester

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/coolin815/iron-condor/blob/claude/spy-options-trading-bot-ri4Ms/notebooks/run.ipynb)

Backtest a **long (reverse) iron condor** on SPY 0DTE options. The thesis is
non-directional: when intraday RSI hits an extreme, SPY is more likely to make a
large move (either continuation or reversal), and a long condor profits from a
breakout in either direction.

Full strategy spec: see `condor strategy brief.md`.

**Running on a phone?** Tap the Colab badge above. No install required — paste
your Polygon key when prompted and step through the cells.

## What it does

For each trading day in the lookback window:

1. Pulls SPY 1-minute bars from Polygon.
2. Computes RSI(9) and RSI(14) on the 1-min closes.
3. Looks for the **first** RSI cross above 70 or below 30 between
   **6:50 AM PT and 11:00 AM PT** (9:50 AM – 2:00 PM ET).
4. Pulls that day's option chain and the entry-minute prices for the four legs.
5. Constructs a long iron condor — buy inner put + inner call, sell outer
   wings — using either:
   - **Fixed** dollar offsets from spot ($1/$1.50/$2/$3 inner with $3–$5 wings), or
   - **Delta-targeted** strikes (inner ≈ 25Δ, outer ≈ 10Δ) computed from
     Black-Scholes IV solved at entry.
6. Sizes the trade with the full available balance (capped at $20,000 once the
   account reaches $20k), one trade per day.
7. Walks the position forward minute-by-minute on real option mid prices and
   exits on the first of:
   - Profit target hit (sweep: 10 / 15 / 20 / 25 / 30 %)
   - Stop loss hit (35 % of debit, configurable)
   - Time stop at 11:30 AM PT (2:30 PM ET)

Each combination of (RSI period × strike rule × profit target) is a separate
backtest run with its own equity curve.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, paste your Polygon key
```

## Run

Smoke test on a single recent day (verifies API + math):

```bash
python -m iron_condor.cli --smoke
```

Full one-year sweep:

```bash
python -m iron_condor.cli --days 365
```

Results land in `results/` as CSVs (per-trade log + summary table).

## Layout

```
src/iron_condor/
  config.py          # Strategy params + sweep grids
  polygon_client.py  # API wrapper, on-disk cache, throttling
  indicators.py      # Wilder RSI
  pricing.py         # Black-Scholes, IV solve, delta
  strategy.py        # Entry signal + strike selection
  backtest.py        # Per-day simulation + parameter sweep
  metrics.py         # Equity / drawdown / win rate
  cli.py             # Entrypoint
tests/               # Unit tests for the math
data/cache/          # Polygon response cache (gitignored)
results/             # Backtest output (gitignored)
```

## Notes

- Polygon Developer plan is assumed (unlimited historical + 1-min option aggs).
- Option prices simulated at the **midpoint** of the 1-min OHLC. A configurable
  per-contract slippage and a per-contract commission can be set in `config.py`.
- The backtest is intentionally pessimistic on missing data: if any leg is
  missing a bar at the entry minute, the day is skipped.
