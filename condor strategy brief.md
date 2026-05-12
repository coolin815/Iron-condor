# SPY 0DTE Long Iron Condor — Strategy Brief

## Overview

This project backtests a **long iron condor** strategy on SPY using intraday RSI extremes as the entry signal. The thesis is non-directional: when SPY hits an RSI extreme intraday, something big is likely to happen — either continuation or a sharp reversal. A long condor profits if price moves significantly in either direction.

-----

## Strategy Logic

### Entry Signal

- Calculate RSI on **1-minute SPY candles** (period 10 is the starting point)
- Trigger when RSI crosses **above 70** (overbought) or **below 30** (oversold)
- Only take the **first signal of the day**
- Only enter if the signal fires within the entry window (see timing below)

### Entry Timing Window

- **Earliest entry: 6:50 AM PT (9:50 AM ET)** — do not enter before this
  - First 20 minutes after open are noise: false RSI signals, widest bid/ask spreads, IV at daily peak
  - An RSI extreme before 6:50 AM PT reflects open auction chaos, not real momentum
- **Sweet spot: 6:50–9:30 AM PT** — IV normalizing, real momentum signals, 90+ min before time stop
- **Still valid: 9:30–10:30 AM PT** — less runway but acceptable
- **Latest entry: 11:00 AM PT** — after this, not enough time before the time stop
- **Do not enter at the open** regardless of RSI — IV crush will work against a long condor even if SPY moves

### Strike Placement

- Place short strikes **$1.50–$2 from the signal price** — close enough that a $1 SPY move hits profit target quickly
- Long wings **$3–4 wide** from short strikes — cheap protection without excessive cost
- Strikes centered on the **signal price at time of entry**, not the open price
- **Asymmetric condor** worth testing: tighten the spread on the side RSI momentum points toward
  - RSI overbought → tighter call spread, wider put spread
  - RSI oversold → tighter put spread, wider call spread
  - Still non-directional but weighted toward the more likely breakout side

### Exit Rules

- **Profit target:** 25% of max profit — quick exit, reduces theta and gamma exposure
- **Stop loss:** 30–35% of debit paid — scaled down to match the faster exit approach
- **Time stop:** Hard close by **11:30 AM PT** — theta decay accelerates badly after this
- Rationale: at 25% target, average time to win is estimated 5–20 min; stop loss should reflect that flat = exit, not wait

### What “Win” Means in Backtest

- SPY’s high or low on any subsequent 1-min bar breaches the short strike distance from the signal price before the time stop
- This is a simplification — real P&L depends on options pricing, IV, and bid/ask spreads

-----

## Key Parameters to Test

|Parameter            |Starting Value    |Range to Test                            |
|---------------------|------------------|-----------------------------------------|
|Candle size          |1 min             |1, 2, 5                                  |
|RSI period           |10                |5, 10, 14                                |
|RSI threshold        |70 / 30           |65, 70, 75, 80                           |
|Short strike distance|$1.50–$2 from spot|1, 1.5, 2, 3                             |
|Wing width           |$3–4 wide         |3, 4, 5                                  |
|Spread symmetry      |symmetric         |symmetric vs asymmetric (RSI-directional)|
|Earliest entry       |6:50 AM PT        |fixed — do not test earlier              |
|Entry cutoff         |11:00 AM PT       |10:00, 10:30, 11:00, 11:30               |
|Time stop            |11:30 AM PT       |11:00, 11:30, 12:00, 12:30               |
|Profit target        |25% of max profit |15%, 20%, 25%, 30%, 40%, 50%             |
|Stop loss            |30–35% of debit   |25%, 30%, 35%, 50%                       |
|Days tested          |20–30             |go to 60+ once logic is confirmed        |

-----

## SPY Move Required to Hit Profit Target

Assumptions: $1.75 debit, $5-wide condor, max profit = $3.25. SPY move is approximate — actual condor response depends on delta, gamma, and IV at time of entry.

|Profit Target|$ Gain Needed|Est. SPY Move|
|-------------|-------------|-------------|
|5%           |$0.09        |~$0.25       |
|10%          |$0.18        |~$0.50       |
|15%          |$0.26        |~$0.75       |
|20%          |$0.35        |~$1.00       |
|25%          |$0.44        |~$1.25       |
|30%          |$0.53        |~$1.50       |
|35%          |$0.61        |~$1.75       |
|40%          |$0.70        |~$2.00       |
|45%          |$0.79        |~$2.25       |
|50%          |$0.88        |~$2.50       |

A $1.00–$1.50 SPY move within 30 minutes of an RSI extreme signal is realistic on an active day, pointing to **20–30% as the practical quick-exit target range**.

-----

## Data Source

- **Polygon.io** Developer plan
- Endpoint: `GET /v2/aggs/ticker/SPY/range/{candleMins}/minute/{date}/{date}`
- Params: `adjusted=true&sort=asc&limit=2000`
- API key is a single key from the Polygon dashboard (no secret key needed)
- Rate limit: stay at ~3–4 requests/sec on Developer plan

-----

## Backtest Output Goals

For each strike distance ($1, $2, $3), report:

- **Win rate** — % of signal days price broke through the strike before time stop
- **Avg time to win** — average minutes from signal to breakout
- **Median time to win**
- **Signal count** — how many days had a qualifying RSI signal
- Broken down by entry cutoff so we can find the optimal latest-entry time

-----

## Important Context

### Why long condor, not short?

- A **short iron condor** collects premium and profits from price staying calm
- A **long iron condor** pays a debit and profits from a big move in either direction
- This strategy uses RSI extremes specifically because they tend to precede large moves, making the long side appropriate

### Why not just a debit spread?

- A single debit spread requires picking direction
- This strategy is intentionally non-directional — we don’t know if the RSI extreme leads to continuation or reversal, just that something moves

### Ballpark options cost (for context, not hardcoded)

- With SPY ~$560–580, a long condor costs roughly **$1.50–$2.25 debit** depending on strike placement and time of entry
- Max profit ~$3.00–$3.50 if SPY blows through either short strike
- Entering earlier (after 6:50 AM PT) gets cheaper debit as IV normalizes through the morning
- These numbers need to be validated against real option chain data

### IV crush risk

- **Do not enter at the open** — IV is at its daily peak, and even if SPY moves, IV dropping will erode condor value
- By 6:50 AM PT, IV has partially normalized and the debit is more fairly priced
- IV crush is most dangerous in the first 20 minutes and on low-news days where the open gap fills quickly

### Theta risk

- On 0DTE, theta decay is severe — every hour without a move bleeds the debit
- This is why the entry cutoff and time stop are critical constraints
- The strategy only makes sense if entered with a clear momentum signal and closed fast

-----

## Existing Environment

- Python 3.9.6
- Project folder: `~/Documents/qqq_bot/` (existing SPY 0DTE bot project)
- Polygon.io Developer plan (existing account)
- IBKR paper trading port 4002

## Suggested New Project Folder

Create a separate folder, e.g. `~/Documents/condor_backtest/`, to keep this isolated from the existing ORB bot.

-----

## Suggested First Script

`backtest.py` — pulls N days of SPY 1-min data from Polygon, calculates rolling RSI, finds first qualifying signal per day, tests $1/$2/$3 strike distances, outputs a results table. Params configurable at the top of the file.

Dependencies likely needed: `requests`, `pandas`, `numpy`