# Cross-venue crypto derivatives research — Kalshi / Polymarket / Deribit

One simultaneous snapshot of every live BTC and ETH contract across three venues,
put on a common measure, and searched for arbitrage.

## Pull the data

```
cd src
python3 fetch_kalshi.py      # KXBTCD/KXBTC/KXETHD/KXETH + order books
python3 fetch_polymarket.py  # above___ / price-on ladders + CLOB books
python3 fetch_deribit.py     # full option + future surface, depth on near expiries
python3 fetch_ref.py         # cross-venue spot, USDT peg, recent bars
python3 fetch_hist.py        # 2y hourly bars (Binance mirror + Coinbase)
python3 export.py            # -> out/*.csv
```

`api.binance.com` is geo-blocked from some hosts (HTTP 451); the fetchers use the
`data-api.binance.vision` mirror, which serves the same global book.

## Analyse

```
python3 scan_exact.py    # LP static-arbitrage scan inside each settlement print
python3 seasonality.py   # hour-of-day + day-of-week variance clock, bootstrapped
python3 rv.py            # every ladder fitted, all venues on one clock
python3 edge.py          # executable edge per contract, with robustness band
python3 weekend.py       # realised variance of the exact contract windows
python3 final.py         # weekend contracts priced off the Deribit anchor
python3 trade.py         # bootstrap validation of the forward-variance signal
```

## What the analysis concluded

* **No static arbitrage** inside any venue. Each event lists two redundant ladders
  settling on the identical print; an LP over the exact state partition, at book
  depth and after fees, found at most 6 bp across 176 events.
* **No level arbitrage** between Kalshi and Polymarket once the USDT peg is applied
  (currently −10.5 bp). Corrected, the two venues' implied forwards agree to $3 on
  a $63,650 underlying. Uncorrected, the peg alone fabricates a ~2 point edge on
  every at-the-money digital.
* **No shape edge versus Deribit** that survives execution. Apparent 1–3 vol point
  richness was mostly a stale-spot artifact; re-centred on each ladder's own implied
  forward and stressed ±2 vol points, what remains is pennies.
* **A real distortion in weekend forward variance.** Deribit lists no Saturday or
  Sunday expiry and Kalshi lists no weekend event, so Polymarket's Sat/Sun ladders
  are the only instruments in the universe exposed to weekend variance and have no
  listed hedge. Measured directly over two years (104 observations per weekday), the
  16:00Z→16:00Z Saturday window realises 0.24× midweek variance in BTC and 0.43× in
  ETH. Polymarket charges ~0.15× fair for the Saturday increment and ~2.0× (BTC) /
  ~2.5× (ETH) for the Sunday one — close to the right total for the weekend,
  distributed across the two days almost backwards.

The binding constraint is execution, not signal: the BTC Aug-9 variance strip costs
0.0051 to buy and fetches 0.0022 to sell. The edge is a maker trade, not a taker one.

`out/report.html` is the full write-up.

---

# The bot

`bot/` is a runnable market maker. Paper mode works against live data with no
credentials; live requires `BOT_MODE=live`, `KALSHI_LIVE=yes`, and keys.

```
cd bot
python3 run.py --cycles 5      # Kalshi digital ladders, paper
python3 poly_mm.py             # Polymarket range-ladder quote plan
```

| module | role |
|---|---|
| `pricing.py` | Deribit surface -> business clock -> recalibrated P(S>K) per strike |
| `strategy.py` | take/make decisions, sanity gates |
| `risk.py` | inventory, delta, limits, hedge sizing |
| `venues.py` | Kalshi (RSA-signed) + Deribit adapters |
| `run.py` | main loop |
| `poly_mm.py` | Polymarket range-ladder maker, with the weekend variance tilt |

## What the validation established

Run in this order; each one gates the next.

1. `backtest_clock.py` — a flat-calendar quoter's predicted vol is wrong by
   −26% at 11Z and +36% at 15Z. The business clock cuts mis-specification by
   **70.1% (BTC) / 69.9% (ETH)**, the same number in both assets independently.
2. `term_short.py` — variance accrues linearly in business time down to 10
   minutes, so extrapolating a 12h option to a 30m contract is safe in *scale*.
3. `calibrate.py` — but **not in shape**. Both clocks over-predict moderate
   moves: predicted-31% events happen 26% of the time. The real distribution is
   far more peaked than lognormal.
4. `recalib.py` — a one-parameter probit sharpening fixes it. Mean calibration
   error **3.7c -> 0.7c (BTC 1h)**, `b` decaying 1.22 -> 1.12 with horizon.
   *Without this the bot invents 4-7c of edge on short-dated contracts that is
   purely its own misspecification, and loses money on every one.*
5. `mmmap.py` — where quoting actually pays, after the 1c residual model error,
   fees and adverse selection.

## Where quoting pays

| book | median spread | net @30% adv. sel. | 24h flow |
|---|---|---|---|
| Polymarket ETH range | 7.0c | **+1.45c** | thin |
| Kalshi ETH range | 4.0c | +0.15c | 21k contracts |
| Polymarket ETH above | 3.0c | +0.05c | — |
| Kalshi BTC above | 1.0c | **-0.90c** | 2.03M contracts |

The liquidity and the edge are in different places. Kalshi's BTC ladder trades
two million contracts a day at a one-cent spread and loses money to quote;
Polymarket's ETH range ladder pays but has traded $18,307 across all six days.

**Capacity is the binding constraint, not signal.** At 15% of ETH-range flow
and 1.45c/contract that is about $6/day. This is a real edge on a small book,
not a high-return business, and it should be sized accordingly.

---

# Full-universe sweep

The first pass covered two coins and one contract type. This covers everything.

```
python3 universe.py        # all 272 Kalshi crypto series -> 106 tradable events
python3 fetch_coins.py     # 2y hourly for 18 coins
python3 volterm.py         # horizon-matched vol, calibrated against Deribit
python3 onetouch.py        # block-bootstrap barrier engine (intra-bar highs)
python3 scan_barrier.py    # price every one-touch market
python3 touched.py         # has a barrier already been hit since issuance?
python3 empirical_touch.py # model-free realised-frequency check
python3 lock.py            # risk-free locked arbitrage at full book depth
python3 calendar_arb.py    # one-touch monotonicity in time
```

## What the sweep found

Kalshi's largest crypto open interest is **not** in the price ladders. It is
15.1M contracts across the one-touch max/min series -- barrier payoffs on the
running extremum, which no venue in the set prices with options.

Three independent methods were required, because two of them lied:

1. **Block-bootstrap Monte Carlo** on the coin's own hourly bars, carrying
   intra-bar highs (closes-only monitoring materially underprices a barrier),
   rescaled to a market-implied volatility. Validated against Deribit's RND to
   1.5-2.0c on terminal digitals before any barrier price was trusted.
2. **Horizon-matched volatility** (`volterm.py`). The first pass priced a
   150-day barrier with 30-day realised vol and duly reported 35c edges on XRP
   and DOGE. XRP's 30-day vol is 36.8%; its two-year vol is 84.4%. A
   mean-reverting forecast, calibrated so it reproduces Deribit's implied term
   structure on BTC/ETH (rms 4.8%), killed those.
3. **Direct realised frequency** -- no volatility model at all. Over two years,
   how often did this coin actually move that far in that many days?

Only 13 of 26 candidates survived all three.

## The result

**Kalshi's BNB annual one-touch ladder (KXBNBMAXY) is stale.** BNB needs +10%
at any point in five months; the market prices 18.5%, the model says 59%, and
it has happened 71% of the time -- 0.750 in *both* halves of the sample, with
BNB's total two-year drift only +11.4%, so it is not a bull-run artifact.

It is confirmed without any model by the calendar relation: touching B by
31 Aug implies touching B by 31 Dec, yet `KXBNBMAXMON @650` bids **0.32**
while `KXBNBMAXY @650` offers **0.19**. Buying the far and selling the near
locks the difference in every state of the world.

**And the whole ladder holds $817 of buyable depth.** Five locked pairs total
$61 of risk-free profit on $743 of capital (8.3% to December, ~20% annualised).

That is the honest shape of this universe: the venues are efficient wherever
there is size, and mispriced only where there is not. Kalshi's BTC digital
ladder trades 2 million contracts a day at a one-cent spread; the ladder with
a 40-point mispricing trades a few hundred contracts a week.

---

# The bot: `bot/arb_bot.py`

Automated scan → size → execute → mark → exit → recycle on the locked-pair
strategy. Paper by default; live needs `BOT_MODE=live` and `KALSHI_LIVE=yes`.

```
cd bot
python3 arb_bot.py --capital 1000 --cycles 1          # one scan, paper
python3 arb_bot.py --capital 1000 --cycles 0 --sleep 300   # run continuously
```

State persists in `bot/positions.json`, so it survives restarts and never
double-books depth it has already committed.

## Mechanics it enforces

| | |
|---|---|
| **Entry** | far window must strictly contain the near window; barrier ordering must hold; both legs walked level-by-level while the pair still costs under $1 after fees |
| **Sizing** | capped by book depth, free capital, and `--max-per-pair`; depth consumed by open positions *and* by earlier pairs in the same cycle is deducted |
| **Execution** | fill-or-kill both legs; if the far leg fails the near leg is unwound immediately — a naked short barrier is the only genuinely dangerous state |
| **Marking** | every position is re-marked against the live book each cycle |
| **Exit** | closes early only when unwinding banks ≥80% of the locked edge; otherwise holds |
| **Recycling** | freed capital is redeployed to the highest edge-per-contract pair available |

## On selling before expiry

Contracts do trade continuously, and the bot marks and can exit at any time.
But on these books the round trip is expensive — measured live:

```
KXBNBMAXMON-65000  NO   buy 0.689 / sell 0.609   round-trip  8.0c
KXBNBMAXY-65000    YES  buy 0.180 / sell 0.123   round-trip  5.7c
                                        total  ~14c per pair
```

against a locked edge of ~10c. **Early exit cannot be manufactured by
crossing the spread** — it only pays if the dislocation genuinely converges.
And the quote history says it does not converge quickly: the edge has been
open in 95-98% of hours since the monthly contract was issued.

So the base case is a hold to the near leg's close (28 days), with early exit
as opportunistic upside. Selling early is a risk-management option here, not a
capital-velocity multiplier.

## Sizing, all three strategies

| | ① Locked pairs | ② BNB annual ladder | ③ Polymarket ETH MM |
|---|---|---|---|
| Risk | none at settlement | full capital | inventory |
| Bankroll | **$573** | $603 | ~$92,000 |
| Min ticket | $0.90 | $0.17 | $1 |
| Max deployable | $573 (620 pairs) | $603 | flow-capped |
| Min profit | **+$47.36 guaranteed** | — | — |
| Expected | $47–$667 | +$587 | ~$16/day |
| Worst case | +$47.36 | **−$603 (total)** | gap risk |
| Horizon | 28d guaranteed | to 31 Dec | continuous |
| Annualised (min) | **184%** | n/a | n/a |

---

# Why the arbitrage bot is Kalshi-only

Two reasons, one legitimate and one that had to be tested rather than assumed.

**Execution.** Kalshi orders are REST + RSA request signing. Polymarket needs an
EIP-712 signed CLOB order and a funded Polygon wallet. Only the Kalshi adapter
is built (`bot/venues.py`). Polymarket *is* targeted by the other bot,
`bot/poly_mm.py`, which quotes its ETH range ladders.

**There is nothing there for this strategy.** `src/poly_nest.py` runs the same
nesting scan on Polymarket, which lists exactly the same structure:

```
what price will X hit on August 4     1 day
what price will X hit August 3-9      7 days
what price will X hit in August      31 days
```

175 legs across BTC/ETH/SOL/XRP, three nested windows each, same Binance feed:
**zero locked pairs.** Polymarket's nested windows are internally consistent.

(The first run of that scan reported 25 arbitrages, all XRP. They were a
parsing bug -- the strike regex dropped decimals, so 1.10 / 1.20 / 0.90 all
collapsed to "1" and produced fake nesting. Fixed; the count went to zero.)

**Cross-venue is also flat.** Kalshi's August one-touch versus Polymarket's,
barrier by barrier, is -0.2c to -4.2c -- Polymarket sits consistently just
above Kalshi, which is what the USDT peg predicts. And the windows do not
actually nest (Polymarket's August window opens ~1 hour later than Kalshi's),
so the implication would not hold even if the prices were favourable.

---

# Both venues, head to head

`src/polyflow.py` measures Polymarket's actual trade tape; `src/mmflow.py`
joins it to live quotes. Two errors in the earlier pass had to be fixed first:

1. **Flow was understated ~450x.** Gamma's `volumeNum` is roughly one day's
   volume, not lifetime. Real flow across Polymarket's crypto ladders is
   **2.65M contracts / $1.17M notional per day**.
2. **Spread was overstated ~20x.** Taking a median across a ladder counts the
   wing buckets, which quote 9-39c wide on contracts worth 2-5c and never
   trade. Weighting each contract's spread by the volume that actually crossed
   on it is the honest statistic.

## Flow-weighted market-making economics (measured, 24h)

| family | contracts/day | median spr | **flow-weighted** | net/ct | $/month |
|---|---|---|---|---|---|
| ETH range | 27,071 | 78.5c | **3.9c** | +0.36c | **+$440** |
| BTC range | 102,513 | 1.6c | 0.9c | −0.69c | −$3,180 |
| ETH above | 245,227 | 2.0c | 0.8c | −0.71c | −$7,849 |
| BTC above | 993,832 | 1.0c | **0.3c** | −0.90c | −$40,069 |
| BTC hit | 711,263 | 0.6c | 0.6c | −0.78c | −$24,869 |

Where the volume is, the flow-weighted spread is **0.3c**. Quoting it loses
money on every fill. The same inverse relationship as Kalshi, only sharper:
the busiest book on either venue is the least profitable to quote.

## The answer

| | Kalshi locked pairs | Polymarket ETH-range MM |
|---|---|---|
| Capital | $573 | ~$1,400 |
| Profit | **$47/mo floor, $252 expected** | ~$440/mo |
| Risk | **none at settlement** | inventory; a gap is a real loss |
| Confidence | arithmetic on live quotes | assumes a 15% fill share, unmeasured |
| Capacity | hard-capped by depth | capped by flow |

Polymarket wins on raw dollars (~$440/mo vs ~$47/mo floor), on ~2.4x the
capital, with real risk and a fill rate that is still an assumption. Kalshi
wins on certainty and on return per dollar of *risked* capital, since its floor
cannot be negative.

Running both costs ~$2,000 and is the sensible answer: they are uncorrelated,
neither is capacity-constrained by the other, and together they are roughly
$490/month against ~$2,000 deployed.
