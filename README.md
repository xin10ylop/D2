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
