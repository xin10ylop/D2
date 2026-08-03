"""Deep hourly history for a trustworthy intraday variance profile.

1000 bars gives ~40 observations per hour-of-day, whose variance estimate has
~22% sampling error -- not enough to bet on. This pages back two years so each
hour-of-day bucket holds ~700 observations, and pulls the same window for ETH
and for the USD venues so the profile and the basis are estimated on a common
sample.
"""
import time

from common import http_json, save, now_ms

BINANCE = "https://data-api.binance.vision/api/v3"


def paged_klines(symbol, interval="1h", bars=17520):
    """Walk backwards from now in 1000-bar pages."""
    out = []
    end = now_ms()
    step_ms = {"1h": 3600_000, "5m": 300_000, "1d": 86400_000}[interval]
    while len(out) < bars:
        d = http_json(f"{BINANCE}/klines",
                      {"symbol": symbol, "interval": interval,
                       "endTime": end, "limit": 1000})
        if not d:
            break
        rows = [{"open_ms": r[0], "o": float(r[1]), "h": float(r[2]),
                 "l": float(r[3]), "c": float(r[4]), "v": float(r[5]),
                 "n": r[8], "qv": float(r[7])} for r in d]
        out = rows + out
        end = rows[0]["open_ms"] - 1
        if len(d) < 1000:
            break
        time.sleep(0.15)
    return out[-bars:]


def coinbase_hist(product, hours=8000):
    """Coinbase candles cap at 300 rows per call, so page by time window."""
    out, end = [], now_ms() // 1000
    while len(out) < hours:
        start = end - 300 * 3600
        try:
            d = http_json(f"https://api.exchange.coinbase.com/products/{product}/candles",
                          {"granularity": 3600, "start": start, "end": end})
        except Exception:  # noqa: BLE001
            break
        if not d:
            break
        rows = [{"open_ms": r[0] * 1000, "l": r[1], "h": r[2],
                 "o": r[3], "c": r[4], "v": r[5]} for r in d]
        out = rows + out
        end = start - 1
        time.sleep(0.2)
        if len(d) < 100:
            break
    seen, ded = set(), []
    for r in sorted(out, key=lambda x: x["open_ms"]):
        if r["open_ms"] not in seen:
            seen.add(r["open_ms"])
            ded.append(r)
    return ded


def main():
    snap = {"fetched_ms": now_ms(), "bars": {}}
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = paged_klines(sym, "1h", 17520)
        snap["bars"][f"{sym}_1h"] = b
        print(f"{sym} 1h: {len(b)} bars, from "
              f"{time.strftime('%Y-%m-%d', time.gmtime(b[0]['open_ms']/1000))}")
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = paged_klines(sym, "5m", 9000)
        snap["bars"][f"{sym}_5m"] = b
        print(f"{sym} 5m: {len(b)} bars")
    for prod in ("BTC-USD", "ETH-USD"):
        c = coinbase_hist(prod, 8000)
        snap["bars"][f"coinbase_{prod}_1h"] = c
        print(f"coinbase {prod}: {len(c)} bars, from "
              f"{time.strftime('%Y-%m-%d', time.gmtime(c[0]['open_ms']/1000)) if c else '-'}")
    save("hist", snap)
    return snap


if __name__ == "__main__":
    main()
