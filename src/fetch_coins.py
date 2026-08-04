"""Hourly history for every coin that has a live Kalshi one-touch market."""
import time

from common import http_json, save, now_ms

BINANCE = "https://data-api.binance.vision/api/v3"
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "BCH", "BNB",
         "LINK", "LTC", "DOT", "SHIB", "PEPE", "TRX", "TON", "SUI", "APT"]


def paged(symbol, interval="1h", bars=17520):
    out, end = [], now_ms()
    while len(out) < bars:
        try:
            d = http_json(f"{BINANCE}/klines",
                          {"symbol": symbol, "interval": interval,
                           "endTime": end, "limit": 1000})
        except Exception:                                # noqa: BLE001
            break
        if not d:
            break
        rows = [{"open_ms": r[0], "o": float(r[1]), "h": float(r[2]),
                 "l": float(r[3]), "c": float(r[4]), "v": float(r[5])} for r in d]
        out = rows + out
        end = rows[0]["open_ms"] - 1
        if len(d) < 1000:
            break
        time.sleep(0.12)
    return out[-bars:]


def main():
    snap = {"fetched_ms": now_ms(), "bars": {}, "spot": {}}
    for c in COINS:
        sym = f"{c}USDT"
        b = paged(sym)
        if not b:
            print(f"{c}: no data")
            continue
        snap["bars"][c] = b
        try:
            snap["spot"][c] = float(http_json(f"{BINANCE}/ticker/price",
                                              {"symbol": sym})["price"])
        except Exception:                                # noqa: BLE001
            snap["spot"][c] = b[-1]["c"]
        yrs = len(b) / 24 / 365.25
        print(f"{c}: {len(b)} bars ({yrs:.2f}y)  spot={snap['spot'][c]:,.6g}")
    save("coins", snap)
    return snap


if __name__ == "__main__":
    main()
