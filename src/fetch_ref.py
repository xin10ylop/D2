"""Reference data the cross-venue comparison cannot be done without.

1. Hourly klines (Binance) -- to estimate the *intraday variance profile*.
   Deribit expires 08:00Z, Polymarket settles 16:00Z, Kalshi 16:00Z and
   21:00Z. Interpolating implied variance linearly in calendar time silently
   assumes crypto vol is flat across the day, which it is not: the US session
   carries far more variance than the Asian small hours. Getting that weighting
   wrong by 10% moves a digital by more than any edge we are hunting.

2. Spot across the BRTI constituent venues and Binance -- to measure the
   index basis. Kalshi settles on CF Benchmarks BRTI (USD pairs); Polymarket
   settles on Binance BTC/USDT. The USDT peg deviation plus venue idiosyncrasy
   is the *only* residual in an otherwise identical pair of contracts, so it
   has to be measured rather than assumed away.
"""
from common import http_json, pmap, save, now_ms

# api.binance.com is geo-blocked from this host (HTTP 451); the public
# data mirror serves the identical global order book.
BINANCE = "https://data-api.binance.vision/api/v3"

SPOT_SOURCES = {
    # BRTI constituents (all USD)
    "coinbase_BTC": ("https://api.exchange.coinbase.com/products/BTC-USD/ticker", "price"),
    "coinbase_ETH": ("https://api.exchange.coinbase.com/products/ETH-USD/ticker", "price"),
    "bitstamp_BTC": ("https://www.bitstamp.net/api/v2/ticker/btcusd/", "last"),
    "bitstamp_ETH": ("https://www.bitstamp.net/api/v2/ticker/ethusd/", "last"),
    "gemini_BTC": ("https://api.gemini.com/v1/pubticker/btcusd", "last"),
    "gemini_ETH": ("https://api.gemini.com/v1/pubticker/ethusd", "last"),
}


def klines(symbol, interval="1h", limit=1000):
    d = http_json(f"{BINANCE}/klines",
                  {"symbol": symbol, "interval": interval, "limit": limit})
    return [{"open_ms": r[0], "o": float(r[1]), "h": float(r[2]),
             "l": float(r[3]), "c": float(r[4]), "v": float(r[5])} for r in d]


def kraken(pair):
    d = http_json("https://api.kraken.com/0/public/Ticker", {"pair": pair})
    k = list(d["result"].values())[0]
    return float(k["c"][0])


def basis_history():
    """Hourly Binance-USDT vs USD-composite basis, the residual in any
    Kalshi/Polymarket pair. Coinbase and Kraken stand in for the BRTI panel."""
    out = {}
    for sym, cb, kr in (("BTCUSDT", "BTC-USD", "XXBTZUSD"),
                        ("ETHUSDT", "ETH-USD", "XETHZUSD")):
        try:
            out[f"binance_{sym}"] = klines(sym, "1h", 1000)
        except Exception as e:  # noqa: BLE001
            print(f"binance {sym}: {e}")
        try:
            d = http_json(f"https://api.exchange.coinbase.com/products/{cb}/candles",
                          {"granularity": 3600})
            out[f"coinbase_{cb}"] = [{"open_ms": r[0] * 1000, "l": r[1], "h": r[2],
                                      "o": r[3], "c": r[4], "v": r[5]} for r in d]
        except Exception as e:  # noqa: BLE001
            print(f"coinbase {cb}: {e}")
        try:
            d = http_json("https://api.kraken.com/0/public/OHLC",
                          {"pair": kr, "interval": 60})
            rows = list(d["result"].values())[0]
            out[f"kraken_{kr}"] = [{"open_ms": r[0] * 1000, "o": float(r[1]),
                                    "h": float(r[2]), "l": float(r[3]),
                                    "c": float(r[4]), "v": float(r[6])} for r in rows]
        except Exception as e:  # noqa: BLE001
            print(f"kraken {kr}: {e}")
    try:
        out["usdt_usd"] = http_json("https://api.kraken.com/0/public/OHLC",
                                    {"pair": "USDTZUSD", "interval": 60})
    except Exception as e:  # noqa: BLE001
        print(f"usdt: {e}")
    return out


def main():
    snap = {"fetched_ms": now_ms()}

    # --- intraday seasonality + realised vol ------------------------------
    bars = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        for iv, lim in (("1h", 1000), ("5m", 1000)):
            try:
                bars[f"{sym}_{iv}"] = klines(sym, iv, lim)
                print(f"{sym} {iv}: {len(bars[f'{sym}_{iv}'])} bars")
            except Exception as e:  # noqa: BLE001
                print(f"{sym} {iv}: {e}")
    snap["bars"] = bars

    # --- index basis ------------------------------------------------------
    spot = {}
    for name, (url, field) in SPOT_SOURCES.items():
        try:
            d = http_json(url)
            spot[name] = float(d[field] if field in d else d["last"])
        except Exception as e:  # noqa: BLE001
            spot[name] = None
            print(f"{name}: {e}")
    for name, pair in (("kraken_BTC", "XBTUSD"), ("kraken_ETH", "ETHUSD")):
        try:
            spot[name] = kraken(pair)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: {e}")
    for name, sym in (("binance_BTC", "BTCUSDT"), ("binance_ETH", "ETHUSDT"),
                      ("binanceUSDC_BTC", "BTCUSDC"), ("binanceUSDC_ETH", "ETHUSDC")):
        try:
            spot[name] = float(http_json(f"{BINANCE}/ticker/price", {"symbol": sym})["price"])
        except Exception as e:  # noqa: BLE001
            print(f"{name}: {e}")
    # USDT/USD peg reads straight off the stable pairs
    for name, sym in (("USDT_USDC", "USDCUSDT"),):
        try:
            spot[name] = float(http_json(f"{BINANCE}/ticker/price", {"symbol": sym})["price"])
        except Exception as e:  # noqa: BLE001
            print(f"{name}: {e}")
    snap["spot"] = spot
    print("spot:", {k: v for k, v in spot.items() if v})

    snap["basis"] = basis_history()
    print("basis series:", {k: len(v) for k, v in snap["basis"].items()
                            if isinstance(v, list)})

    save("ref", snap)
    return snap


if __name__ == "__main__":
    main()
