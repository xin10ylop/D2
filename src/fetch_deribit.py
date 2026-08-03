"""Pull the full Deribit BTC/ETH options + futures surface.

Public market-data endpoints only (no auth needed, and none of this touches
the account). For each currency we take:
  * every live future + perpetual (mark, basis, funding)
  * every live option (mark IV, greeks, book summary)
  * full order-book depth on the near expiries, where the cross-venue
    relative-value work actually happens
  * the index price + realised-vol history for the reference model
"""
import sys

from common import http_json, pmap, save, now_ms

API = "https://www.deribit.com/api/v2/public"
CCYS = ["BTC", "ETH"]
# Expiries that overlap the Kalshi (through Aug 7) / Polymarket (through Aug 9+)
# horizons -- these get full depth.
NEAR = {"4AUG26", "5AUG26", "6AUG26", "7AUG26", "14AUG26", "21AUG26"}


def rpc(method, **params):
    d = http_json(f"{API}/{method}", params)
    if "result" not in d:
        raise RuntimeError(f"{method}: {d}")
    return d["result"]


def get_depth(name):
    r = rpc("get_order_book", instrument_name=name, depth=20)
    return {
        "bids": r.get("bids", []),
        "asks": r.get("asks", []),
        "mark_price": r.get("mark_price"),
        "mark_iv": r.get("mark_iv"),
        "index_price": r.get("index_price"),
        "underlying_price": r.get("underlying_price"),
        "interest_rate": r.get("interest_rate"),
        "greeks": r.get("greeks"),
        "open_interest": r.get("open_interest"),
        "last_price": r.get("last_price"),
        "settlement_price": r.get("settlement_price"),
        "min_price": r.get("min_price"),
        "max_price": r.get("max_price"),
        "funding_8h": r.get("funding_8h"),
        "current_funding": r.get("current_funding"),
    }


def main(deep=True):
    snap = {"fetched_ms": now_ms(), "venue": "deribit", "ccy": {}}
    for c in CCYS:
        blob = {}
        blob["index"] = rpc("get_index_price", index_name=f"{c.lower()}_usd")
        blob["instruments"] = {
            k: rpc("get_instruments", currency=c, kind=k, expired="false")
            for k in ("option", "future")
        }
        blob["book_summary"] = {
            k: rpc("get_book_summary_by_currency", currency=c, kind=k)
            for k in ("option", "future")
        }
        try:
            blob["hist_vol"] = rpc("get_historical_volatility", currency=c)[-200:]
        except Exception as e:  # noqa: BLE001
            blob["hist_vol"] = {"__error__": repr(e)}

        names = [i["instrument_name"] for i in blob["instruments"]["future"]]
        if deep:
            opts = [i["instrument_name"] for i in blob["instruments"]["option"]
                    if i["instrument_name"].split("-")[1] in NEAR]
            names += opts
        print(f"{c}: {len(blob['instruments']['option'])} options, "
              f"{len(blob['instruments']['future'])} futures, "
              f"deep books on {len(names)}")
        depth = {}
        for n, r in pmap(get_depth, names, workers=12):
            if isinstance(r, dict) and "__error__" not in r:
                depth[n] = r
            else:
                print(f"  !! {n}: {r}", file=sys.stderr)
        blob["depth"] = depth
        snap["ccy"][c] = blob
    save("deribit", snap)
    return snap


if __name__ == "__main__":
    main(deep="--shallow" not in sys.argv)
