"""Flatten every venue's snapshot into CSVs for inspection."""
import csv
import datetime as dt
import os

from common import load

OUT = "/home/user/D2/out"


def w(name, rows, cols):
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{name}.csv"
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"{p}: {len(rows)} rows")
    return p


def main():
    kal, poly, der = load("kalshi"), load("polymarket"), load("deribit")

    # ---- Kalshi
    rows = []
    for series, evs in kal["series"].items():
        asset = "ETH" if "ETH" in series else "BTC"
        kind = "range" if series in ("KXBTC", "KXETH") else "above"
        for ev, blob in evs.items():
            for m in blob["markets"]:
                if m["yes_ask"] is None or not (0.0 < m["yes_ask"] <= 1.0):
                    continue
                bk = kal.get("books", {}).get(m["ticker"], {})
                rows.append({
                    "venue": "kalshi", "asset": asset, "ladder": kind,
                    "event": ev, "ticker": m["ticker"], "contract": m["subtitle"],
                    "settle_utc": m["close_time"], "strike_type": m["strike_type"],
                    "floor": m["floor_strike"], "cap": m["cap_strike"],
                    "yes_bid": m["yes_bid"], "yes_ask": m["yes_ask"],
                    "no_bid": m["no_bid"], "no_ask": m["no_ask"],
                    "last": m["last"], "volume": m["volume"],
                    "open_interest": m["open_interest"],
                    "bid_depth": sum(s for _, s in bk.get("yes", [])),
                    "ask_depth": sum(s for _, s in bk.get("no", [])),
                })
    w("kalshi_markets", sorted(rows, key=lambda r: (r["asset"], r["event"], r["floor"] or 0)),
      list(rows[0].keys()))

    # ---- Polymarket
    rows = []
    for slug, e in poly["events"].items():
        for m in e["markets"]:
            bk = poly.get("books", {}).get(m["slug"], {})
            rows.append({
                "venue": "polymarket", "asset": e["asset"], "ladder": e["kind"],
                "event": slug, "ticker": m["slug"], "contract": m["group_title"],
                "settle_utc": e["end_date"],
                "best_bid": m["best_bid"], "best_ask": m["best_ask"],
                "last": m["last"], "volume": m["volume"], "liquidity": m["liquidity"],
                "bid_depth": sum(s for _, s in bk.get("bids", [])),
                "ask_depth": sum(s for _, s in bk.get("asks", [])),
                "yes_token": m["yes_token"],
            })
    w("polymarket_markets", rows, list(rows[0].keys()))

    # ---- Deribit
    rows = []
    for ccy, blob in der["ccy"].items():
        idx = blob["index"]["index_price"]
        for kind in ("option", "future"):
            for s in blob["book_summary"][kind]:
                n = s["instrument_name"]
                d = blob["depth"].get(n, {})
                parts = n.split("-")
                rows.append({
                    "venue": "deribit", "asset": ccy, "kind": kind,
                    "instrument": n,
                    "expiry": parts[1] if len(parts) > 1 else "",
                    "strike": parts[2] if len(parts) > 3 else "",
                    "cp": parts[3] if len(parts) > 3 else "",
                    "bid_coin": s.get("bid_price"), "ask_coin": s.get("ask_price"),
                    "mark_coin": s.get("mark_price"),
                    "bid_usd": (s.get("bid_price") or 0) * idx if kind == "option" else s.get("bid_price"),
                    "ask_usd": (s.get("ask_price") or 0) * idx if kind == "option" else s.get("ask_price"),
                    "mark_iv": s.get("mark_iv"),
                    "underlying": s.get("underlying_price") or idx,
                    "open_interest": s.get("open_interest"),
                    "volume": s.get("volume"),
                    "delta": (d.get("greeks") or {}).get("delta"),
                    "vega": (d.get("greeks") or {}).get("vega"),
                    "gamma": (d.get("greeks") or {}).get("gamma"),
                    "theta": (d.get("greeks") or {}).get("theta"),
                    "funding_8h": d.get("funding_8h"),
                })
    w("deribit_instruments", rows, list(rows[0].keys()))

    print(f"\nsnapshot: {dt.datetime.utcfromtimestamp(kal['fetched_ms']/1000)}Z")


if __name__ == "__main__":
    main()
