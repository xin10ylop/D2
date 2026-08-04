"""Full crypto universe across Kalshi and Polymarket -- every series, every coin.

The first pass only looked at BTC/ETH daily and hourly above/below ladders.
Kalshi lists 272 crypto series and Polymarket lists hundreds of crypto events,
including whole contract families that were never examined:

  one-touch / running maximum  KXBTCMAXD, KXBTCMAXMON, KX*MAXY, KX*MINY
                               "What price will Bitcoin hit in August?"
  first passage / double barrier KXBTC50VS100, KXBTC60VS100, KXBTC75VS100
  very short horizon           KX*15M, Polymarket 5m/15m up-or-down
  many more coins              SOL, XRP, DOGE, ADA, AVAX, BCH, BNB, LINK, ...

Barrier payoffs are the interesting ones: they depend on the distribution of
the running maximum, not the terminal price, and that is exactly the quantity
retail intuition gets wrong.
"""
import re
import sys

from common import http_json, pmap, save, now_ms

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"


def kalshi_series():
    out = []
    cursor = None
    while True:
        p = {"category": "Crypto", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        d = http_json(f"{KALSHI}/series", p)
        out += d.get("series", [])
        cursor = d.get("cursor")
        if not cursor:
            return out


def open_events(series):
    try:
        d = http_json(f"{KALSHI}/events",
                      {"series_ticker": series, "status": "open", "limit": 200})
        return d.get("events", [])
    except Exception:                                    # noqa: BLE001
        return []


def markets(event_ticker):
    try:
        d = http_json(f"{KALSHI}/markets",
                      {"event_ticker": event_ticker, "limit": 500})
        return d.get("markets", [])
    except Exception:                                    # noqa: BLE001
        return []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def classify(series_ticker, title):
    """Group a series by payoff type, because that is what determines pricing."""
    t = (title or "").lower()
    s = series_ticker.upper()
    if "15m" in s.lower() or "up down" in t or "up or down" in t:
        return "updown"
    if "max" in s.lower() or "how high" in t or "one touch" in t or "hit" in t:
        return "onetouch_max"
    if "min" in s.lower() or "how low" in t:
        return "onetouch_min"
    if "vs" in s.lower() and "before" in t:
        return "double_barrier"
    if "range" in t:
        return "range"
    if "above" in t or "below" in t or "directional" in t:
        return "digital"
    return "other"


def main():
    ss = kalshi_series()
    print(f"kalshi crypto series: {len(ss)}")
    rows = pmap(lambda s: open_events(s["ticker"]), ss, workers=12)
    live = [(s, evs) for s, evs in rows if isinstance(evs, list) and evs]
    print(f"series with OPEN events: {len(live)}")

    tasks = [(s, e) for s, evs in live for e in evs]
    print(f"open events: {len(tasks)}  -> pulling markets")
    mres = pmap(lambda t: markets(t[1]["event_ticker"]), tasks, workers=14)

    uni = []
    for (s, e), ms in mres:
        if not isinstance(ms, list):
            continue
        q = [m for m in ms
             if m.get("status") == "active"
             and fnum(m.get("yes_ask_dollars")) is not None
             and 0.0 < fnum(m.get("yes_ask_dollars")) <= 1.0]
        if not q:
            continue
        oi = sum(fnum(m.get("open_interest_fp")) or 0 for m in ms)
        v24 = sum(fnum(m.get("volume_24h_fp")) or 0 for m in ms)
        spreads = [fnum(m["yes_ask_dollars"]) - (fnum(m.get("yes_bid_dollars")) or 0)
                   for m in q
                   if 0.05 <= 0.5 * ((fnum(m.get("yes_bid_dollars")) or 0)
                                     + fnum(m["yes_ask_dollars"])) <= 0.95]
        med = sorted(spreads)[len(spreads) // 2] if spreads else None
        uni.append({
            "venue": "kalshi", "series": s["ticker"], "title": s.get("title"),
            "kind": classify(s["ticker"], s.get("title")),
            "freq": s.get("frequency"),
            "event": e["event_ticker"], "event_title": e.get("title"),
            "n_markets": len(ms), "n_quoted": len(q),
            "oi": oi, "vol24": v24, "med_spread": med,
            "close": q[0].get("close_time"),
        })

    uni.sort(key=lambda r: -(r["vol24"] or 0))
    save("universe_kalshi", uni)

    print(f"\ntradable Kalshi events: {len(uni)}")
    by_kind = {}
    for r in uni:
        k = by_kind.setdefault(r["kind"], {"n": 0, "oi": 0.0, "v": 0.0, "series": set()})
        k["n"] += 1
        k["oi"] += r["oi"]
        k["v"] += r["vol24"]
        k["series"].add(r["series"])
    print(f"\n{'payoff type':16s} {'events':>7s} {'series':>7s} {'open interest':>15s} "
          f"{'24h volume':>14s}")
    for k, v in sorted(by_kind.items(), key=lambda x: -x[1]["v"]):
        print(f"{k:16s} {v['n']:7d} {len(v['series']):7d} {v['oi']:15,.0f} {v['v']:14,.0f}")

    print(f"\ntop 30 tradable events by 24h volume:")
    print(f"{'series':18s} {'kind':15s} {'event':26s} {'quoted':>6s} "
          f"{'spread':>7s} {'OI':>12s} {'vol24':>12s}")
    for r in uni[:30]:
        sp = f"{r['med_spread']*100:5.1f}c" if r["med_spread"] else "    -"
        print(f"{r['series'][:18]:18s} {r['kind']:15s} {r['event'][:26]:26s} "
              f"{r['n_quoted']:6d} {sp:>7s} {r['oi']:12,.0f} {r['vol24']:12,.0f}")
    return uni


if __name__ == "__main__":
    main()
