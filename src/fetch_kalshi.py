"""Pull every open Kalshi BTC/ETH hourly-and-daily market with full order books.

Series covered:
  KXBTCD / KXETHD  -- "price at or above X" digital ladders
  KXBTC  / KXETH   -- "price in range [a,b)" bucket ladders

Both settle on the 60-second average of the CF Benchmarks Real-Time Index
(BRTI / ETHRTI) immediately preceding the stated hour.
"""
import sys

from common import http_json, pmap, save, now_ms

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXBTCD", "KXBTC", "KXETHD", "KXETH"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_events(series):
    """Only live + not-yet-opened events; the full history runs to thousands."""
    out, cursor = [], None
    while True:
        p = {"series_ticker": series, "status": "open,unopened", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        d = http_json(f"{BASE}/events", p)
        out += d.get("events", [])
        cursor = d.get("cursor")
        if not cursor:
            return out


def get_markets(event_ticker):
    out, cursor = [], None
    while True:
        p = {"event_ticker": event_ticker, "limit": 500}
        if cursor:
            p["cursor"] = cursor
        d = http_json(f"{BASE}/markets", p)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor:
            return out


def get_book(ticker):
    d = http_json(f"{BASE}/markets/{ticker}/orderbook", {"depth": 100})
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    return {
        "yes": [[float(p), float(q)] for p, q in (ob.get("yes_dollars") or ob.get("yes") or [])],
        "no": [[float(p), float(q)] for p, q in (ob.get("no_dollars") or ob.get("no") or [])],
    }


def norm_market(m):
    """Flatten a Kalshi market into the fields the analytics layer needs.

    Kalshi quotes YES and NO as separate books. A NO bid at price p is
    economically a YES offer at 1-p, so the executable YES spread is
    [yes_bid, yes_ask] where each side may come from either book.
    """
    return {
        "ticker": m["ticker"],
        "event": m["event_ticker"],
        "subtitle": m.get("subtitle") or m.get("yes_sub_title"),
        "strike_type": m.get("strike_type"),
        "floor_strike": fnum(m.get("floor_strike")),
        "cap_strike": fnum(m.get("cap_strike")),
        "close_time": m.get("close_time"),
        "expected_expiration_time": m.get("expected_expiration_time"),
        "status": m.get("status"),
        "yes_bid": fnum(m.get("yes_bid_dollars")),
        "yes_ask": fnum(m.get("yes_ask_dollars")),
        "no_bid": fnum(m.get("no_bid_dollars")),
        "no_ask": fnum(m.get("no_ask_dollars")),
        "yes_bid_size": fnum(m.get("yes_bid_size_fp")),
        "yes_ask_size": fnum(m.get("yes_ask_size_fp")),
        "last": fnum(m.get("last_price_dollars")),
        "volume": fnum(m.get("volume_fp")),
        "volume_24h": fnum(m.get("volume_24h_fp")),
        "open_interest": fnum(m.get("open_interest_fp")),
        "rules": m.get("rules_primary"),
    }


def main(with_books=True):
    snap = {"fetched_ms": now_ms(), "venue": "kalshi", "series": {}}
    all_markets = []
    for s in SERIES:
        evs = get_events(s)
        rows = pmap(lambda e: get_markets(e["event_ticker"]), evs, workers=10)
        series_blob = {}
        for e, ms in rows:
            if not isinstance(ms, list):
                print(f"  !! {e['event_ticker']}: {ms}", file=sys.stderr)
                continue
            norm = [norm_market(m) for m in ms]
            if not norm:
                continue
            # A market is only actionable if someone is actually quoting it.
            tradable = any(m["yes_ask"] is not None and m["yes_ask"] < 1.0 for m in norm)
            series_blob[e["event_ticker"]] = {
                "title": e.get("title"),
                "strike_date": e.get("strike_date"),
                "tradable": tradable,
                "markets": norm,
            }
            all_markets += norm
        snap["series"][s] = series_blob
        print(f"{s}: {len(series_blob)} events, "
              f"{sum(len(v['markets']) for v in series_blob.values())} markets")

    if with_books:
        # Books only matter where there is a two-sided quote worth crossing.
        live = [m for m in all_markets if (m["yes_ask"] or 1) < 0.995 and (m["yes_bid"] or 0) > 0.005]
        print(f"fetching {len(live)} order books...")
        books = {}
        for m, b in pmap(lambda x: get_book(x["ticker"]), live, workers=12):
            if isinstance(b, dict) and "__error__" not in b:
                books[m["ticker"]] = b
        snap["books"] = books
        print(f"books: {len(books)}")

    save("kalshi", snap)
    return snap


if __name__ == "__main__":
    main(with_books="--nobooks" not in sys.argv)
