"""Pull Polymarket's BTC/ETH daily digital + range ladders with CLOB depth.

Two question families, mirroring the Kalshi pair:
  bitcoin-above-on-august-N-2026   -- "above X" digital ladder
  bitcoin-price-on-august-N-2026   -- "in bucket [a,b)" range ladder
(and the ethereum-* equivalents)

All of them settle on the Binance BTC/USDT (resp. ETH/USDT) 1-minute candle
CLOSE at 12:00 ET = 16:00 UTC on the named date.
"""
import json
import sys

from common import http_json, pmap, save, now_ms

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

FAMILIES = [
    ("BTC", "above", "bitcoin-above-on-august-{d}-2026"),
    ("BTC", "range", "bitcoin-price-on-august-{d}-2026"),
    ("ETH", "above", "ethereum-above-on-august-{d}-2026"),
    ("ETH", "range", "ethereum-price-on-august-{d}-2026"),
]
DAYS = list(range(3, 32))


def jparse(s, default=None):
    if isinstance(s, (list, dict)):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


def get_event(slug):
    d = http_json(f"{GAMMA}/events", {"slug": slug})
    return d[0] if d else None


def get_book(token_id):
    return http_json(f"{CLOB}/book", {"token_id": token_id})


def norm_market(m):
    prices = jparse(m.get("outcomePrices"), []) or []
    toks = jparse(m.get("clobTokenIds"), []) or []
    return {
        "slug": m.get("slug"),
        "question": m.get("question"),
        "group_title": m.get("groupItemTitle"),
        "end_date": m.get("endDate"),
        "outcomes": jparse(m.get("outcomes"), []),
        "outcome_prices": [float(x) for x in prices] if prices else [],
        "yes_token": toks[0] if toks else None,
        "no_token": toks[1] if len(toks) > 1 else None,
        "best_bid": m.get("bestBid"),
        "best_ask": m.get("bestAsk"),
        "spread": m.get("spread"),
        "last": m.get("lastTradePrice"),
        "volume": m.get("volumeNum"),
        "liquidity": m.get("liquidityNum"),
        "tick": m.get("orderPriceMinTickSize"),
        "neg_risk": m.get("negRisk"),
        "accepting": m.get("acceptingOrders"),
        "closed": m.get("closed"),
    }


def main(with_books=True):
    snap = {"fetched_ms": now_ms(), "venue": "polymarket", "events": {}}
    slugs = [(a, k, t.format(d=d), d) for a, k, t in FAMILIES for d in DAYS]
    got = pmap(lambda s: get_event(s[2]), slugs, workers=10)

    tokens = []
    for (asset, kind, slug, day), ev in got:
        if not isinstance(ev, dict) or not ev:
            continue
        mkts = [norm_market(m) for m in ev.get("markets", [])]
        mkts = [m for m in mkts if not m["closed"]]
        if not mkts:
            continue
        snap["events"][slug] = {
            "asset": asset,
            "kind": kind,
            "day": day,
            "title": ev.get("title"),
            "end_date": ev.get("endDate"),
            "neg_risk": ev.get("negRisk") or ev.get("enableNegRisk"),
            "volume": ev.get("volume"),
            "liquidity": ev.get("liquidity"),
            "description": ev.get("description"),
            "markets": mkts,
        }
        for m in mkts:
            if m["yes_token"]:
                tokens.append((slug, m["slug"], m["yes_token"]))

    print(f"events: {len(snap['events'])}  markets: "
          f"{sum(len(e['markets']) for e in snap['events'].values())}")

    if with_books:
        print(f"fetching {len(tokens)} CLOB books...")
        books = {}
        for (slug, mslug, tok), b in pmap(lambda t: get_book(t[2]), tokens, workers=12):
            if isinstance(b, dict) and "__error__" not in b:
                books[mslug] = {
                    "bids": [[float(x["price"]), float(x["size"])] for x in b.get("bids", [])],
                    "asks": [[float(x["price"]), float(x["size"])] for x in b.get("asks", [])],
                }
        snap["books"] = books
        print(f"books: {len(books)}")

    save("polymarket", snap)
    return snap


if __name__ == "__main__":
    main(with_books="--nobooks" not in sys.argv)
