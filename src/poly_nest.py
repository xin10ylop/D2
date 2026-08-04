"""Nested one-touch scan on Polymarket -- the same logic, the other venue.

Polymarket lists "what price will X hit" over nested horizons:

    hit on August 4        (one day)
    hit August 3-9         (one week)
    hit in August          (one month)

Each shorter window sits strictly inside the longer one, and all of them
resolve on the same Binance feed, so the implication that makes the Kalshi
trade risk-free applies here too:

    touch B during the short window  =>  touch B during the long window

Buying NO on the short leg and YES on the long leg pays at least $1. If the
pair costs less, the difference is locked -- no model, no volatility view.

Polymarket charges no maker or taker fee, so unlike Kalshi the entire gap is
kept; the cost is that execution needs a signed CLOB order and a funded wallet.
"""
import datetime as dt
import itertools
import re

from common import http_json, pmap, save

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
# must capture decimals: XRP/DOGE strikes are 1.10, 0.90, 0.075 ...
NUM = re.compile(r"[\d,]+(?:\.\d+)?")

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def event(slug):
    try:
        d = http_json(f"{GAMMA}/events", {"slug": slug})
        return d[0] if d else None
    except Exception:                                    # noqa: BLE001
        return None


def candidate_slugs(asset="bitcoin", month="august", year=2026, days=range(1, 32)):
    """Daily, weekly and monthly 'what price will X hit' events."""
    out = [f"what-price-will-{asset}-hit-in-{month}-{year}"]
    for d in days:
        out.append(f"what-price-will-{asset}-hit-on-{month}-{d}")
        out.append(f"what-price-will-{asset}-hit-{month}-{d}-{d+6}-{year}")
    return out


def parse_leg(m):
    """(direction, barrier) from a 'up 70,000' / 'down 60,000' style label."""
    t = (m.get("groupItemTitle") or "").strip()
    nums = [float(x.replace(",", "")) for x in NUM.findall(t)]
    if not nums:
        return None
    up = t.startswith("↑") or "up" in t.lower()
    dn = t.startswith("↓") or "down" in t.lower()
    if not up and not dn:
        return None
    return ("up" if up else "dn"), nums[0]


def window(ev):
    """(start_ms, end_ms) of the measurement window."""
    end = ev.get("endDate")
    start = ev.get("startDate") or ev.get("creationDate")
    f = lambda s: (dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000
                   if s else None)                       # noqa: E731
    return f(start), f(end)


def book(token):
    try:
        b = http_json(f"{CLOB}/book", {"token_id": token})
        return {"bids": [[float(x["price"]), float(x["size"])] for x in b.get("bids", [])],
                "asks": [[float(x["price"]), float(x["size"])] for x in b.get("asks", [])]}
    except Exception:                                    # noqa: BLE001
        return None


def main(assets=("bitcoin", "ethereum", "solana", "xrp"), month="august"):
    legs = []
    for asset in assets:
        slugs = candidate_slugs(asset, month)
        for s, ev in pmap(event, slugs, workers=10):
            if not isinstance(ev, dict) or not ev:
                continue
            t0, t1 = window(ev)
            if not t0 or not t1:
                continue
            for m in ev.get("markets", []):
                if m.get("closed"):
                    continue
                p = parse_leg(m)
                if not p:
                    continue
                side, B = p
                toks = m.get("clobTokenIds")
                import json as _j
                try:
                    toks = _j.loads(toks) if isinstance(toks, str) else toks
                except Exception:                        # noqa: BLE001
                    toks = None
                legs.append({
                    "asset": asset, "slug": s, "event": ev.get("title"),
                    "market": m.get("slug"), "side": side, "barrier": B,
                    "start": t0, "end": t1,
                    "bid": m.get("bestBid"), "ask": m.get("bestAsk"),
                    "yes_token": toks[0] if toks else None,
                    "no_token": toks[1] if toks and len(toks) > 1 else None,
                    "vol": m.get("volumeNum") or 0,
                })
    print(f"Polymarket one-touch legs found: {len(legs)}")
    by_asset = {}
    for l in legs:
        by_asset.setdefault(l["asset"], []).append(l)
    for a, ls in by_asset.items():
        wins = sorted({(l["slug"], l["start"], l["end"]) for l in ls},
                      key=lambda x: x[2] - x[1])
        print(f"  {a}: {len(ls)} legs across {len(wins)} windows")
        for s, t0, t1 in wins[:6]:
            print(f"     {s[:52]:52s} {(t1-t0)/86400000:5.1f}d "
                  f"{dt.datetime.utcfromtimestamp(t0/1000):%m-%d} -> "
                  f"{dt.datetime.utcfromtimestamp(t1/1000):%m-%d}")

    # nested pairs: short window inside long, same or easier barrier
    pairs = []
    for a, ls in by_asset.items():
        for x, y in itertools.permutations(ls, 2):
            if x["side"] != y["side"]:
                continue
            if not (y["start"] <= x["start"] + 3600_000 and y["end"] >= x["end"] - 1000):
                continue
            if x["side"] == "up" and y["barrier"] > x["barrier"] + 1e-9:
                continue
            if x["side"] == "dn" and y["barrier"] < x["barrier"] - 1e-9:
                continue
            if x["market"] == y["market"]:
                continue
            bid = x["bid"] or 0.0
            ask = y["ask"]
            if not ask or bid <= 0:
                continue
            cost = (1.0 - bid) + ask          # Polymarket charges no fee
            if cost < 1.0:
                pairs.append({"asset": a, "side": x["side"],
                              "short": x, "long": y,
                              "edge": 1.0 - cost})
    pairs.sort(key=lambda r: -r["edge"])
    print(f"\nnested pairs with a locked edge at top of book: {len(pairs)}")
    if not pairs:
        print("  none -- Polymarket's nested windows are internally consistent")
        save("poly_nest", [])
        return []

    print(f"{'asset':9s} {'d':2s} {'barrier':>10s} {'SELL short window':34s} {'bid':>5s} "
          f"{'BUY long window':34s} {'ask':>5s} {'edge':>7s}")
    out = []
    for r in pairs[:25]:
        x, y = r["short"], r["long"]
        print(f"{r['asset'][:9]:9s} {r['side']:2s} {x['barrier']:10,.0f} "
              f"{x['slug'][-34:]:34s} {x['bid']:5.3f} "
              f"{y['slug'][-34:]:34s} {y['ask']:5.3f} {r['edge']:+7.3f}")
        out.append(r)
    save("poly_nest", [{k: (v if not isinstance(v, dict) else
                            {kk: vv for kk, vv in v.items() if kk != "yes_token"})
                        for k, v in r.items()} for r in out])
    return out


if __name__ == "__main__":
    main()
