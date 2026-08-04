"""General logical-implication arbitrage across every Polymarket crypto market.

The Kalshi money came from one implication: touching a barrier in a short
window forces touching it in a longer window. Polymarket supports a much richer
set, because it lists terminal markets and path markets side by side on the
same asset:

    above(K, D)        S at the close of day D is above K
    range(a, b, D)     S at the close of day D is in [a, b)
    touch_up(K, W)     S reaches K at some point during window W
    touch_dn(K, W)     S falls to K at some point during window W

Every one of these implications is a fact about paths, not a model:

    above(K, D)      => touch_up(K', W)   for K' <= K and D inside W
    range(a, b, D)   => touch_up(K, W)    for K <= a and D inside W
    range(a, b, D)   => touch_dn(K, W)    for K >= b and D inside W
    NOT above(K, D)  => touch_dn(K', W)   for K' >= K and D inside W
    touch_up(K, W1)  => touch_up(K', W2)  for K' <= K and W1 inside W2
    touch_dn(K, W1)  => touch_dn(K', W2)  for K' >= K and W1 inside W2

Whenever A implies B, P(A) <= P(B) must hold. So if B can be bought for less
than A can be sold for, buying NO on A and YES on B pays at least $1 in every
state of the world and costs less than $1 -- locked, with no volatility view.

Polymarket charges no maker or taker fee, so the whole gap is kept.
"""
import datetime as dt
import itertools
import json
import re
import time

from common import http_json, pmap, save

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
NUM = re.compile(r"[\d,]+(?:\.\d+)?")

ASSETS = ("bitcoin", "ethereum", "solana", "xrp")


def event(slug):
    try:
        d = http_json(f"{GAMMA}/events", {"slug": slug})
        return d[0] if d else None
    except Exception:                                    # noqa: BLE001
        return None


def ms(s):
    return (dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000
            if s else None)


def slugs_for(asset, month="august", year=2026, days=range(1, 32)):
    out = {
        "touch": [f"what-price-will-{asset}-hit-in-{month}-{year}"],
        "above": [], "range": [],
    }
    for d in days:
        out["touch"].append(f"what-price-will-{asset}-hit-on-{month}-{d}")
        out["touch"].append(f"what-price-will-{asset}-hit-{month}-{d}-{d+6}-{year}")
        out["above"].append(f"{asset}-above-on-{month}-{d}-{year}")
        out["range"].append(f"{asset}-price-on-{month}-{d}-{year}")
    return out


def nums(t):
    return [float(x.replace(",", "")) for x in NUM.findall(t or "")]


def load_asset(asset):
    """Every live claim on one asset, normalised to a predicate."""
    sl = slugs_for(asset)
    claims = []
    for kind, lst in sl.items():
        for s, ev in pmap(event, lst, workers=10):
            if not isinstance(ev, dict) or not ev:
                continue
            t0 = ms(ev.get("startDate") or ev.get("creationDate"))
            t1 = ms(ev.get("endDate"))
            if t1 is None:
                continue
            for m in ev.get("markets", []):
                if m.get("closed"):
                    continue
                bid, ask = m.get("bestBid"), m.get("bestAsk")
                if bid is None and ask is None:
                    continue
                title = (m.get("groupItemTitle") or "").strip()
                n = nums(title)
                if not n:
                    continue
                toks = m.get("clobTokenIds")
                try:
                    toks = json.loads(toks) if isinstance(toks, str) else toks
                except Exception:                        # noqa: BLE001
                    toks = None
                base = {"asset": asset, "slug": s, "market": m.get("slug"),
                        "title": title, "start": t0, "end": t1,
                        "bid": bid or 0.0, "ask": ask,
                        "yes_token": toks[0] if toks else None,
                        "no_token": toks[1] if toks and len(toks) > 1 else None,
                        "vol": m.get("volumeNum") or 0}
                if kind == "touch":
                    up = title.startswith("↑") or "up" in title.lower()
                    dn = title.startswith("↓") or "down" in title.lower()
                    if not (up or dn):
                        continue
                    claims.append({**base, "kind": "touch_up" if up else "touch_dn",
                                   "K": n[0]})
                elif kind == "above":
                    claims.append({**base, "kind": "above", "K": n[0]})
                else:
                    if title.startswith("<"):
                        lo, hi = 0.0, n[0]
                    elif title.startswith(">"):
                        lo, hi = n[0], float("inf")
                    elif len(n) >= 2:
                        lo, hi = n[0], n[1]
                    else:
                        continue
                    claims.append({**base, "kind": "range", "lo": lo, "hi": hi})
    return claims


def inside(t, w0, w1, tol=3600_000):
    return w0 - tol <= t <= w1 + tol


def implications(claims):
    """All (A, B) with A => B, so P(A) <= P(B) must hold.

    Everything below is per-asset: an implication about Bitcoin's path says
    nothing about Ethereum's, and comparing across assets manufactures
    spectacular nonsense.
    """
    assets = {c["asset"] for c in claims}
    if len(assets) > 1:
        out = []
        for a in sorted(assets):
            out += implications([c for c in claims if c["asset"] == a])
        return out
    out = []
    touch_up = [c for c in claims if c["kind"] == "touch_up"]
    touch_dn = [c for c in claims if c["kind"] == "touch_dn"]
    above = [c for c in claims if c["kind"] == "above"]
    rng = [c for c in claims if c["kind"] == "range"]

    # terminal above => touched up at or below that level, in any covering window
    for a in above:
        for b in touch_up:
            if b["K"] <= a["K"] + 1e-9 and inside(a["end"], b["start"], b["end"]):
                out.append((a, b, "above => touch_up"))
    # terminal in [lo,hi) => touched up at or below lo
    for a in rng:
        for b in touch_up:
            if a["lo"] > 0 and b["K"] <= a["lo"] + 1e-9 and inside(a["end"], b["start"], b["end"]):
                out.append((a, b, "range => touch_up"))
    # terminal in [lo,hi) => touched down at or above hi
    for a in rng:
        for b in touch_dn:
            if a["hi"] < float("inf") and b["K"] >= a["hi"] - 1e-9 \
                    and inside(a["end"], b["start"], b["end"]):
                out.append((a, b, "range => touch_dn"))
    # touch nesting, both directions
    for a, b in itertools.permutations(touch_up, 2):
        if b["K"] <= a["K"] + 1e-9 and b["start"] <= a["start"] + 3600_000 \
                and b["end"] >= a["end"] - 1000 and a["market"] != b["market"]:
            out.append((a, b, "touch_up nesting"))
    for a, b in itertools.permutations(touch_dn, 2):
        if b["K"] >= a["K"] - 1e-9 and b["start"] <= a["start"] + 3600_000 \
                and b["end"] >= a["end"] - 1000 and a["market"] != b["market"]:
            out.append((a, b, "touch_dn nesting"))
    return out


def book_depth(token):
    try:
        b = http_json(f"{CLOB}/book", {"token_id": token})
        return {"bids": sorted([(float(x["price"]), float(x["size"]))
                                for x in b.get("bids", [])], key=lambda z: -z[0]),
                "asks": sorted([(float(x["price"]), float(x["size"]))
                                for x in b.get("asks", [])], key=lambda z: z[0])}
    except Exception:                                    # noqa: BLE001
        return None


def main(min_edge=0.005):
    claims = []
    for a in ASSETS:
        c = load_asset(a)
        print(f"{a}: {len(c)} live claims")
        claims += c
    print(f"total claims: {len(claims)}")

    imps = implications(claims)
    print(f"logical implications found: {len(imps):,}")

    # screen on top of book before spending order-book requests
    hits = []
    for a, b, why in imps:
        if b["ask"] is None or a["bid"] <= 0:
            continue
        cost = (1.0 - a["bid"]) + b["ask"]        # no Polymarket fee
        if cost < 1.0 - min_edge:
            hits.append({"A": a, "B": b, "why": why, "edge": 1.0 - cost})
    hits.sort(key=lambda r: -r["edge"])
    print(f"violations at top of book: {len(hits)}\n")
    if not hits:
        print("  none -- Polymarket is consistent across every implication tested")
        save("implies", [])
        return []

    # size the survivors against real depth
    need = {}
    for h in hits[:60]:
        need[h["A"]["no_token"]] = None
        need[h["B"]["yes_token"]] = None
    books = {}
    for t, bk in pmap(book_depth, [t for t in need if t], workers=10):
        if bk:
            books[t] = bk

    print(f"{'why':22s} {'asset':9s} {'SELL A':38s} {'bid':>5s} "
          f"{'BUY B':38s} {'ask':>5s} {'edge':>6s} {'qty':>7s} {'profit':>8s}")
    out = []
    for h in hits[:40]:
        a, b = h["A"], h["B"]
        ba = books.get(a["no_token"])
        bb = books.get(b["yes_token"])
        qty = 0.0
        cost = 0.0
        if ba and bb:
            # buying NO on A means lifting A's NO asks; buying YES on B lifts B's asks
            la = ba["asks"]
            lb = bb["asks"]
            ia = ib = 0
            while ia < len(la) and ib < len(lb):
                pa, qa = la[ia]
                pb, qb = lb[ib]
                if pa + pb >= 1.0 - min_edge:
                    break
                q = min(qa, qb)
                qty += q
                cost += (pa + pb) * q
                la[ia] = (pa, qa - q)
                lb[ib] = (pb, qb - q)
                if la[ia][1] <= 1e-9:
                    ia += 1
                if lb[ib][1] <= 1e-9:
                    ib += 1
        profit = qty - cost
        print(f"{h['why']:22s} {a['asset'][:9]:9s} "
              f"{(a['slug'][-26:] + ' ' + a['title'])[:38]:38s} {a['bid']:5.3f} "
              f"{(b['slug'][-26:] + ' ' + b['title'])[:38]:38s} {b['ask']:5.3f} "
              f"{h['edge']:+6.3f} {qty:7,.0f} ${profit:7,.2f}")
        out.append({"why": h["why"], "asset": a["asset"], "edge": h["edge"],
                    "sell": a["market"], "sell_title": a["title"], "bid": a["bid"],
                    "buy": b["market"], "buy_title": b["title"], "ask": b["ask"],
                    "qty": qty, "cost": cost, "profit": profit})
    tot = sum(r["profit"] for r in out)
    cap = sum(r["cost"] for r in out)
    print(f"\n  TOTAL at real depth: ${tot:,.2f} locked on ${cap:,.2f} of capital"
          f"{f' ({tot/cap*100:.1f}%)' if cap else ''}")
    save("implies", out)
    return out


if __name__ == "__main__":
    main()
