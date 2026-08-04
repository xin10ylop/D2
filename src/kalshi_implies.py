"""Cross-family logical implication on Kalshi: terminal digitals => one-touch.

The locked-pair scanner only compared one-touch contracts with each other. But
Kalshi lists terminal digitals on the same coins over the same period, and a
terminal outcome forces a path outcome:

    KXBTCD-...-T64999.99  "BRTI above 65,000 at 5pm on Aug 7"
        implies
    KXBTCMAXMON-...-65000 "BRTI ever above 65,000 during August"

because being above the level at 5pm on the 7th is one way of having touched
it during the month. Same for the range ladders, and mirrored for the minimum
series on the downside.

This is the same free-money test as before -- if the implied contract can be
bought for less than the implying one can be sold for, the pair is locked --
applied to a family pairing that the first scanner never considered.
"""
import datetime as dt
import itertools
import math

from common import http_json, load, pmap, save, now_ms

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fee(p, n=1):
    return math.ceil(0.07 * n * p * (1 - p) * 100) / 100


def iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000 if s else None


def book(t):
    d = http_json(f"{KALSHI}/markets/{t}/orderbook", {"depth": 100})
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    return {"yes": [[float(p), float(q)] for p, q in (ob.get("yes_dollars") or [])],
            "no": [[float(p), float(q)] for p, q in (ob.get("no_dollars") or [])]}


def buy_levels(bk, side):
    lad = bk.get("no" if side == "YES" else "yes", [])
    return sorted([(round(1.0 - p, 2), q) for p, q in lad if 0 < 1 - p < 1 and q > 0],
                  key=lambda x: x[0])


def claims():
    """Every live crypto claim on Kalshi, as a predicate."""
    from scan_barrier import series_coin
    ser = http_json(f"{KALSHI}/series", {"category": "Crypto", "limit": 200})
    names = [s["ticker"] for s in ser.get("series", [])]

    def opens(s):
        try:
            return http_json(f"{KALSHI}/events",
                             {"series_ticker": s, "status": "open",
                              "limit": 60}).get("events", [])
        except Exception:                                # noqa: BLE001
            return []

    evs = []
    for s, es in pmap(opens, names, workers=12):
        for e in es or []:
            evs.append((s, e["event_ticker"]))

    def mk(pair):
        try:
            return http_json(f"{KALSHI}/markets",
                             {"event_ticker": pair[1], "limit": 300}).get("markets", [])
        except Exception:                                # noqa: BLE001
            return []

    out = []
    for (s, ev), ms_ in pmap(mk, evs, workers=12):
        coin = series_coin(s)
        if not coin:
            continue
        is_touch = any(k in s.upper() for k in ("MAX", "MIN"))
        for m in ms_ or []:
            if m.get("status") != "active":
                continue
            st = m.get("strike_type")
            bid = fnum(m.get("yes_bid_dollars")) or 0.0
            ask = fnum(m.get("yes_ask_dollars"))
            base = {"coin": coin, "series": s, "event": ev, "ticker": m["ticker"],
                    "bid": bid, "ask": ask,
                    "open": iso(m.get("open_time")), "close": iso(m.get("close_time")),
                    "sub": m.get("subtitle")}
            if st in ("greater", "greater_or_equal"):
                K = fnum(m.get("floor_strike"))
                if K is None:
                    continue
                out.append({**base, "kind": "touch_up" if is_touch else "above", "K": K})
            elif st in ("less", "less_or_equal"):
                K = fnum(m.get("cap_strike"))
                if K is None:
                    continue
                out.append({**base, "kind": "touch_dn" if is_touch else "below", "K": K})
            elif st == "between" and not is_touch:
                lo, hi = fnum(m.get("floor_strike")), fnum(m.get("cap_strike"))
                if lo is None or hi is None:
                    continue
                out.append({**base, "kind": "range", "lo": lo, "hi": hi})
    return out


def implications(cl):
    """(A, B) with A => B. Per coin only."""
    out = []
    for coin in sorted({c["coin"] for c in cl}):
        cs = [c for c in cl if c["coin"] == coin]
        tu = [c for c in cs if c["kind"] == "touch_up"]
        td = [c for c in cs if c["kind"] == "touch_dn"]
        ab = [c for c in cs if c["kind"] == "above"]
        be = [c for c in cs if c["kind"] == "below"]
        rg = [c for c in cs if c["kind"] == "range"]

        def within(t, b):
            return b["open"] is not None and b["open"] - 3600_000 <= t <= b["close"] + 1000

        for a in ab:                       # above K at close => touched K' <= K
            for b in tu:
                if b["K"] <= a["K"] + 1e-9 and within(a["close"], b):
                    out.append((a, b, "above => touch_up"))
        for a in be:                       # below K at close => touched K' >= K
            for b in td:
                if b["K"] >= a["K"] - 1e-9 and within(a["close"], b):
                    out.append((a, b, "below => touch_dn"))
        for a in rg:                       # in [lo,hi] => touched up at lo, down at hi
            for b in tu:
                if b["K"] <= a["lo"] + 1e-9 and within(a["close"], b):
                    out.append((a, b, "range => touch_up"))
            for b in td:
                if b["K"] >= a["hi"] - 1e-9 and within(a["close"], b):
                    out.append((a, b, "range => touch_dn"))
        for a, b in itertools.permutations(tu, 2):
            if b["K"] <= a["K"] + 1e-9 and b["open"] is not None and a["open"] is not None \
                    and b["open"] <= a["open"] + 3600_000 and b["close"] >= a["close"] - 1000 \
                    and a["ticker"] != b["ticker"]:
                out.append((a, b, "touch_up nesting"))
        for a, b in itertools.permutations(td, 2):
            if b["K"] >= a["K"] - 1e-9 and b["open"] is not None and a["open"] is not None \
                    and b["open"] <= a["open"] + 3600_000 and b["close"] >= a["close"] - 1000 \
                    and a["ticker"] != b["ticker"]:
                out.append((a, b, "touch_dn nesting"))
    return out


def main(min_edge=0.005):
    cl = claims()
    print(f"live Kalshi crypto claims: {len(cl)}")
    kinds = {}
    for c in cl:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    imps = implications(cl)
    print(f"logical implications: {len(imps):,}")

    hits = []
    for a, b, why in imps:
        if b["ask"] is None or a["bid"] <= 0:
            continue
        cost = (1 - a["bid"]) + fee(1 - a["bid"]) + b["ask"] + fee(b["ask"])
        if cost < 1.0 - min_edge:
            hits.append({"A": a, "B": b, "why": why, "edge": 1 - cost})
    hits.sort(key=lambda r: -r["edge"])
    print(f"violations at top of book: {len(hits)}\n")
    if not hits:
        print("  none")
        save("kalshi_implies", [])
        return []

    need = sorted({t for h in hits[:80] for t in (h["A"]["ticker"], h["B"]["ticker"])})
    books = {}
    for t, bk in pmap(book, need, workers=12):
        if isinstance(bk, dict) and "__error__" not in bk:
            books[t] = bk

    print(f"{'why':20s} {'coin':5s} {'SELL A':34s} {'bid':>5s} {'BUY B':34s} "
          f"{'ask':>5s} {'edge':>6s} {'qty':>6s} {'profit':>8s}")
    out = []
    for h in hits[:40]:
        a, b = h["A"], h["B"]
        la, lb = buy_levels(books.get(a["ticker"], {}), "NO"), \
            buy_levels(books.get(b["ticker"], {}), "YES")
        ia = ib = 0
        qty = cost = 0.0
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
        if qty <= 0:
            continue
        f = fee(cost / qty / 2, qty) * 2
        profit = qty - cost - f
        if profit <= 0:
            continue
        print(f"{h['why']:20s} {a['coin']:5s} "
              f"{(a['series'] + ' ' + str(a['sub']))[:34]:34s} {a['bid']:5.2f} "
              f"{(b['series'] + ' ' + str(b['sub']))[:34]:34s} {b['ask']:5.2f} "
              f"{h['edge']:+6.3f} {qty:6,.0f} ${profit:7,.2f}")
        out.append({"why": h["why"], "coin": a["coin"], "sell": a["ticker"],
                    "buy": b["ticker"], "qty": qty, "capital": cost + f,
                    "profit": profit})
    if out:
        print(f"\n  TOTAL: ${sum(r['profit'] for r in out):,.2f} locked on "
              f"${sum(r['capital'] for r in out):,.2f}")
    save("kalshi_implies", out)
    return out


if __name__ == "__main__":
    main()
