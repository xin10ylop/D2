"""Model-free calendar arbitrage between one-touch markets on the same asset.

A one-touch is monotone in time at a fixed barrier: if the price touches B by
the end of August it has necessarily touched B by the end of December. So for
the same asset and the same barrier,

    P(touch B by the near date)  <=  P(touch B by the far date)

with no model, no volatility assumption and no distributional view whatsoever.
Kalshi lists both monthly (KX*MAXMON / KX*MINMON) and annual (KX*MAXY /
KX*MINY) one-touch ladders on the same coins, frequently at the same strikes,
in two separate order books. Whenever the near contract's bid exceeds the far
contract's ask, buying the far and selling the near is a locked profit.

The same monotonicity gives a second, weaker relation *within* one ladder:
P(touch B) must be non-increasing in B for an up-barrier, non-decreasing for a
down-barrier. Both are checked here at executable prices, after Kalshi fees.
"""
import datetime as dt

from common import http_json, load, save, now_ms

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fee(p):
    return 0.07 * p * (1 - p)


def iso_ms(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def load_ladder(event):
    d = http_json(f"{KALSHI}/markets", {"event_ticker": event, "limit": 200})
    out = []
    for m in d.get("markets", []):
        if m.get("status") != "active":
            continue
        st = m.get("strike_type")
        if st in ("greater", "greater_or_equal"):
            B, side = fnum(m.get("floor_strike")), "up"
        elif st in ("less", "less_or_equal"):
            B, side = fnum(m.get("cap_strike")), "dn"
        else:
            continue
        if B is None:
            continue
        out.append({"ticker": m["ticker"], "barrier": B, "side": side,
                    "bid": fnum(m.get("yes_bid_dollars")) or 0.0,
                    "ask": fnum(m.get("yes_ask_dollars")),
                    "bid_sz": fnum(m.get("yes_bid_size_fp")) or 0,
                    "ask_sz": fnum(m.get("yes_ask_size_fp")) or 0,
                    "oi": fnum(m.get("open_interest_fp")) or 0,
                    "close": m.get("close_time"),
                    "open": m.get("open_time")})
    return out


def main():
    from scan_barrier import series_coin
    uni = load("universe_kalshi")
    touch = [r for r in uni if r["kind"] in ("onetouch_max", "onetouch_min")]

    # index every ladder by (coin, direction), keeping its close time
    books = {}
    for ev in touch:
        coin = series_coin(ev["series"])
        if not coin:
            continue
        lad = load_ladder(ev["event"])
        if not lad:
            continue
        books[ev["event"]] = {"coin": coin, "series": ev["series"], "rows": lad,
                              "close": iso_ms(lad[0]["close"]),
                              "open": iso_ms(lad[0]["open"])}
    print(f"loaded {len(books)} one-touch ladders\n")

    # ---- 1. calendar: same coin, same barrier, same direction, two dates
    found = []
    evs = list(books.items())
    for i, (e1, b1) in enumerate(evs):
        for e2, b2 in evs[i + 1:]:
            if b1["coin"] != b2["coin"]:
                continue
            near, far = (b1, b2) if b1["close"] < b2["close"] else (b2, b1)
            # Both windows must already be running, otherwise the near contract
            # can cover a period the far one does not.
            if near["open"] is None or far["open"] is None:
                continue
            if far["open"] > near["open"] + 3600_000:
                continue
            for rn in near["rows"]:
                for rf in far["rows"]:
                    if rn["side"] != rf["side"] or abs(rn["barrier"] - rf["barrier"]) > 1e-9:
                        continue
                    # sell the near (buy NO at 1-bid), buy the far at its ask
                    if rf["ask"] is None or rn["bid"] <= 0:
                        continue
                    cost = (1.0 - rn["bid"]) + fee(1.0 - rn["bid"]) \
                        + rf["ask"] + fee(rf["ask"])
                    edge = 1.0 - cost
                    if edge > 0.005:
                        found.append({
                            "coin": b1["coin"], "side": rn["side"],
                            "barrier": rn["barrier"], "edge": edge,
                            "near_series": near["series"], "near_ticker": rn["ticker"],
                            "near_bid": rn["bid"], "near_sz": rn["bid_sz"],
                            "far_series": far["series"], "far_ticker": rf["ticker"],
                            "far_ask": rf["ask"], "far_sz": rf["ask_sz"],
                            "near_close": near["close"], "far_close": far["close"],
                            "size": min(rn["bid_sz"], rf["ask_sz"]),
                        })

    found.sort(key=lambda r: -r["edge"])
    print(f"=== CALENDAR ARBITRAGE (sell near touch, buy far touch): {len(found)}")
    if found:
        print(f"{'coin':5s} {'d':2s} {'barrier':>11s} {'near':17s} {'bid':>5s} "
              f"{'far':17s} {'ask':>5s} {'edge':>7s} {'size':>8s} {'profit':>9s}")
        tot = 0.0
        for r in found[:30]:
            p = r["edge"] * r["size"]
            tot += p
            print(f"{r['coin']:5s} {r['side']:2s} {r['barrier']:11,.5g} "
                  f"{r['near_series'][:17]:17s} {r['near_bid']:5.2f} "
                  f"{r['far_series'][:17]:17s} {r['far_ask']:5.2f} "
                  f"{r['edge']:+7.3f} {r['size']:8,.0f} ${p:8,.0f}")
        print(f"\n  total locked profit at displayed size: "
              f"${sum(r['edge']*r['size'] for r in found):,.0f}")
    else:
        print("   none")

    # ---- 2. monotonicity in the barrier, inside a single ladder
    mono = []
    for e, b in books.items():
        rows = [r for r in b["rows"] if r["ask"] is not None]
        ups = sorted([r for r in rows if r["side"] == "up"], key=lambda r: r["barrier"])
        dns = sorted([r for r in rows if r["side"] == "dn"], key=lambda r: -r["barrier"])
        for lad in (ups, dns):
            for i in range(len(lad) - 1):
                lo_b, hi_b = lad[i], lad[i + 1]     # hi_b is strictly harder to touch
                if hi_b["ask"] is None or lo_b["bid"] <= 0:
                    continue
                cost = (1.0 - lo_b["bid"]) + fee(1.0 - lo_b["bid"]) \
                    + hi_b["ask"] + fee(hi_b["ask"])
                edge = 1.0 - cost
                if edge > 0.005:
                    mono.append({"event": e, "coin": b["coin"], "side": lo_b["side"],
                                 "easy": lo_b["barrier"], "hard": hi_b["barrier"],
                                 "easy_bid": lo_b["bid"], "hard_ask": hi_b["ask"],
                                 "edge": edge,
                                 "size": min(lo_b["bid_sz"], hi_b["ask_sz"])})
    mono.sort(key=lambda r: -r["edge"])
    print(f"\n=== BARRIER MONOTONICITY VIOLATIONS inside one ladder: {len(mono)}")
    if mono:
        print(f"{'event':30s} {'d':2s} {'easier':>11s} {'bid':>5s} {'harder':>11s} "
              f"{'ask':>5s} {'edge':>7s} {'size':>8s}")
        for r in mono[:25]:
            print(f"{r['event'][:30]:30s} {r['side']:2s} {r['easy']:11,.5g} "
                  f"{r['easy_bid']:5.2f} {r['hard']:11,.5g} {r['hard_ask']:5.2f} "
                  f"{r['edge']:+7.3f} {r['size']:8,.0f}")
    else:
        print("   none")

    save("calendar_arb", {"calendar": found, "monotonicity": mono})
    return found, mono


if __name__ == "__main__":
    main()
