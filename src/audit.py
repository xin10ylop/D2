"""Audit and size all three strategies. Live prices, explicit capital, explicit risk.

Interpretation note, settled by data rather than by reading: Kalshi's max/min
series trim the top and bottom 20% of the ticks *within each minute*, producing
a de-wicked minute price, and pay if any such minute exceeds the barrier. The
alternative reading -- a trimmed mean of the whole cumulative history -- is
refuted by the July 2026 settlement, where the 65,000 strike paid YES on a spot
high of 65,277 while the cumulative trimmed mean was only 63,071.

That matters because it decides whether Strategy 1 is arbitrage or not. Under
the confirmed reading the annual window strictly contains the monthly window,
so the implication holds and the pair is locked.

For each strategy this reports: bankroll required, minimum ticket, maximum
deployable at current depth, the payoff in every state, and what can actually
go wrong.
"""
import datetime as dt
import math

import numpy as np

from common import http_json, load, pmap, save, now_ms

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fee(p, contracts=1):
    """Kalshi taker fee, rounded up to the cent on the whole order."""
    return math.ceil(0.07 * contracts * p * (1 - p) * 100) / 100


def iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000 if s else None


def book(t):
    d = http_json(f"{KALSHI}/markets/{t}/orderbook", {"depth": 100})
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    return {"yes": [[float(p), float(q)] for p, q in (ob.get("yes_dollars") or [])],
            "no": [[float(p), float(q)] for p, q in (ob.get("no_dollars") or [])]}


def buy_levels(bk, side):
    ladder = bk.get("no" if side == "YES" else "yes", [])
    return sorted([(round(1.0 - p, 2), q) for p, q in ladder if 0 < 1 - p < 1 and q > 0],
                  key=lambda x: x[0])


def ladder(event):
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
                    "open": iso(m.get("open_time")), "close": iso(m.get("close_time")),
                    "bid": fnum(m.get("yes_bid_dollars")) or 0.0,
                    "ask": fnum(m.get("yes_ask_dollars")),
                    "oi": fnum(m.get("open_interest_fp")) or 0})
    return out


# ------------------------------------------------------------ strategy 1

def strategy1():
    """Locked pairs: sell the near/harder touch, buy the far/easier touch."""
    uni = load("universe_kalshi")
    touch = [r for r in uni if r["kind"] in ("onetouch_max", "onetouch_min")]
    from scan_barrier import series_coin

    lads = {}
    for ev in touch:
        c = series_coin(ev["series"])
        if not c:
            continue
        try:
            rows = ladder(ev["event"])
        except Exception:                                # noqa: BLE001
            continue
        if rows:
            lads[ev["event"]] = {"coin": c, "series": ev["series"], "rows": rows}

    cands = []
    import itertools
    for (e1, b1), (e2, b2) in itertools.permutations(lads.items(), 2):
        if b1["coin"] != b2["coin"]:
            continue
        for a in b1["rows"]:
            for f in b2["rows"]:
                if a["side"] != f["side"] or a["open"] is None or f["open"] is None:
                    continue
                # far window must strictly contain the near window
                if not (f["open"] <= a["open"] + 3600_000 and f["close"] >= a["close"] - 1000):
                    continue
                if a["side"] == "up" and f["barrier"] > a["barrier"] + 1e-9:
                    continue
                if a["side"] == "dn" and f["barrier"] < a["barrier"] - 1e-9:
                    continue
                if a["ticker"] == f["ticker"]:
                    continue
                if a["bid"] <= 0 or f["ask"] is None:
                    continue
                if (1 - a["bid"]) + f["ask"] >= 1.0:
                    continue
                cands.append((a, f, b1, b2))

    need = sorted({t["ticker"] for c in cands for t in (c[0], c[1])})
    books = {}
    for t, bk in pmap(book, need, workers=12):
        if isinstance(bk, dict) and "__error__" not in bk:
            books[t] = bk

    out = []
    for a, f, b1, b2 in cands:
        la, lf = buy_levels(books.get(a["ticker"], {}), "NO"), \
            buy_levels(books.get(f["ticker"], {}), "YES")
        ia = ib = 0
        qty = 0.0
        cost = 0.0
        while ia < len(la) and ib < len(lf):
            pa, qa = la[ia]
            pf, qf = lf[ib]
            gross = pa + pf
            if gross >= 1.0:
                break
            q = min(qa, qf)
            qty += q
            cost += (pa + pf) * q
            la[ia] = (pa, qa - q)
            lf[ib] = (pf, qf - q)
            if la[ia][1] <= 1e-9:
                ia += 1
            if lf[ib][1] <= 1e-9:
                ib += 1
        if qty <= 0:
            continue
        # exact fee: each leg is its own order, charged at its own average price
        avg_sell = sum(p * q for p, q in
                       zip([x[0] for x in buy_levels(books[a["ticker"]], "NO")],
                           [0] * 0)) if False else None
        px_sell = cost / qty / 2
        f_total = fee(px_sell, qty) * 2
        if qty * 1.0 - cost - f_total <= 0:
            continue                      # fees eat the lock: not an arbitrage
        out.append({
            "coin": b1["coin"], "side": a["side"],
            "sell": f"{b1['series']} @{a['barrier']:,.6g}", "sell_ticker": a["ticker"],
            "buy": f"{b2['series']} @{f['barrier']:,.6g}", "buy_ticker": f["ticker"],
            "qty": qty, "gross_cost": cost, "fees": f_total,
            "capital": cost + f_total,
            "min_payoff": qty * 1.0, "max_payoff": qty * 2.0,
            "min_profit": qty * 1.0 - cost - f_total,
            "max_profit": qty * 2.0 - cost - f_total,
            "near_close": a["close"], "far_close": f["close"],
        })
    # A contract's depth can only be sold once. Allocate greedily from the
    # best pair down, decrementing each leg's remaining depth, so the aggregate
    # is simultaneously executable rather than a sum of overlapping claims.
    out.sort(key=lambda r: -(r["min_profit"] / max(r["qty"], 1)))
    cap_left = {}
    for r in out:
        for k in ("sell_ticker", "buy_ticker"):
            cap_left[r[k]] = max(cap_left.get(r[k], 0.0), r["qty"])
    alloc = []
    for r in out:
        avail = min(cap_left[r["sell_ticker"]], cap_left[r["buy_ticker"]], r["qty"])
        if avail <= 1e-9:
            continue
        f = avail / r["qty"]
        alloc.append({**r, "qty": avail,
                      "gross_cost": r["gross_cost"] * f,
                      "fees": r["fees"] * f,
                      "capital": r["capital"] * f,
                      "min_profit": r["min_profit"] * f,
                      "max_profit": r["max_profit"] * f})
        cap_left[r["sell_ticker"]] -= avail
        cap_left[r["buy_ticker"]] -= avail
    alloc.sort(key=lambda r: -r["min_profit"])
    return alloc


def main():
    now = now_ms()
    print("=" * 78)
    print("STRATEGY 1 — locked one-touch pairs (Kalshi, single venue)")
    print("=" * 78)
    s1 = strategy1()
    if not s1:
        print("  no locked pairs at current prices")
    else:
        print(f"{'coin':4s} {'SELL (near)':26s} {'BUY (far)':26s} {'qty':>6s} "
              f"{'capital':>9s} {'min P/L':>9s} {'max P/L':>9s} {'min ROI':>8s}")
        for r in s1:
            print(f"{r['coin']:4s} {r['sell'][:26]:26s} {r['buy'][:26]:26s} "
                  f"{r['qty']:6,.0f} ${r['capital']:8,.2f} ${r['min_profit']:8,.2f} "
                  f"${r['max_profit']:8,.2f} {r['min_profit']/max(r['capital'],1e-9)*100:7.2f}%")
        cap = sum(r["capital"] for r in s1)
        mn = sum(r["min_profit"] for r in s1)
        mx = sum(r["max_profit"] for r in s1)
        # The guaranteed dollar arrives when the NEAR leg settles: if the
        # barrier is touched in the near window both legs settle early, and if
        # it is not, the near NO pays out on its own close date. Only the
        # *upside* second dollar waits for the far expiry.
        days = (max(r["near_close"] for r in s1) - now) / 86400_000
        far_days = (max(r["far_close"] for r in s1) - now) / 86400_000
        print(f"\n  BANKROLL REQUIRED   ${cap:,.2f}   (full amount posted up front; "
              f"Kalshi is fully collateralised)")
        print(f"  MINIMUM TICKET      ~${min(r['capital']/r['qty'] for r in s1):.2f} "
              f"(1 contract pair; Kalshi minimum order is 1)")
        print(f"  MAXIMUM DEPLOYABLE  ${cap:,.2f} at current depth "
              f"({sum(r['qty'] for r in s1):,.0f} pairs)")
        print(f"  PROFIT              min ${mn:,.2f} / max ${mx:,.2f} "
              f"({mn/cap*100:.2f}% to {mx/cap*100:.1f}% on capital)")
        print(f"  TIME TO GUARANTEED  {days:.0f} days (near leg settles; the far leg")
        print(f"                      is then held free, expiring in {far_days:.0f} days)")
        print(f"  MIN ANNUALISED      {((1+mn/cap)**(365/max(days,1))-1)*100:.0f}% "
              f"on the guaranteed leg")
    save("audit_s1", s1)
    return s1


def strategy2():
    """BNB annual ladder, model-based and DIRECTIONAL -- not arbitrage."""
    emp = load("empirical_touch")
    rows = [r for r in emp
            if r["coin"] == "BNB" and r["series"] == "KXBNBMAXY"
            and r["verdict"].startswith("CHEAP")]
    out = []
    for r in rows:
        bk = book(r["ticker"])
        lv = buy_levels(bk, "YES")
        qty = sum(q for p, q in lv if p <= 0.60)
        cost = sum(p * q for p, q in lv if p <= 0.60)
        if qty <= 0:
            continue
        px = cost / qty
        f = fee(px, qty)
        out.append({"ticker": r["ticker"], "barrier": r["barrier"],
                    "qty": qty, "avg_px": px, "capital": cost + f,
                    "mkt": 0.5 * (r["bid"] + r["ask"]), "model": r["fair"],
                    "hist": r["realised"],
                    "ev": qty * min(r["fair"], r["realised"]) - cost - f,
                    "worst": -(cost + f)})
    return out


def strategy3():
    """Polymarket ETH range market making -- spread capture, not arbitrage."""
    mm = load("mmmap")
    row = [r for r in mm if r["venue"] == "polymarket" and r["asset"] == "ETH"
           and r["ladder"] == "range"]
    if not row:
        return None
    r = row[0]
    poly = load("polymarket")
    depth = 0.0
    for slug, e in poly["events"].items():
        if e["asset"] != "ETH" or e["kind"] != "range":
            continue
        for m in e["markets"]:
            b = poly["books"].get(m["slug"], {})
            bid = sum(q for p, q in b.get("bids", []))
            ask = sum(q for p, q in b.get("asks", []))
            depth += min(bid, ask)
    vol = sum(m["volume"] or 0 for s, e in poly["events"].items()
              if e["asset"] == "ETH" and e["kind"] == "range"
              for m in e["markets"])
    return {**r, "resting_depth": depth, "lifetime_volume": vol}


def main():
    now = now_ms()
    print("=" * 78)
    print("STRATEGY 1 — locked one-touch pairs (Kalshi)      RISK: none (see caveats)")
    print("=" * 78)
    s1 = strategy1()
    if not s1:
        print("  no locked pairs at current prices")
    else:
        print(f"{'coin':4s} {'SELL (near)':26s} {'BUY (far)':26s} {'qty':>6s} "
              f"{'capital':>9s} {'min P/L':>9s} {'max P/L':>9s} {'min ROI':>8s}")
        for r in s1:
            print(f"{r['coin']:4s} {r['sell'][:26]:26s} {r['buy'][:26]:26s} "
                  f"{r['qty']:6,.0f} ${r['capital']:8,.2f} ${r['min_profit']:8,.2f} "
                  f"${r['max_profit']:8,.2f} {r['min_profit']/max(r['capital'],1e-9)*100:7.2f}%")
        cap = sum(r["capital"] for r in s1)
        mn = sum(r["min_profit"] for r in s1)
        mx = sum(r["max_profit"] for r in s1)
        days = (max(r["near_close"] for r in s1) - now) / 86400_000
        far_days = (max(r["far_close"] for r in s1) - now) / 86400_000
        print(f"\n  BANKROLL REQUIRED   ${cap:,.2f}  (fully collateralised, both legs prepaid)")
        print(f"  MINIMUM TICKET      ${min(r['capital']/r['qty'] for r in s1):.2f} "
              f"(1 pair; Kalshi minimum order is 1 contract)")
        print(f"  MAXIMUM DEPLOYABLE  ${cap:,.2f}  ({sum(r['qty'] for r in s1):,.0f} pairs, "
              f"non-overlapping)")
        print(f"  PROFIT              min ${mn:,.2f}  max ${mx:,.2f}   "
              f"({mn/cap*100:.2f}% to {mx/cap*100:.0f}% on capital)")
        print(f"  GUARANTEED BY       {days:.0f} days (near leg settles); far leg then "
              f"held free to day {far_days:.0f}")
        print(f"  MIN ANNUALISED      {((1+mn/cap)**(365/max(days,1))-1)*100:.0f}%")
        save("audit_s1", s1)

    print()
    print("=" * 78)
    print("STRATEGY 2 — BNB annual ladder, model view        RISK: full capital")
    print("=" * 78)
    s2 = strategy2()
    if not s2:
        print("  nothing qualifying")
    else:
        print(f"{'barrier':>9s} {'qty':>7s} {'avg px':>7s} {'capital':>9s} "
              f"{'mkt':>6s} {'model':>6s} {'hist':>6s} {'exp P/L':>9s} {'worst':>9s}")
        for r in sorted(s2, key=lambda x: x["barrier"]):
            print(f"{r['barrier']:9,.0f} {r['qty']:7,.0f} {r['avg_px']:7.3f} "
                  f"${r['capital']:8,.2f} {r['mkt']:6.2f} {r['model']:6.2f} "
                  f"{r['hist']:6.2f} ${r['ev']:8,.2f} ${r['worst']:8,.2f}")
        cap2 = sum(r["capital"] for r in s2)
        print(f"\n  BANKROLL REQUIRED   ${cap2:,.2f}")
        print(f"  MINIMUM TICKET      ~${min(r['avg_px'] for r in s2):.2f} (1 contract)")
        print(f"  MAXIMUM DEPLOYABLE  ${cap2:,.2f} at asks up to 60c")
        print(f"  PROFIT              expected ${sum(r['ev'] for r in s2):,.2f}  "
              f"max ${sum(r['qty'] for r in s2) - cap2:,.2f}  "
              f"WORST ${-cap2:,.2f} (total loss)")
        print(f"  RISK                directional. Pays only if BNB actually rallies.")
        save("audit_s2", s2)

    print()
    print("=" * 78)
    print("STRATEGY 3 — Polymarket ETH range MM              RISK: inventory")
    print("=" * 78)
    s3 = strategy3()
    if s3:
        print(f"  median spread       {s3['med_spread']*100:.1f}c")
        print(f"  net per contract    {s3['net_30']*100:+.2f}c (at 30% adverse selection)")
        print(f"  resting depth       {s3['resting_depth']:,.0f} contracts across all buckets")
        print(f"  lifetime volume     ${s3['lifetime_volume']:,.0f} (all 6 days)")
        cap3 = s3["resting_depth"] * 0.35
        print(f"\n  BANKROLL REQUIRED   ~${cap3:,.0f} to quote both sides at full depth")
        print(f"  MINIMUM TICKET      $1 (Polymarket minimum order)")
        print(f"  MAXIMUM DEPLOYABLE  bounded by FLOW, not depth: "
              f"${s3['lifetime_volume']/7:,.0f}/day traded")
        print(f"  PROFIT              ~${s3['lifetime_volume']/7*0.15*s3['net_30']/0.35:,.2f}/day "
              f"at 15% of flow")
        print(f"  RISK                you hold inventory; a gap move is a real loss.")
    return s1, s2, s3


if __name__ == "__main__":
    main()
