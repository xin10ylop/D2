"""Executable edge, contract by contract, at real prices and real sizes.

Mid-price comparisons are where bad strategies come from. Everything here is
priced the way it would actually trade: buying pays the ask, selling pays the
bid (or, on Kalshi, buys the opposite side of the book), fees are charged on
entry, and available size is read off the order book level by level.

Edge is measured against the Deribit reference measure on the business clock,
and every number is reported with a sensitivity band: if the reference vol is
wrong by +/- vol_bump, how much of the edge survives? An "edge" that vanishes
under a 2-point vol shift is a model artifact, not an opportunity.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from engine import kalshi_fee_rate
from fitladder import model_cdf
from rv import Reference, iso_ms, hm
from seasonality import Clock

YEAR_H = 365.25 * 24.0


def ref_prob(R, lo, hi, tau_h, F, dvol=0.0):
    """Reference probability of S landing in [lo, hi), with a vol bump."""
    w = R.total_var(tau_h)
    if not w or w <= 0:
        return None
    T = tau_h / YEAR_H
    s0 = math.sqrt(w / T) + dvol
    beta, gamma = R.skew_at(tau_h)
    ks = [k for k in (lo, hi) if math.isfinite(k) and k > 0]
    p = {}
    for k in ks:
        p[k] = float(model_cdf([k], F, T, s0, beta, gamma)[0])
    a = p.get(lo, 1.0) if (math.isfinite(lo) and lo > 0) else 1.0
    b = p.get(hi, 0.0) if math.isfinite(hi) else 0.0
    return max(a - b, 0.0)


def kalshi_book_levels(book, side):
    """Executable (price, size) levels for buying YES or NO on Kalshi."""
    ladder = book.get("no" if side == "YES" else "yes", [])
    return [(1.0 - p, s) for p, s in sorted(ladder, key=lambda x: -x[0])
            if 0.0 < 1.0 - p < 1.0 and s > 0]


def poly_book_levels(book, side):
    if side == "YES":
        return [(p, s) for p, s in sorted(book.get("asks", []), key=lambda x: x[0])
                if 0.0 < p < 1.0 and s > 0]
    return [(1.0 - p, s) for p, s in sorted(book.get("bids", []), key=lambda x: -x[0])
            if 0.0 < 1.0 - p < 1.0 and s > 0]


def ladder_forward(ladder, tau_h, F0):
    """Each venue's own implied forward, from its own quotes."""
    from fitladder import fit_ladder
    f = fit_ladder(ladder, tau_h / YEAR_H, F0)
    return (f["F"] if f else None), f


def main(vol_bump=0.02, min_edge=0.01, min_hours=1.0, recenter=True):
    kal, poly, ref, seas = load("kalshi"), load("polymarket"), load("ref"), load("seasonality")
    der = load("deribit")
    clock = Clock(seas["clock_hour"], seas["clock_dow"])
    now = kal["fetched_ms"]
    spot = ref["spot"]
    usd = {a: float(np.mean([spot[f"{v}_{a}"] for v in
                             ("coinbase", "bitstamp", "gemini", "kraken")
                             if spot.get(f"{v}_{a}")])) for a in ("BTC", "ETH")}
    peg = {a: usd[a] / spot[f"binance_{a}"] for a in ("BTC", "ETH")}
    R = {a: Reference(der["ccy"][a], der["fetched_ms"], clock, a) for a in ("BTC", "ETH")}

    cands = []

    # ---------------- Kalshi
    books = kal.get("books", {})
    for series, evs in kal["series"].items():
        asset = "ETH" if "ETH" in series else "BTC"
        for ev, blob in evs.items():
            live = [m for m in blob["markets"]
                    if m["yes_ask"] is not None and 0.0 < m["yes_ask"] <= 1.0]
            if not live:
                continue
            t1 = iso_ms(live[0]["close_time"])
            cal = (t1 - now) / 3600_000.0
            tau = clock.tau(now, t1)
            if cal <= min_hours or tau <= 0:
                continue
            # Centre the reference on this ladder's OWN implied forward. Using an
            # external spot instead turns every few-dollar snapshot-timing gap
            # into a fake edge on the near-the-money strikes -- at 24 minutes to
            # expiry a $54 offset is 0.6 sigma and swamps everything real. What
            # is left after re-centring is pure distribution shape.
            lad = [(m["floor_strike"], m["yes_bid"], m["yes_ask"]) for m in live
                   if m["strike_type"] in ("greater", "greater_or_equal")]
            Fv, _ = ladder_forward(lad, tau, usd[asset]) if (recenter and len(lad) >= 6) \
                else (None, None)
            Fuse = Fv or usd[asset]
            for m in live:
                bk = books.get(m["ticker"])
                if not bk:
                    continue
                st, fl, cap = m["strike_type"], m["floor_strike"], m["cap_strike"]
                if st in ("greater", "greater_or_equal"):
                    lo, hi = fl, float("inf")
                elif st in ("less", "less_or_equal"):
                    lo, hi = 0.0, cap
                elif st == "between":
                    lo, hi = fl, cap
                else:
                    continue
                fair = ref_prob(R[asset], lo, hi, tau, Fuse)
                if fair is None:
                    continue
                lo_f = ref_prob(R[asset], lo, hi, tau, Fuse, -vol_bump)
                hi_f = ref_prob(R[asset], lo, hi, tau, Fuse, +vol_bump)
                for side in ("YES", "NO"):
                    tgt = fair if side == "YES" else 1 - fair
                    band = ((lo_f, hi_f) if side == "YES" else (1 - hi_f, 1 - lo_f))
                    for px, sz in kalshi_book_levels(bk, side)[:6]:
                        cost = px + kalshi_fee_rate(px)
                        e = tgt - cost
                        if e < min_edge:
                            break
                        cands.append({
                            "venue": "kalshi", "asset": asset, "event": ev,
                            "ticker": m["ticker"], "desc": m["subtitle"], "side": side,
                            "price": px, "fee": kalshi_fee_rate(px), "cost": cost,
                            "fair": tgt, "edge": e, "size": sz,
                            "edge_lo": min(band) - cost, "edge_hi": max(band) - cost,
                            "cal_h": cal, "tau_h": tau, "settle": live[0]["close_time"], "F": Fuse,
                            "notional": sz * cost, "profit": e * sz})

    # ---------------- Polymarket
    pbooks = poly.get("books", {})
    import re
    NUM = re.compile(r"[\d,]+")
    for slug, e in poly["events"].items():
        asset = e["asset"]
        sc = peg[asset]
        t1 = iso_ms(e["end_date"])
        cal = (t1 - now) / 3600_000.0
        tau = clock.tau(now, t1)
        if cal <= min_hours or tau <= 0:
            continue
        lad = []
        for m in e["markets"]:
            t = (m["group_title"] or "").replace(",", "").replace("$", "")
            try:
                lad.append((float(t) * sc, m["best_bid"], m["best_ask"]))
            except ValueError:
                pass
        Fv, _ = ladder_forward(sorted(lad), tau, usd[asset]) if (
            recenter and len(lad) >= 6) else (None, None)
        Fuse = Fv or usd[asset]
        for m in e["markets"]:
            bk = pbooks.get(m["slug"])
            if not bk:
                continue
            t = (m["group_title"] or "").replace("$", "").strip()
            nums = [float(x.replace(",", "")) * sc for x in NUM.findall(t)]
            if not nums:
                continue
            if e["kind"] == "above":
                lo, hi = nums[0], float("inf")
            elif t.startswith("<"):
                lo, hi = 0.0, nums[0]
            elif t.startswith(">"):
                lo, hi = nums[0], float("inf")
            elif len(nums) >= 2:
                lo, hi = nums[0], nums[1]
            else:
                continue
            fair = ref_prob(R[asset], lo, hi, tau, Fuse)
            if fair is None:
                continue
            lo_f = ref_prob(R[asset], lo, hi, tau, Fuse, -vol_bump)
            hi_f = ref_prob(R[asset], lo, hi, tau, Fuse, +vol_bump)
            for side in ("YES", "NO"):
                tgt = fair if side == "YES" else 1 - fair
                band = ((lo_f, hi_f) if side == "YES" else (1 - hi_f, 1 - lo_f))
                for px, sz in poly_book_levels(bk, side)[:6]:
                    ed = tgt - px
                    if ed < min_edge:
                        break
                    cands.append({
                        "venue": "polymarket", "asset": asset, "event": slug,
                        "ticker": m["slug"], "desc": m["group_title"], "side": side,
                        "price": px, "fee": 0.0, "cost": px,
                        "fair": tgt, "edge": ed, "size": sz,
                        "edge_lo": min(band) - px, "edge_hi": max(band) - px,
                        "cal_h": cal, "tau_h": tau, "settle": e["end_date"], "F": Fuse,
                        "notional": sz * px, "profit": ed * sz})

    # Robust candidates keep a positive edge under a +/- vol_bump reference shift.
    for c in cands:
        c["robust"] = min(c["edge_lo"], c["edge_hi"]) > 0
    cands.sort(key=lambda c: -c["profit"])
    save("edge", cands)

    rob = [c for c in cands if c["robust"]]
    print(f"candidates with edge > {min_edge:.0%}: {len(cands)}   "
          f"robust to +/-{vol_bump*100:.0f} vol pts: {len(rob)}")
    print(f"\ntop executable edges (robust only), ranked by expected profit")
    print(f"{'venue':11s} {'asset':4s} {'event':22s} {'contract':26s} {'sd':3s} "
          f"{'px':>6s} {'fair':>6s} {'edge':>6s} {'lo':>6s} {'hi':>6s} "
          f"{'size':>9s} {'$prof':>9s}")
    for c in rob[:30]:
        print(f"{c['venue']:11s} {c['asset']:4s} {c['event'][:22]:22s} "
              f"{str(c['desc'])[:26]:26s} {c['side']:3s} "
              f"{c['price']:6.3f} {c['fair']:6.3f} {c['edge']:+6.3f} "
              f"{min(c['edge_lo'],c['edge_hi']):+6.3f} {max(c['edge_lo'],c['edge_hi']):+6.3f} "
              f"{c['size']:9,.0f} {c['profit']:9,.0f}")

    agg = {}
    for c in rob:
        k = (c["venue"], c["asset"], c["event"])
        a = agg.setdefault(k, {"profit": 0.0, "notional": 0.0, "n": 0})
        a["profit"] += c["profit"]
        a["notional"] += c["notional"]
        a["n"] += 1
    print("\nby event (robust candidates only):")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]["profit"])[:20]:
        print(f"  {k[0]:11s} {k[1]:4s} {k[2][:34]:34s} n={v['n']:3d} "
              f"notional=${v['notional']:11,.0f} exp.profit=${v['profit']:10,.0f} "
              f"({v['profit']/max(v['notional'],1)*100:5.2f}%)")
    return cands


if __name__ == "__main__":
    main()
