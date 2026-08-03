"""Tightest implied-distribution bounds consistent with a venue's quotes.

Reading a ladder off mid prices breaks down where quotes are pinned: a Kalshi
event with 188 buckets each quoted 0 bid / 1c ask carries 94c of "phantom"
mid-price mass, which is why a naive read of KXBTC-26AUG0312 implies 610%
volatility on a print 23 minutes away.

The honest statement is not a point estimate but an interval. Any distribution
the market cannot be arbitraged against must satisfy

    p >= 0,  sum p = 1,  bid_m <= sum_{cells in m} p <= ask_m   for every quote m

which is a polytope. Linear functionals of p can then be minimised and
maximised over it exactly. We use

    W = sum_i p_i * ln(K_i / F)^2        (total implied variance about the forward)

so each ladder yields a *band* [sigma_lo, sigma_hi] rather than a number, and
two venues only genuinely disagree when their bands are disjoint. Infeasibility
of the polytope is itself the signature of a static arbitrage.
"""
import datetime as dt
import math

import numpy as np
from scipy.optimize import linprog

from common import load, save

YEAR_H = 365.25 * 24.0


def build_polytope(quotes, reps):
    """A_ub/b_ub encoding bid <= p.a <= ask for every quote, plus sum p = 1."""
    n = len(reps)
    A_ub, b_ub = [], []
    for q in quotes:
        a = np.array([1.0 if q["in"](c) else 0.0 for c in reps])
        if a.sum() == 0:
            continue
        # ask == 0 is Kalshi's placeholder for "no offer", not a free option
        if q["ask"] is not None and 0.0 < q["ask"] < 1.0:
            A_ub.append(a)
            b_ub.append(q["ask"])
        if q["bid"] is not None and 0.0 < q["bid"] < 1.0 and (
                q["ask"] is None or q["bid"] <= q["ask"]):
            A_ub.append(-a)
            b_ub.append(-q["bid"])
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    return (np.array(A_ub) if A_ub else np.zeros((0, n)),
            np.array(b_ub) if b_ub else np.zeros(0), A_eq, b_eq)


def optimise(c, A_ub, b_ub, A_eq, b_eq, sense="min"):
    cc = np.array(c, float) * (1.0 if sense == "min" else -1.0)
    r = linprog(cc, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=[(0.0, 1.0)] * len(cc), method="highs")
    if not r.success:
        return None, None
    val = float(np.dot(c, r.x))
    return val, r.x


def ladder_bounds(quotes, edges, F, tau_h, tail_mult=1.0):
    """Bounds on total implied variance and on the CDF at every edge.

    `edges` are the interior boundaries; cells are the intervals between them
    plus the two open tails. Open tails get a representative point one bucket
    width beyond the last edge -- their probability is capped by the quotes, so
    the convention moves the answer only when the wings carry real mass.
    """
    e = sorted(set(edges))
    if len(e) < 3:
        return None
    reps = [e[0] - tail_mult * (e[1] - e[0])]
    for i in range(len(e) - 1):
        reps.append(0.5 * (e[i] + e[i + 1]))
    reps.append(e[-1] + tail_mult * (e[-1] - e[-2]))
    reps = [max(r, 1e-6) for r in reps]

    A_ub, b_ub, A_eq, b_eq = build_polytope(quotes, reps)
    w = np.array([math.log(r / F) ** 2 for r in reps])

    lo, _ = optimise(w, A_ub, b_ub, A_eq, b_eq, "min")
    hi, _ = optimise(w, A_ub, b_ub, A_eq, b_eq, "max")
    if lo is None or hi is None:
        return {"feasible": False}

    T = tau_h / YEAR_H
    out = {"feasible": True, "W_lo": lo, "W_hi": hi,
           "sigma_lo": math.sqrt(max(lo, 0) / T) if T > 0 else None,
           "sigma_hi": math.sqrt(max(hi, 0) / T) if T > 0 else None,
           "n_cells": len(reps), "n_quotes": len(quotes)}

    # CDF band at a handful of representative strikes
    band = []
    for K in e:
        ind = np.array([1.0 if r > K else 0.0 for r in reps])
        a, _ = optimise(ind, A_ub, b_ub, A_eq, b_eq, "min")
        b, _ = optimise(ind, A_ub, b_ub, A_eq, b_eq, "max")
        if a is not None:
            band.append((K, a, b))
    out["cdf_band"] = band
    return out


# ---------------------------------------------------------------- adapters

def kalshi_quotes(markets):
    qs, edges = [], []
    for m in markets:
        if m["yes_ask"] is None:
            continue
        st, fl, cap = m["strike_type"], m["floor_strike"], m["cap_strike"]
        if st in ("greater", "greater_or_equal"):
            qs.append({"in": (lambda x, f=fl: x > f), "bid": m["yes_bid"],
                       "ask": m["yes_ask"]})
            edges.append(round(fl + 0.01, 2))
        elif st in ("less", "less_or_equal"):
            qs.append({"in": (lambda x, c=cap: x < c), "bid": m["yes_bid"],
                       "ask": m["yes_ask"]})
            edges.append(cap)
        elif st == "between":
            qs.append({"in": (lambda x, a=fl, b=cap: a <= x <= b),
                       "bid": m["yes_bid"], "ask": m["yes_ask"]})
            edges += [fl, round(cap + 0.01, 2)]
    return qs, edges


def poly_quotes(event, scale=1.0):
    import re
    NUM = re.compile(r"[\d,]+")
    qs, edges = [], []
    for m in event["markets"]:
        t = (m["group_title"] or "").replace("$", "").strip()
        nums = [float(x.replace(",", "")) * scale for x in NUM.findall(t)]
        if not nums:
            continue
        bid = m["best_bid"] if m["best_bid"] is not None else 0.0
        ask = m["best_ask"] if m["best_ask"] is not None else 1.0
        if event["kind"] == "above":
            qs.append({"in": (lambda x, k=nums[0]: x > k), "bid": bid, "ask": ask})
            edges.append(nums[0])
        else:
            if t.startswith("<"):
                lo, hi = 0.0, nums[0]
            elif t.startswith(">"):
                lo, hi = nums[0], float("inf")
            elif len(nums) >= 2:
                lo, hi = nums[0], nums[1]
            else:
                continue
            qs.append({"in": (lambda x, a=lo, b=hi: a <= x < b), "bid": bid, "ask": ask})
            if lo > 0:
                edges.append(lo)
            if math.isfinite(hi):
                edges.append(hi)
    return qs, edges


def iso_ms(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def main():
    kal, poly, ref, seas = load("kalshi"), load("polymarket"), load("ref"), load("seasonality")
    der = load("deribit")
    from seasonality import Clock
    from rnd import TermSurface
    from implied import sigma_eff

    clock = Clock(seas["clock_hour"], seas["clock_dow"])
    now = kal["fetched_ms"]
    spot = ref["spot"]
    usd = {a: float(np.mean([spot[f"{v}_{a}"] for v in
                             ("coinbase", "bitstamp", "gemini", "kraken")
                             if spot.get(f"{v}_{a}")])) for a in ("BTC", "ETH")}
    peg = {a: usd[a] / spot[f"binance_{a}"] for a in ("BTC", "ETH")}
    surf = {a: TermSurface(der["ccy"][a], der["fetched_ms"], a) for a in ("BTC", "ETH")}

    rows = []

    # --- Kalshi: combine BOTH ladders, since they settle on the same print
    groups = {}
    for series, evs in kal["series"].items():
        asset = "ETH" if "ETH" in series else "BTC"
        for ev, blob in evs.items():
            live = [m for m in blob["markets"]
                    if m["yes_ask"] is not None and 0.0 < m["yes_ask"] < 1.0]
            if len(live) < 8:
                continue
            key = (asset, ev.split("-", 1)[1])
            g = groups.setdefault(key, {"markets": [], "close": live[0]["close_time"]})
            g["markets"] += live
    for (asset, suffix), g in sorted(groups.items()):
        t1 = iso_ms(g["close"])
        tau, cal = clock.tau(now, t1), (t1 - now) / 3600_000.0
        if cal <= 0:
            continue
        qs, edges = kalshi_quotes(g["markets"])
        r = ladder_bounds(qs, edges, usd[asset], tau)
        if r:
            r.update({"venue": "kalshi", "asset": asset, "event": suffix,
                      "cal_h": cal, "tau_h": tau, "settle": g["close"]})
            rows.append(r)

    # --- Polymarket: combine the above + range ladders for the same day
    pg = {}
    for slug, e in poly["events"].items():
        pg.setdefault((e["asset"], e["day"]), []).append(e)
    for (asset, day), evs in sorted(pg.items()):
        t1 = iso_ms(evs[0]["end_date"])
        tau, cal = clock.tau(now, t1), (t1 - now) / 3600_000.0
        if cal <= 0:
            continue
        qs, edges = [], []
        for e in evs:
            q, ed = poly_quotes(e, peg[asset])
            qs += q
            edges += ed
        r = ladder_bounds(qs, edges, usd[asset], tau)
        if r:
            r.update({"venue": "polymarket", "asset": asset, "event": f"aug{day}",
                      "cal_h": cal, "tau_h": tau, "settle": evs[0]["end_date"]})
            rows.append(r)

    # --- Deribit reference point
    for asset in ("BTC", "ETH"):
        for e, s in sorted(surf[asset].exps.items(), key=lambda x: x[1]["T"]):
            cal = (e - now) / 3600_000.0
            if cal <= 0 or cal > 200:
                continue
            tau = clock.tau(now, e)
            d = s["dens"]
            qd = {q: d.quantile(q) for q in (0.16, 0.5, 0.84)}
            sg = sigma_eff(qd, tau)
            rows.append({"venue": "deribit", "asset": asset,
                         "event": dt.datetime.utcfromtimestamp(e / 1000).strftime("%d%b-%HZ"),
                         "cal_h": cal, "tau_h": tau, "feasible": True,
                         "sigma_lo": sg, "sigma_hi": sg, "n_quotes": s["n_quotes"],
                         "settle": dt.datetime.utcfromtimestamp(e / 1000).isoformat() + "Z"})

    rows.sort(key=lambda r: (r["asset"], r["cal_h"], r["venue"]))
    save("bounds", rows)

    for asset in ("BTC", "ETH"):
        print(f"\n=== {asset}: implied volatility bands consistent with quotes "
              f"(business clock)")
        print(f"{'venue':11s} {'event':14s} {'cal_h':>7s} {'biz_h':>7s} {'nq':>4s} "
              f"{'sigma_lo':>9s} {'sigma_hi':>9s}   width")
        for r in rows:
            if r["asset"] != asset or not r.get("feasible"):
                continue
            lo = r.get("sigma_lo") or 0
            hi = r.get("sigma_hi") or 0
            print(f"{r['venue']:11s} {r['event'][:14]:14s} {r['cal_h']:7.2f} "
                  f"{r['tau_h']:7.2f} {r.get('n_quotes',0):4d} "
                  f"{lo*100:8.2f}% {hi*100:8.2f}%   {(hi-lo)*100:6.2f}pp")
        bad = [r for r in rows if r["asset"] == asset and not r.get("feasible")]
        for r in bad:
            print(f"{r['venue']:11s} {r['event'][:14]:14s}  INFEASIBLE -> static arbitrage")
    return rows


if __name__ == "__main__":
    main()
