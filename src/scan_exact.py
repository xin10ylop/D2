"""Exact (risk-free) arbitrage scan inside each settlement print.

Two families of contract on each venue settle on *the identical number*:

  Kalshi      KXBTCD-<t>  digitals   and  KXBTC-<t>  250-wide buckets
              -> both on the 60s BRTI average at time <t>
  Polymarket  bitcoin-above-on-august-N and bitcoin-price-on-august-N
              -> both on the Binance BTC/USDT 1m close at 16:00Z on day N

So the two order books are redundant descriptions of one distribution, and any
inconsistency between them is money. We throw every quoted leg from *both*
ladders into one LP and let it find the combination.
"""
import json
import re
import sys

from common import load, save
from engine import Leg, kalshi_legs, poly_legs, solve_arb

FEE_MULT = 0.07


def canonical_states(bounds):
    """One representative price strictly inside every partition cell.

    Kalshi quotes digitals at X-0.01 and buckets starting at X, so a naive
    grid would invent sub-cent states that no settlement print can occupy.
    Snapping every boundary to the cent grid and sampling cell midpoints
    keeps the state space to the outcomes that can actually happen.
    """
    b = sorted({round(x, 2) for x in bounds if x is not None})
    if not b:
        return []
    out = [b[0] - 500.0]
    for i in range(len(b) - 1):
        out.append(0.5 * (b[i] + b[i + 1]))
    out.append(b[-1] + 500.0)
    return [x for x in out if x > 0]


# ------------------------------------------------------------------ Kalshi

def kalshi_groups(kal):
    """Bundle the digital ladder and the bucket ladder that share a print."""
    groups = {}
    for series, evs in kal["series"].items():
        asset = "ETH" if "ETH" in series else "BTC"
        kind = "range" if series in ("KXBTC", "KXETH") else "above"
        for ev, blob in evs.items():
            suffix = ev.split("-", 1)[1]
            g = groups.setdefault((asset, suffix), {"asset": asset, "suffix": suffix,
                                                    "ladders": {}, "close": None})
            g["ladders"][kind] = blob["markets"]
            if blob["markets"]:
                g["close"] = blob["markets"][0]["close_time"]
    return groups


def kalshi_scan(kal, capital=100_000, max_size=None):
    books = kal.get("books", {})
    results = []
    for (asset, suffix), g in sorted(kalshi_groups(kal).items()):
        legs, bounds = [], []
        for kind, mkts in g["ladders"].items():
            for m in mkts:
                if m["yes_ask"] is None or m["yes_ask"] >= 1.0:
                    continue
                bk = books.get(m["ticker"])
                ls = kalshi_legs(m, bk, FEE_MULT, max_size)
                for lg in ls:
                    lg.meta["ladder"] = kind
                legs += ls
                if m["floor_strike"] is not None:
                    bounds.append(round(m["floor_strike"] + 0.01, 2)
                                  if m["strike_type"] in ("greater", "greater_or_equal")
                                  else m["floor_strike"])
                if m["cap_strike"] is not None:
                    bounds.append(round(m["cap_strike"] + 0.01, 2))
        if len(legs) < 4:
            continue
        states = canonical_states(bounds)
        r = solve_arb(legs, states, capital=capital)
        if not r:
            continue
        results.append({
            "venue": "kalshi", "asset": asset, "event": suffix,
            "close": g["close"], "n_legs": len(legs), "n_states": len(states),
            "profit": r["profit"], "outlay": r["outlay"], "roi": r["roi"],
            "picks": [(lg.key, lg.price, lg.fee, round(q, 2), lg.meta.get("ladder"),
                       lg.meta.get("subtitle")) for lg, q in r["legs"]],
        })
    return results


# -------------------------------------------------------------- Polymarket

NUM = re.compile(r"[\d,]+")


def poly_bounds(group_title, kind):
    """Map a Polymarket outcome label to a half-open price interval."""
    t = (group_title or "").replace("$", "").strip()
    nums = [float(x.replace(",", "")) for x in NUM.findall(t)]
    if kind == "above":
        return (nums[0], float("inf")) if nums else None
    if t.startswith("<"):
        return (0.0, nums[0])
    if t.startswith(">"):
        return (nums[0], float("inf"))
    if len(nums) >= 2:
        return (nums[0], nums[1])
    return None


def poly_scan(poly, capital=100_000, max_size=None):
    ev = poly["events"]
    books = poly.get("books", {})
    groups = {}
    for slug, e in ev.items():
        groups.setdefault((e["asset"], e["day"]), {})[e["kind"]] = (slug, e)

    results = []
    for (asset, day), fam in sorted(groups.items()):
        legs, bounds = [], []
        for kind, (slug, e) in fam.items():
            for m in e["markets"]:
                bk = books.get(m["slug"])
                if not bk:
                    continue
                bd = poly_bounds(m["group_title"], kind)
                if not bd:
                    continue
                ls = poly_legs(m, bk, kind, bd, max_size)
                for lg in ls:
                    lg.meta["ladder"] = kind
                legs += ls
                for v in bd:
                    if v not in (0.0, float("inf")):
                        bounds.append(v)
        if len(legs) < 4:
            continue
        states = canonical_states(bounds)
        r = solve_arb(legs, states, capital=capital)
        if not r:
            continue
        results.append({
            "venue": "polymarket", "asset": asset, "event": f"aug{day}",
            "close": f"2026-08-{day:02d}T16:00:00Z",
            "n_legs": len(legs), "n_states": len(states),
            "profit": r["profit"], "outlay": r["outlay"], "roi": r["roi"],
            "picks": [(lg.key, lg.price, lg.fee, round(q, 2), lg.meta.get("ladder"),
                       lg.meta.get("group")) for lg, q in r["legs"]],
        })
    return results


def main():
    kal, poly = load("kalshi"), load("polymarket")
    out = kalshi_scan(kal) + poly_scan(poly)
    out.sort(key=lambda r: -r["profit"])
    save("arb_exact", out)
    for r in out:
        flag = "***" if r["profit"] > 1.0 else "   "
        print(f"{flag} {r['venue']:11s} {r['asset']} {r['event']:12s} "
              f"legs={r['n_legs']:4d} states={r['n_states']:4d} "
              f"profit=${r['profit']:12,.2f} outlay=${r['outlay']:12,.2f} "
              f"roi={(r['roi'] or 0)*100:7.3f}%")
    return out


if __name__ == "__main__":
    main()
