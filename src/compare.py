"""Head-to-head: Kalshi locked pairs vs Polymarket market making.

Both venues, one metric -- return per dollar per day -- with the risk class
stated rather than buried, and a capital allocation for any bankroll.

The comparison has to be careful about two things that flatter or penalise a
strategy unfairly:

  * capital for market making is not the total size quoted. Quotes are not
    capital until they fill, and inventory turns over. What is actually at
    risk is the peak inventory carried, which is far smaller than the depth
    posted -- costing this at full posted depth understated Polymarket badly
    in the first pass.
  * the locked pair's return is realised when the NEAR leg settles, not the
    far one, so its clock is 28 days rather than 150.
"""
import datetime as dt
import math

from common import load, save, now_ms


def kalshi_lock():
    rows = load("audit_s1")
    if not rows:
        return None
    now = now_ms()
    cap = sum(r["capital"] for r in rows)
    mn = sum(r["min_profit"] for r in rows)
    mx = sum(r["max_profit"] for r in rows)
    days = (max(r["near_close"] for r in rows) - now) / 86400_000
    return {
        "name": "Kalshi locked pairs",
        "risk": "none at settlement (legging risk on entry only)",
        "capital": cap,
        "min_ret": mn / cap,
        "exp_ret": None,       # filled from the model below
        "days": days,
        "ret_per_day": (mn / cap) / days,
        "capacity_note": "hard-capped by book depth",
    }


def poly_mm(inventory_frac=0.25, flow_share=0.15):
    """Polymarket ETH range MM, costed on inventory rather than posted depth.

    `inventory_frac` is the share of quoted size assumed to be carried as
    live inventory at any moment; `flow_share` is the fraction of the venue's
    traded flow won.
    """
    mm = load("mmmap")
    row = [r for r in mm if r["venue"] == "polymarket" and r["asset"] == "ETH"
           and r["ladder"] == "range"]
    if not row:
        return None
    r = row[0]
    poly = load("polymarket")

    depth = 0.0
    vol = 0.0
    for slug, e in poly["events"].items():
        if e["asset"] != "ETH" or e["kind"] != "range":
            continue
        for m in e["markets"]:
            b = poly["books"].get(m["slug"], {})
            bid = sum(q for p, q in b.get("bids", []))
            ask = sum(q for p, q in b.get("asks", []))
            depth += min(bid, ask)
            vol += m["volume"] or 0

    daily_flow = vol / 7.0                       # lifetime volume over ~a week
    contracts_won = daily_flow * flow_share / 0.35   # ~35c average contract
    pnl_day = contracts_won * r["net_30"]
    # capital = inventory actually carried, at ~35c a contract
    capital = depth * inventory_frac * 0.35
    return {
        "name": "Polymarket ETH range MM",
        "risk": "inventory: a gap move is a real loss",
        "capital": capital,
        "min_ret": None,
        "exp_ret": pnl_day * 30 / capital if capital else None,
        "days": 1.0,
        "ret_per_day": pnl_day / capital if capital else 0.0,
        "pnl_day": pnl_day,
        "daily_flow": daily_flow,
        "capacity_note": "capped by FLOW, not depth",
    }


def bnb_directional():
    rows = load("audit_s2")
    if not rows:
        return None
    cap = sum(r["capital"] for r in rows)
    ev = sum(r["ev"] for r in rows)
    days = 150.0
    return {
        "name": "BNB annual ladder (directional)",
        "risk": "FULL CAPITAL AT RISK",
        "capital": cap,
        "min_ret": -1.0,
        "exp_ret": ev / cap,
        "days": days,
        "ret_per_day": (ev / cap) / days,
        "capacity_note": "capped by book depth",
    }


def main(bankroll=10000.0):
    ks = kalshi_lock()
    pm = poly_mm()
    bn = bnb_directional()
    # expected return on the locked pair, including the $2 branch
    bs = load("barrier_scan")
    fair = {(r["event"], r["barrier"]): r["fair"] for r in bs}
    rows = load("audit_s1")
    ev = 0.0
    for r in rows:
        B = r["sell"].split("@")[1].replace(",", "")
        try:
            B = float(B)
        except ValueError:
            continue
        pm_ = fair.get(("KXBNBMAXMON-BNB-26AUG31", B))
        py_ = fair.get(("KXBNBMAXY-BNB-26DEC31", B))
        if pm_ is None or py_ is None:
            continue
        ev += r["qty"] * (1 + max(py_ - pm_, 0)) - r["capital"]
    if ks:
        ks["exp_ret"] = ev / ks["capital"]

    strats = [s for s in (ks, pm, bn) if s]
    print("=" * 80)
    print("HEAD TO HEAD — both venues, return per dollar per day")
    print("=" * 80)
    print(f"{'strategy':32s} {'capital':>10s} {'min ret':>9s} {'exp ret':>9s} "
          f"{'days':>6s} {'%/day':>8s} {'ann. exp':>10s}")
    for s in sorted(strats, key=lambda x: -(x["ret_per_day"] or 0)):
        mr = f"{s['min_ret']*100:+8.1f}%" if s["min_ret"] is not None else "        -"
        er = f"{s['exp_ret']*100:+8.1f}%" if s["exp_ret"] is not None else "        -"
        ann = ((1 + (s["exp_ret"] or 0)) ** (365 / max(s["days"], 1)) - 1) * 100
        print(f"{s['name']:32s} ${s['capital']:9,.0f} {mr} {er} "
              f"{s['days']:6.0f} {s['ret_per_day']*100:7.3f}% {ann:9,.0f}%")
    print()
    for s in strats:
        print(f"  {s['name']:32s} risk: {s['risk']}")
        print(f"  {'':32s} capacity: {s['capacity_note']}")

    print()
    print("=" * 80)
    print(f"ALLOCATION for a ${bankroll:,.0f} bankroll")
    print("=" * 80)
    left = bankroll
    lines = []
    if ks:
        take = min(ks["capital"], left)
        left -= take
        lines.append((ks["name"], take, take * ks["min_ret"], take * ks["exp_ret"],
                      "risk-free floor"))
    if pm:
        take = min(pm["capital"], left)
        left -= take
        lines.append((pm["name"], take, 0.0,
                      take * (pm["exp_ret"] or 0), "inventory risk"))
    if bn:
        take = min(bn["capital"], left)
        left -= take
        lines.append((bn["name"], take, -take, take * bn["exp_ret"],
                      "can lose it all"))
    for n, c, mn, e, note in lines:
        print(f"  {n:32s} ${c:9,.0f}   floor ${mn:9,.0f}   expected ${e:9,.0f}   {note}")
    print(f"  {'UNDEPLOYED (no edge available)':32s} ${left:9,.0f}")
    print()
    dep = bankroll - left
    print(f"  deployable: ${dep:,.0f} of ${bankroll:,.0f} "
          f"({dep/bankroll*100:.1f}%) -- the rest has nowhere to go")
    save("compare", {"strategies": strats, "allocation":
                     [{"name": n, "capital": c, "floor": mn, "expected": e}
                      for n, c, mn, e, _ in lines]})
    return strats


if __name__ == "__main__":
    main()
