"""Measured Polymarket flow -- the number that decides which strategy wins.

The market-making case rested on an assumed fill rate. Polymarket publishes
every print, so it can be measured instead: pull the trade tape for each live
crypto ladder, count contracts actually traded in the last 24 hours, and split
them by whether the aggressor bought or sold.

A maker only earns on flow that crosses to them, so contracts/day is the
ceiling on the strategy -- no amount of capital raises it.
"""
import collections
import datetime as dt
import statistics
import time

from common import http_json, pmap, save

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"


def event(slug):
    try:
        d = http_json(f"{GAMMA}/events", {"slug": slug})
        return d[0] if d else None
    except Exception:                                    # noqa: BLE001
        return None


def trades(cid, limit=500):
    try:
        d = http_json(f"{DATA}/trades", {"market": cid, "limit": limit})
        return d if isinstance(d, list) else []
    except Exception:                                    # noqa: BLE001
        return []


def scan(slugs, hours=24):
    cut = time.time() - hours * 3600
    out = []
    evs = [e for _, e in pmap(event, slugs, workers=10) if isinstance(e, dict) and e]
    conds = []
    for e in evs:
        for m in e.get("markets", []):
            if m.get("closed") or not m.get("conditionId"):
                continue
            conds.append((e.get("slug"), m.get("groupItemTitle"), m["conditionId"]))
    for (slug, grp, cid), ts in pmap(lambda c: trades(c[2]), conds, workers=10):
        for t in ts or []:
            if t.get("timestamp", 0) < cut:
                continue
            out.append({"slug": slug, "bucket": grp,
                        "side": t.get("side"), "size": float(t.get("size", 0)),
                        "price": float(t.get("price", 0)),
                        "ts": t.get("timestamp")})
    return out


def main(hours=24):
    fams = {
        "ETH range": [f"ethereum-price-on-august-{d}-2026" for d in range(4, 12)],
        "BTC range": [f"bitcoin-price-on-august-{d}-2026" for d in range(4, 12)],
        "ETH above": [f"ethereum-above-on-august-{d}-2026" for d in range(4, 12)],
        "BTC above": [f"bitcoin-above-on-august-{d}-2026" for d in range(4, 12)],
        "BTC hit": ["what-price-will-bitcoin-hit-in-august-2026",
                    "what-price-will-bitcoin-hit-august-3-9-2026"],
        "ETH hit": ["what-price-will-ethereum-hit-in-august-2026",
                    "what-price-will-ethereum-hit-august-3-9-2026"],
    }
    print(f"MEASURED FLOW, last {hours}h (Polymarket trade tape)\n")
    print(f"{'family':12s} {'trades':>7s} {'contracts':>11s} {'notional':>11s} "
          f"{'med size':>9s} {'buy%':>6s}  maker capture at 1.45c")
    summary = {}
    for name, slugs in fams.items():
        rows = scan(slugs, hours)
        if not rows:
            print(f"{name:12s} {'-':>7s}")
            continue
        ct = sum(r["size"] for r in rows)
        notional = sum(r["size"] * r["price"] for r in rows)
        buys = sum(r["size"] for r in rows if r["side"] == "BUY")
        med = statistics.median([r["size"] for r in rows])
        # a maker sits on one side; assume it wins `share` of the flow
        for share in (0.15,):
            cap = ct * share * 0.0145
        print(f"{name:12s} {len(rows):7d} {ct:11,.0f} ${notional:10,.0f} "
              f"{med:9,.0f} {buys/ct*100:5.0f}% "
              f"  ${cap:,.2f}/day at 15% share")
        summary[name] = {"trades": len(rows), "contracts": ct,
                         "notional": notional, "buy_frac": buys / ct,
                         "maker_15pct_day": cap}

    tot_ct = sum(v["contracts"] for v in summary.values())
    tot_n = sum(v["notional"] for v in summary.values())
    print(f"\n{'TOTAL':12s} {sum(v['trades'] for v in summary.values()):7d} "
          f"{tot_ct:11,.0f} ${tot_n:10,.0f}")
    print(f"\n  A maker capturing 15% of ALL of it at 1.45c nets "
          f"${tot_ct*0.15*0.0145:,.2f}/day  (${tot_ct*0.15*0.0145*30:,.0f}/month)")
    print(f"  At 30% share:                                    "
          f"${tot_ct*0.30*0.0145:,.2f}/day  (${tot_ct*0.30*0.0145*30:,.0f}/month)")
    print("\n  Capital needed is peak inventory, not flow: at ~35c a contract and")
    print("  a one-day holding period, carrying even 20% of a day's flow is")
    print(f"  ${tot_ct*0.20*0.35:,.0f}.")
    save("polyflow", summary)
    return summary


if __name__ == "__main__":
    main()
