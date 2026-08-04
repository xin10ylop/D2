"""Flow-weighted market-making economics -- what a maker would actually earn.

A median spread across a ladder is the wrong statistic. Polymarket's wing
buckets quote 9-39c wide on contracts worth 2-5c and almost never trade, while
the at-the-money bucket that carries the volume quotes 1-6c. Taking the median
over all of them produced an 80c "spread" and a fictitious $35k a month.

The honest calculation weights each contract's spread by the volume that
actually crossed on it, using Polymarket's own trade tape. A maker earns on
flow, so flow is the weight.
"""
import collections
import statistics
import time

from common import http_json, load, pmap, save

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
MODEL_ERR = 0.010


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


FAMS = {
    "ETH range": [f"ethereum-price-on-august-{d}-2026" for d in range(4, 12)],
    "BTC range": [f"bitcoin-price-on-august-{d}-2026" for d in range(4, 12)],
    "ETH above": [f"ethereum-above-on-august-{d}-2026" for d in range(4, 12)],
    "BTC above": [f"bitcoin-above-on-august-{d}-2026" for d in range(4, 12)],
    "BTC hit": ["what-price-will-bitcoin-hit-in-august-2026",
                "what-price-will-bitcoin-hit-august-3-9-2026"],
    "ETH hit": ["what-price-will-ethereum-hit-in-august-2026",
                "what-price-will-ethereum-hit-august-3-9-2026"],
}


def family(slugs, hours=24):
    """Per-contract quotes joined to the volume that crossed on them."""
    cut = time.time() - hours * 3600
    evs = [e for _, e in pmap(event, slugs, workers=10) if isinstance(e, dict) and e]
    meta = {}
    for e in evs:
        for m in e.get("markets", []):
            if m.get("closed") or not m.get("conditionId"):
                continue
            b, a = m.get("bestBid"), m.get("bestAsk")
            if b is None or a is None or a <= b:
                continue
            meta[m["conditionId"]] = {"spread": a - b, "mid": 0.5 * (a + b),
                                      "bucket": m.get("groupItemTitle")}
    rows = []
    for cid, ts in pmap(lambda c: trades(c), list(meta), workers=10):
        vol = sum(float(t.get("size", 0)) for t in (ts or [])
                  if t.get("timestamp", 0) >= cut)
        if vol <= 0:
            continue
        rows.append({**meta[cid], "vol": vol})
    return rows


def main(share=0.15, adverse=0.30, hours=24):
    print("FLOW-WEIGHTED market-making economics, Polymarket, last 24h")
    print(f"(maker wins {share:.0%} of flow, {adverse:.0%} of half-spread lost to")
    print(f" adverse selection, {MODEL_ERR*100:.0f}c model error, no Polymarket fee)\n")
    print(f"{'family':11s} {'contracts/d':>12s} {'med spr':>8s} {'flow-wtd':>9s} "
          f"{'half':>6s} {'net/ct':>7s} {'$/day':>9s} {'$/month':>9s}  verdict")
    out, tot = [], 0.0
    for name, slugs in FAMS.items():
        rows = family(slugs, hours)
        if not rows:
            continue
        vol = sum(r["vol"] for r in rows)
        med = statistics.median([r["spread"] for r in rows])
        fw = sum(r["spread"] * r["vol"] for r in rows) / vol
        half = fw / 2.0
        net = half - MODEL_ERR - adverse * half
        day = vol * share * net
        v = "QUOTE" if net > 0.004 else ("thin" if net > 0 else "SKIP")
        if net > 0:
            tot += day
        print(f"{name:11s} {vol:12,.0f} {med*100:7.1f}c {fw*100:8.1f}c "
              f"{half*100:5.1f}c {net*100:+6.2f}c ${day:8,.2f} ${day*30:8,.0f}  {v}")
        out.append({"family": name, "vol_day": vol, "median_spread": med,
                    "flow_wtd_spread": fw, "net_per_ct": net,
                    "day": day, "month": day * 30, "verdict": v})

    good = [r for r in out if r["net_per_ct"] > 0]
    ct = sum(r["vol_day"] for r in good) * share
    inv = ct * 0.35
    print(f"\n  profitable families: ${tot:,.2f}/day  (${tot*30:,.0f}/month)")
    print(f"  contracts won/day: {ct:,.0f}   inventory carried ~${inv:,.0f}")
    if inv > 0:
        print(f"  monthly return on inventory capital: {tot*30/inv*100:,.0f}%")
    save("mmflow", out)
    return out


if __name__ == "__main__":
    main()
