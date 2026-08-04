"""Measure adverse selection from the trade tape, instead of assuming it.

The $440/month figure charged "30% of the half-spread" to adverse selection --
a number pulled from convention, not from data. It is the single most
important input, because a market maker is filled preferentially when it is
wrong, and if the post-fill drift exceeds the spread captured the strategy is
negative no matter how wide the quotes look.

Polymarket publishes every print, so this is measurable. For each trade at
price P and time t, find the volume-weighted price of trades on the SAME
contract in the window (t, t+H]. If the taker bought at P and the price then
rises, the maker who sold at P lost the difference. Averaged over the tape and
weighted by size, that is realised adverse selection in cents per contract.

Also checks the direction imbalance: 85-95% of prints showing as BUY would
mean flow is one-sided, and a maker quoting both sides gets filled only on the
losing side.
"""
import collections
import statistics
import time

import numpy as np

from common import http_json, load, pmap, save

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


def drift(ts, horizons=(300, 900, 3600)):
    """Size-weighted post-trade price drift, per contract, by horizon.

    Returns cents the maker loses per contract on the side it filled: if a
    taker BUYS (maker sold) and price later rises, the maker is down.
    """
    ts = sorted([t for t in ts if t.get("timestamp") and t.get("price")],
                key=lambda x: x["timestamp"])
    if len(ts) < 10:
        return None
    out = {}
    for H in horizons:
        loss, wt = 0.0, 0.0
        for i, t in enumerate(ts):
            t0 = t["timestamp"]
            p0 = float(t["price"])
            sz = float(t.get("size", 0))
            fut = [(float(x["price"]), float(x.get("size", 0)))
                   for x in ts[i + 1:] if t0 < x["timestamp"] <= t0 + H]
            if not fut or sz <= 0:
                continue
            fv = sum(p * s for p, s in fut) / max(sum(s for _, s in fut), 1e-9)
            # taker BUY  => maker sold at p0; maker loses if price rises
            # taker SELL => maker bought at p0; maker loses if price falls
            d = (fv - p0) if t.get("side") == "BUY" else (p0 - fv)
            loss += d * sz
            wt += sz
        if wt > 0:
            out[H] = loss / wt
    return out


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


def main(hours=24, lo=0.10, hi=0.90, max_spread=0.10, min_vol=500):
    """Restricted to contracts a real maker would quote.

    A 2-cent wing contract can double on one print, so including the wings
    makes drift look enormous for reasons that have nothing to do with being
    picked off. Restricting to mid-priced contracts is the fair test."""
    cut = time.time() - hours * 3600
    print("MEASURED ADVERSE SELECTION (post-trade drift on Polymarket's own tape)\n")
    print(f"{'family':11s} {'contracts':>11s} {'spr(fw)':>8s} {'buy%':>6s} "
          f"{'drift 5m':>9s} {'drift 15m':>10s} {'drift 60m':>10s} {'top ct%':>8s}")
    out = {}
    for name, slugs in FAMS.items():
        evs = [e for _, e in pmap(event, slugs, workers=10)
               if isinstance(e, dict) and e]
        meta = {}
        for e in evs:
            for m in e.get("markets", []):
                if m.get("closed") or not m.get("conditionId"):
                    continue
                b, a = m.get("bestBid"), m.get("bestAsk")
                if b is None or a is None or a <= b:
                    continue
                mid = 0.5 * (a + b)
                # a real market: priced in the belly AND actually quoted tight.
                # A 0.02/0.98 book has a mid of 0.50 but is not a market.
                if not (lo <= mid <= hi) or (a - b) > max_spread:
                    continue
                meta[m["conditionId"]] = {"spread": a - b, "mid": mid,
                                          "bucket": m.get("groupItemTitle")}
        per, allvol, buyvol = [], 0.0, 0.0
        dr = collections.defaultdict(list)
        for cid, ts in pmap(trades, list(meta), workers=10):
            ts = [t for t in (ts or []) if t.get("timestamp", 0) >= cut]
            v = sum(float(t.get("size", 0)) for t in ts)
            if v < min_vol:
                continue
            allvol += v
            buyvol += sum(float(t.get("size", 0)) for t in ts if t.get("side") == "BUY")
            per.append((meta[cid]["spread"], v, meta[cid]["bucket"]))
            d = drift(ts)
            if d:
                for H, val in d.items():
                    dr[H].append((val, v))
        if not per or allvol <= 0:
            continue
        fw = sum(s * v for s, v, _ in per) / allvol
        top = max(v for _, v, _ in per) / allvol
        row = {"vol": allvol, "fw_spread": fw, "buy_frac": buyvol / allvol,
               "top_contract_share": top}
        for H in (300, 900, 3600):
            if dr[H]:
                w = sum(v for _, v in dr[H])
                row[f"drift_{H}"] = sum(d * v for d, v in dr[H]) / w
        print(f"{name:11s} {allvol:11,.0f} {fw*100:7.2f}c {row['buy_frac']*100:5.0f}% "
              f"{row.get('drift_300',0)*100:+8.2f}c {row.get('drift_900',0)*100:+9.2f}c "
              f"{row.get('drift_3600',0)*100:+9.2f}c {top*100:7.0f}%")
        out[name] = row

    print("\n  drift = cents the MAKER loses per contract after being filled.")
    print("  Positive drift means the taker was right and the maker was picked off.")
    print("  A family only pays if half-spread - model error - drift > 0.\n")
    print(f"{'family':11s} {'half spr':>9s} {'model':>7s} {'drift15':>8s} "
          f"{'NET/ct':>8s}   verdict")
    for name, r in out.items():
        half = r["fw_spread"] / 2
        d15 = r.get("drift_900", 0.0)
        net = half - 0.010 - max(d15, 0.0)
        print(f"{name:11s} {half*100:8.2f}c {1.0:6.1f}c {d15*100:+7.2f}c "
              f"{net*100:+7.2f}c   {'QUOTE' if net > 0.002 else 'SKIP'}")
    save("adverse", out)
    return out


if __name__ == "__main__":
    main()
