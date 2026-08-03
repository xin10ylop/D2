"""Where is market making actually profitable? The opportunity map.

Quoting only pays where the half-spread you capture exceeds what you lose to
being wrong. After recalibration the model's mean absolute error is about 1
cent, so the per-contract economics of resting a quote are

    edge = half_spread - model_error - maker_fee - adverse_selection

Adverse selection is the honest unknown: you are filled preferentially when
the market is moving against you. It is charged here as a fraction of the
half-spread, and the map is shown across a range of that assumption so the
conclusion can be read at whatever level of pessimism you prefer.

This is the map that says which books to point the bot at -- and, just as
importantly, which to leave alone.
"""
import statistics as st

import numpy as np

from common import load, save

MODEL_ERR = 0.010          # cents/contract, from src/recalib.py after fitting
KALSHI_MAKER = 0.0025
POLY_MAKER = 0.0


def spread_rows(kal, poly):
    rows = []
    for series, evs in kal["series"].items():
        asset = "ETH" if "ETH" in series else "BTC"
        kind = "range" if series in ("KXBTC", "KXETH") else "above"
        for ev, blob in evs.items():
            for m in blob["markets"]:
                b, a = m["yes_bid"], m["yes_ask"]
                if b is None or a is None or not (0 < a <= 1) or a <= b:
                    continue
                if not (0.05 <= 0.5 * (a + b) <= 0.95):
                    continue          # only the part of the ladder with real value
                bk = kal.get("books", {}).get(m["ticker"], {})
                depth = min(sum(s for _, s in bk.get("yes", [])),
                            sum(s for _, s in bk.get("no", [])))
                rows.append({"venue": "kalshi", "asset": asset, "ladder": kind,
                             "event": ev, "spread": a - b, "mid": 0.5 * (a + b),
                             "depth": depth, "maker_fee": KALSHI_MAKER,
                             "oi": m["open_interest"] or 0,
                             "vol24": m["volume_24h"] or 0})
    for slug, e in poly["events"].items():
        for m in e["markets"]:
            b, a = m["best_bid"], m["best_ask"]
            if b is None or a is None or a <= b:
                continue
            if not (0.05 <= 0.5 * (a + b) <= 0.95):
                continue
            bk = poly.get("books", {}).get(m["slug"], {})
            depth = min(sum(s for _, s in bk.get("bids", [])),
                        sum(s for _, s in bk.get("asks", [])))
            rows.append({"venue": "polymarket", "asset": e["asset"],
                         "ladder": e["kind"], "event": slug,
                         "spread": a - b, "mid": 0.5 * (a + b), "depth": depth,
                         "maker_fee": POLY_MAKER, "oi": 0,
                         "vol24": m["volume"] or 0})
    return rows


def main():
    kal, poly = load("kalshi"), load("polymarket")
    rows = spread_rows(kal, poly)
    print(f"quotable contracts (mid between 5c and 95c): {len(rows)}\n")

    groups = {}
    for r in rows:
        groups.setdefault((r["venue"], r["asset"], r["ladder"]), []).append(r)

    print(f"{'venue':11s} {'asset':4s} {'ladder':6s} {'n':>4s} {'med spr':>8s} "
          f"{'half':>7s} {'med depth':>10s} "
          f"{'net @0%':>8s} {'@30%':>7s} {'@50%':>7s}  verdict")
    out = []
    for k, g in sorted(groups.items(), key=lambda x: -st.median([r["spread"] for r in x[1]])):
        sp = st.median([r["spread"] for r in g])
        dp = st.median([r["depth"] for r in g])
        fee = g[0]["maker_fee"]
        half = sp / 2.0
        nets = {}
        for adv in (0.0, 0.30, 0.50):
            nets[adv] = half - MODEL_ERR - fee - adv * half
        v = ("GOOD" if nets[0.5] > 0.004 else
             "marginal" if nets[0.3] > 0.002 else "no")
        print(f"{k[0]:11s} {k[1]:4s} {k[2]:6s} {len(g):4d} {sp*100:7.2f}c "
              f"{half*100:6.2f}c {dp:10,.0f} "
              f"{nets[0.0]*100:7.2f}c {nets[0.3]*100:6.2f}c {nets[0.5]*100:6.2f}c  {v}")
        out.append({"venue": k[0], "asset": k[1], "ladder": k[2], "n": len(g),
                    "med_spread": sp, "med_depth": dp,
                    "net_0": nets[0.0], "net_30": nets[0.3], "net_50": nets[0.5],
                    "verdict": v})

    print(f"\n  model error charged: {MODEL_ERR*100:.1f}c  "
          f"(post-recalibration MAE from src/recalib.py)")
    print("  '@30%' = 30% of the half-spread lost to adverse selection.\n")

    # daily capacity: how much flow is there to trade against?
    print("flow available (Kalshi 24h volume, by ladder):")
    for k, g in sorted(groups.items()):
        if k[0] != "kalshi":
            continue
        v = sum(r["vol24"] for r in g)
        if v:
            print(f"   {k[1]} {k[2]:6s}: {v:12,.0f} contracts/24h")
    save("mmmap", out)
    return out


if __name__ == "__main__":
    main()
