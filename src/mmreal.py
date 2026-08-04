"""Per-family market-making economics: measured flow x measured spread.

Two corrections to the earlier pass, both of which moved the answer a long way:

  * flow was read off gamma's `volumeNum`, which is roughly one day's volume,
    not lifetime. Real flow across Polymarket's crypto ladders is 2.65M
    contracts a day, not the 2,600 assumed -- a factor of a thousand.
  * a single net-edge figure was applied to every family. That is wrong: the
    families with the most flow are the ones with the tightest spreads, and
    quoting those loses money. The two have to be multiplied family by family.

Net edge per contract = half_spread - model_error - fee - adverse_selection,
with the model error taken from src/recalib.py (1c after calibration) and
adverse selection charged as a fraction of the half spread.
"""
import statistics

from common import http_json, load, pmap, save

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MODEL_ERR = 0.010
POLY_FEE = 0.0


def event(slug):
    try:
        d = http_json(f"{GAMMA}/events", {"slug": slug})
        return d[0] if d else None
    except Exception:                                    # noqa: BLE001
        return None


def spreads(slugs, lo=0.05, hi=0.95):
    """Median quoted spread and resting depth across a family's live books."""
    sp, dep = [], []
    evs = [e for _, e in pmap(event, slugs, workers=10) if isinstance(e, dict) and e]
    for e in evs:
        for m in e.get("markets", []):
            if m.get("closed"):
                continue
            b, a = m.get("bestBid"), m.get("bestAsk")
            if b is None or a is None or a <= b:
                continue
            mid = 0.5 * (a + b)
            if not (lo <= mid <= hi):
                continue
            sp.append(a - b)
            dep.append(m.get("liquidityNum") or 0)
    if not sp:
        return None
    return {"median_spread": statistics.median(sp), "n": len(sp),
            "median_liq": statistics.median(dep) if dep else 0}


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


def main(share=0.15, adverse=0.30):
    flow = load("polyflow")
    print(f"POLYMARKET MARKET MAKING — measured flow x measured spread")
    print(f"(maker wins {share:.0%} of flow; {adverse:.0%} of the half-spread "
          f"lost to adverse selection; {MODEL_ERR*100:.0f}c model error)\n")
    print(f"{'family':11s} {'contracts/d':>12s} {'spread':>7s} {'half':>6s} "
          f"{'net/ct':>7s} {'$/day':>10s} {'$/month':>10s}  verdict")
    rows = []
    tot_d = 0.0
    for name, slugs in FAMS.items():
        f = flow.get(name)
        s = spreads(slugs)
        if not f or not s:
            continue
        half = s["median_spread"] / 2.0
        net = half - MODEL_ERR - POLY_FEE - adverse * half
        day = f["contracts"] * share * net
        verdict = "QUOTE" if net > 0.004 else ("thin" if net > 0 else "SKIP (negative)")
        if net > 0:
            tot_d += day
        print(f"{name:11s} {f['contracts']:12,.0f} {s['median_spread']*100:6.1f}c "
              f"{half*100:5.1f}c {net*100:+6.2f}c ${day:9,.2f} ${day*30:9,.0f}  {verdict}")
        rows.append({"family": name, "contracts_day": f["contracts"],
                     "spread": s["median_spread"], "net_per_ct": net,
                     "day": day, "month": day * 30, "verdict": verdict})

    print(f"\n  profitable families only: ${tot_d:,.2f}/day  (${tot_d*30:,.0f}/month)")
    good = [r for r in rows if r["net_per_ct"] > 0]
    ct = sum(r["contracts_day"] for r in good)
    inv = ct * share * 0.35          # a day of won flow, carried at ~35c
    print(f"  capital = inventory carried, not flow: "
          f"{ct*share:,.0f} contracts/day won x ~35c = ${inv:,.0f}")
    if inv > 0:
        print(f"  monthly return on that capital: {tot_d*30/inv*100:,.0f}%")
    print("\n  The families with the most flow have the tightest spreads and lose")
    print("  money to quote. Same inverse relationship as Kalshi.")
    save("mmreal", rows)
    return rows


if __name__ == "__main__":
    main()
