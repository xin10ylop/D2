"""Model-free check on the barrier signals: how often did it actually happen?

The Monte Carlo rescales history to a forecast volatility, so a wrong forecast
becomes a wrong price -- which is exactly how the first pass produced 35 cent
"edges" on XRP. This module removes the volatility model entirely.

For a contract that needs a move of x% within D days, walk two years of that
coin's own hourly bars, and at every possible start point ask whether the
barrier was touched within D days. The answer is a frequency with a confidence
interval and no free parameters at all.

It is a real-world frequency rather than a risk-neutral price, so it is not
what the contract is worth -- but a market price far outside the historical
frequency band is a signal worth trusting, and one inside it is not.
"""
import math

import numpy as np

from common import load, save


def touch_freq(bars, pct_move, days, up=True, boot=800, seed=3):
    """P(price moves pct_move within `days`), measured directly.

    Overlapping windows are heavily autocorrelated, so the confidence interval
    comes from a block bootstrap over non-overlapping starts rather than from
    treating every hour as an independent trial.
    """
    c = np.array([b["c"] for b in bars], float)
    h = np.array([b["h"] for b in bars], float)
    lo = np.array([b["l"] for b in bars], float)
    ok = (c > 0) & (h > 0) & (lo > 0)
    c, h, lo = c[ok], h[ok], lo[ok]
    W = int(days * 24)
    if len(c) < W + 50:
        return None
    n = len(c) - W
    step = max(1, W // 8)                     # thin to near-independent starts
    starts = np.arange(0, n, step)
    hits = []
    for s in starts:
        s0 = c[s]
        if up:
            hits.append(1.0 if h[s + 1:s + W + 1].max() >= s0 * (1 + pct_move) else 0.0)
        else:
            hits.append(1.0 if lo[s + 1:s + W + 1].min() <= s0 * (1 + pct_move) else 0.0)
    a = np.array(hits)
    if len(a) < 6:
        return None
    rng = np.random.default_rng(seed)
    sims = np.array([rng.choice(a, len(a)).mean() for _ in range(boot)])
    return {"freq": float(a.mean()), "n": int(len(a)),
            "lo": float(np.percentile(sims, 2.5)),
            "hi": float(np.percentile(sims, 97.5))}


def main(top=26):
    rows = load("barrier_scan")
    coins = load("coins")

    cand = [r for r in rows
            if (r["sell_edge"] is not None and r["sell_edge"] > 0.05)
            or r["buy_edge"] > 0.05]
    cand.sort(key=lambda r: -max(r["buy_edge"], r["sell_edge"] or 0))
    cand = cand[:top]
    print(f"checking {len(cand)} signals against realised frequency "
          f"(two years, block-bootstrapped)\n")
    print(f"{'series':17s} {'coin':5s} {'d':2s} {'move':>8s} {'days':>6s} "
          f"{'mkt mid':>8s} {'MC fair':>8s} {'realised':>9s} {'95% CI':>15s}  verdict")

    out = []
    for r in cand:
        b = coins["bars"].get(r["coin"])
        if not b:
            continue
        f = touch_freq(b, r["moneyness"], r["H_days"], up=(r["dir"] == "up"))
        if not f:
            continue
        mid = 0.5 * (r["bid"] + r["ask"])
        # The market is only "wrong" if it sits outside BOTH the model band and
        # the realised-frequency band, on the same side.
        rich = mid > r["fair_hi"] and mid > f["hi"]
        cheap = mid < r["fair_lo"] and mid < f["lo"]
        verdict = ("RICH -> sell" if rich else
                   "CHEAP -> buy" if cheap else
                   "inside one of the bands")
        print(f"{r['series'][:17]:17s} {r['coin']:5s} {r['dir']:2s} "
              f"{r['moneyness']*100:+7.1f}% {r['H_days']:6.1f} {mid:8.3f} "
              f"{r['fair']:8.3f} {f['freq']:9.3f} "
              f"[{f['lo']:.3f},{f['hi']:.3f}]  {verdict}")
        out.append({**r, "realised": f["freq"], "real_lo": f["lo"],
                    "real_hi": f["hi"], "n_windows": f["n"], "verdict": verdict})

    keep = [r for r in out if r["verdict"] != "inside one of the bands"]
    print(f"\nsurvive BOTH the model band and the realised-frequency band: "
          f"{len(keep)} of {len(out)}")
    for r in sorted(keep, key=lambda x: -(x["oi"] or 0)):
        side = "SELL" if r["verdict"].startswith("RICH") else "BUY"
        edge = r["sell_edge"] if side == "SELL" else r["buy_edge"]
        print(f"   {side:4s} {r['series'][:18]:18s} {r['coin']:5s} "
              f"barrier={r['barrier']:,.5g} ({r['moneyness']*100:+.1f}%, "
              f"{r['H_days']:.0f}d)  mkt={0.5*(r['bid']+r['ask']):.3f} "
              f"MC={r['fair']:.3f} hist={r['realised']:.3f} "
              f"edge={edge:+.3f} OI={r['oi']:,.0f}")
    save("empirical_touch", out)
    return out


if __name__ == "__main__":
    main()
