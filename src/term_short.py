"""Does variance really accrue linearly in business time at short horizons?

The bot's largest apparent edges were on contracts under an hour from
settlement, priced by extrapolating Deribit's 12-hour option down to 30
minutes. That extrapolation assumes variance is linear in business time all
the way down. If it is not -- if short-horizon variance is sub-linear, as
microstructure mean-reversion would imply -- then the model overstates
short-dated volatility and the "edge" is a bug that systematically buys
out-of-the-money digitals and pays for it.

Measured here directly from 5-minute bars: realised variance over horizons
from 10 minutes to 12 hours, each normalised by the business time it spans.
A flat line means linear accrual and the extrapolation is safe.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from seasonality import Clock

YEAR_H = 365.25 * 24.0


def scaling(bars, clock, steps, bar_min=5):
    t = np.array([b["open_ms"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    ok = c > 0
    t, c = t[ok], c[ok]
    out = []
    for k in steps:
        if k >= len(c) // 3:
            continue
        r = np.log(c[k:] / c[:-k])
        tau = np.array([clock.tau(a, a + k * bar_min * 60_000) for a in t[:-k]])
        v = float(np.mean(r ** 2))
        tb = float(np.mean(tau))
        # bootstrap the variance estimate; overlapping windows inflate n, so
        # thin to independent blocks before resampling
        step = max(k, 1)
        ind = r[::step]
        rng = np.random.default_rng(0)
        sims = np.array([np.mean(rng.choice(ind, len(ind)) ** 2) for _ in range(1500)])
        out.append({"bars": k, "minutes": k * bar_min, "n_indep": len(ind),
                    "var": v, "tau_h": tb, "rate": v / tb,
                    "rate_lo": float(np.percentile(sims, 2.5)) / tb,
                    "rate_hi": float(np.percentile(sims, 97.5)) / tb})
    return out


def main():
    hist = load("hist")
    seas = load("seasonality")
    clock = Clock(seas["clock_hour"], seas["clock_dow"])
    steps = [2, 3, 6, 12, 24, 48, 96, 144]        # 10min .. 12h on 5m bars
    res = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        rows = scaling(hist["bars"][f"{sym}_5m"], clock, steps)
        base = [r for r in rows if r["minutes"] == 720]
        base = base[0]["rate"] if base else rows[-1]["rate"]
        res[asset] = {"rows": rows, "base_rate": base}
        print(f"\n=== {asset}: variance rate per business-hour, by horizon")
        print(f"   (1.00 = the 12-hour rate, which is what Deribit's front "
              f"expiry pins)")
        print(f"   {'horizon':>9s} {'n':>6s} {'var/biz-h':>11s} {'ratio':>7s} "
              f"{'95% CI':>16s}   vol scale")
        for r in rows:
            lab = (f"{r['minutes']}m" if r["minutes"] < 60
                   else f"{r['minutes']//60}h")
            ratio = r["rate"] / base
            print(f"   {lab:>9s} {r['n_indep']:6d} {r['rate']:11.3e} "
                  f"{ratio:7.3f} [{r['rate_lo']/base:6.2f},{r['rate_hi']/base:5.2f}]"
                  f"   x{math.sqrt(ratio):.3f}")
    save("term_short", res)
    return res


if __name__ == "__main__":
    main()
