"""Hour-of-week variance clock -- no separability assumption.

The separable clock multiplies an hour-of-day profile by a day-of-week profile.
That is only right if the two effects are independent, and they are probably
not: Monday's elevated variance plausibly lives in the Asian re-open after the
weekend gap rather than being spread evenly across Monday's 24 hours. Where the
two disagree, the bot's fair value is wrong -- and it was disagreeing worst
exactly where the bot found its biggest "edges".

So estimate the rate directly on all 168 hours of the week. Two years gives
~104 observations per bucket, the same sample the day-of-week result rests on.
The raw estimate is then shrunk toward the separable model, because 104
observations of a squared return is a noisy variance estimate and the separable
model is a sensible prior.
"""
import datetime as dt
import math

import numpy as np

from common import load, save

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def how_index(ts_ms):
    u = dt.datetime.utcfromtimestamp(ts_ms / 1000)
    return u.weekday() * 24 + u.hour


def raw_profile(bars):
    t = np.array([b["open_ms"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    ok = c > 0
    t, c = t[ok], c[ok]
    r = np.diff(np.log(c))
    idx = np.array([how_index(x) for x in t[1:]])
    base = float(np.mean(r ** 2))
    prof = np.ones(168)
    cnt = np.zeros(168)
    for i in range(168):
        s = idx == i
        cnt[i] = s.sum()
        if s.sum() > 5:
            prof[i] = float(np.mean(r[s] ** 2)) / base
    return prof, cnt, r, idx


def smooth_week(p, k=(0.15, 0.35, 0.7, 1.0, 0.7, 0.35, 0.15)):
    """Circular smoothing across the week -- adjacent hours are not independent."""
    k = np.array(k, float)
    k /= k.sum()
    pad = len(k) // 2
    x = np.r_[p[-pad:], p, p[:pad]]
    return np.convolve(x, k, mode="same")[pad:-pad]


def shrink(raw, prior, n, n0=60.0):
    """James-Stein style shrinkage toward the separable prior."""
    w = n / (n + n0)
    return w * raw + (1 - w) * prior


class WeekClock:
    """Variance clock on the 168-hour week."""

    def __init__(self, profile):
        p = np.asarray(profile, float)
        self.p = p / p.mean()

    def rate(self, ms):
        return float(self.p[how_index(ms)])

    def tau(self, t0, t1, sub=4):
        if t1 <= t0:
            return 0.0
        step = 3600_000.0 / sub
        acc, a = 0.0, float(t0)
        while a < t1:
            b = min(a + step, t1)
            acc += self.rate(a) * (b - a) / 3600_000.0
            a = b
        return acc


def main():
    hist = load("hist")
    seas = load("seasonality")
    out = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        prof, cnt, r, idx = raw_profile(hist["bars"][f"{sym}_1h"])
        hod = np.array(seas[asset]["hour_smooth"] if asset in seas
                       else seas["clock_hour"], float)
        dow = np.array(seas[asset]["dow"] if asset in seas else seas["clock_dow"], float)
        hod = hod / hod.mean()
        dow = dow / dow.mean()
        prior = np.array([dow[i // 24] * hod[i % 24] for i in range(168)])
        sm = smooth_week(prof)
        blend = shrink(sm, prior, cnt)
        blend = blend / blend.mean()
        out[asset] = {"raw": prof.tolist(), "smooth": sm.tolist(),
                      "prior_separable": prior.tolist(), "clock": blend.tolist(),
                      "n": cnt.tolist()}

        print(f"\n=== {asset}: separable prior vs measured hour-of-week")
        print(f"   n per bucket ~ {int(np.median(cnt))}")
        diff = blend / prior
        worst = np.argsort(-np.abs(np.log(diff)))[:10]
        print(f"   {'slot':10s} {'separable':>10s} {'measured':>10s} {'ratio':>7s}")
        for i in sorted(worst):
            print(f"   {DOW[i//24]} {i%24:02d}Z   {prior[i]:10.3f} "
                  f"{blend[i]:10.3f} {diff[i]:7.2f}")
        # the specific slot the bot was mispricing
        for lbl, i in (("Mon 20Z", 0 * 24 + 20), ("Mon 21Z", 0 * 24 + 21),
                       ("Mon 15Z", 0 * 24 + 15), ("Tue 14Z", 1 * 24 + 14)):
            print(f"   {lbl}: separable {prior[i]:.3f} -> measured {blend[i]:.3f} "
                  f"({diff[i]:.2f}x)")
    save("howclock", out)
    return out


if __name__ == "__main__":
    main()
