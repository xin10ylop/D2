"""Correct pricer for Kalshi's max/min series: a running TRIMMED-MEAN barrier.

The audit of the settlement rules overturned the earlier treatment. These are
not one-touch contracts on spot. The rule is:

    at each minute t, take every minute price from ISSUANCE to t, discard the
    top 20% and the bottom 20% of that cumulative dataset, average the rest;
    if that trimmed mean ever exceeds the threshold, the market pays.

Three consequences, all of which invalidate the previous pass:

  * a spike does not trigger it -- the spike lands in the discarded top 20%;
    the price has to move and then *stay* long enough to drag the middle 60%
    of the whole history past the barrier;
  * the measurement window starts at ISSUANCE, so the already-observed prices
    are part of the statistic and must be carried as a fixed prefix;
  * an earlier-issued contract is therefore *harder* to trigger on the upside
    than a later-issued one at the same barrier, because it carries more old,
    lower prices. The apparent "calendar arbitrage" between the annual ladder
    (issued 22 Jul) and the monthly ladder (issued 1 Aug) was this effect, not
    a mispricing, and the implication that made it risk-free does not hold.

Simulation: observed prefix + block-bootstrapped future, running trimmed mean
evaluated on a grid of checkpoints, maximum compared with the barrier.
"""
import datetime as dt
import math

import numpy as np

from common import load, save

YEAR_H = 365.25 * 24.0


def prefix_prices(bars, t0_ms, t1_ms=None):
    """Observed hourly closes from issuance to now -- the fixed part of the set."""
    t1_ms = t1_ms or bars[-1]["open_ms"]
    return np.array([b["c"] for b in bars
                     if t0_ms <= b["open_ms"] <= t1_ms and b["c"] > 0], float)


def running_trimmed_max(prefix, future, checkpoints=40, trim=0.2, mode="max"):
    """Max (or min) over time of the trimmed mean of prefix+future[:k].

    `future` is (n_paths, steps) of prices. Evaluating on a grid of checkpoints
    rather than every step is a deliberate approximation: the trimmed mean of a
    cumulative sample moves slowly, and sampling 40 points understates the
    extreme only slightly -- which is conservative for an up-barrier.
    """
    n, steps = future.shape
    if steps < 2:
        return np.full(n, np.nan)
    ks = np.unique(np.linspace(1, steps, min(checkpoints, steps)).astype(int))
    best = np.full(n, -np.inf if mode == "max" else np.inf)
    pre = np.asarray(prefix, float)
    for k in ks:
        # cumulative sample at this checkpoint, one row per path
        block = np.concatenate(
            [np.repeat(pre[None, :], n, axis=0), future[:, :k]], axis=1)
        m = block.shape[1]
        loi, hii = int(math.floor(m * trim)), int(math.ceil(m * (1 - trim)))
        if hii - loi < 2:
            continue
        part = np.sort(block, axis=1)[:, loi:hii]
        tm = part.mean(axis=1)
        best = np.maximum(best, tm) if mode == "max" else np.minimum(best, tm)
    return best


def to_daily(bars):
    """Aggregate hourly bars to daily, for horizons where hourly is unaffordable."""
    out, cur = [], None
    for b in bars:
        d = dt.datetime.utcfromtimestamp(b["open_ms"] / 1000).date()
        if cur is None or cur["d"] != d:
            if cur:
                out.append(cur)
            cur = {"d": d, "open_ms": b["open_ms"], "o": b["o"], "h": b["h"],
                   "l": b["l"], "c": b["c"]}
        else:
            cur["h"] = max(cur["h"], b["h"])
            cur["l"] = min(cur["l"], b["l"])
            cur["c"] = b["c"]
    if cur:
        out.append(cur)
    return out


class TrimMeanPricer:
    def __init__(self, bars, n_paths=4000, block=24, seed=11):
        from onetouch import bars_arrays
        self.bars = bars
        self.r, self.up, self.dn = bars_arrays(bars)
        self.dr, _, _ = bars_arrays(to_daily(bars))
        self.n = n_paths
        self.block = block
        self.rng = np.random.default_rng(seed)

    def future_paths(self, S0, hours, target_var, max_steps=400):
        """Simulated forward prices, coarsened so the sort stays affordable."""
        daily = hours > 24 * 25
        r = self.dr if daily else self.r
        steps = int(math.ceil(hours / 24.0)) if daily else int(math.ceil(hours))
        stride = max(1, steps // max_steps)
        eff = steps // stride
        block = max(1, self.block // stride) if not daily else 5
        nb = int(math.ceil(eff / block))
        starts = self.rng.integers(0, len(r) - block * stride - 1,
                                   size=(self.n, nb))
        idx = starts[:, :, None] + (np.arange(block) * stride)[None, None, :]
        idx = idx.reshape(self.n, -1)[:, :eff]
        # aggregate `stride` consecutive returns into each coarse step
        R = r[idx] * math.sqrt(stride)
        realised = float(np.mean(np.sum(R, axis=1) ** 2))
        if realised > 0 and target_var:
            R *= math.sqrt(target_var / realised)
        C = np.cumsum(R, axis=1)
        C -= math.log(float(np.mean(np.exp(C[:, -1]))))
        return S0 * np.exp(C)

    def price(self, S0, hours, target_var, prefix, barriers_up=(), barriers_dn=()):
        fut = self.future_paths(S0, hours, target_var)
        out = {"up": {}, "dn": {}}
        if barriers_up:
            mx = running_trimmed_max(prefix, fut, mode="max")
            for B in barriers_up:
                out["up"][B] = float(np.mean(mx > B))
        if barriers_dn:
            mn = running_trimmed_max(prefix, fut, mode="min")
            for B in barriers_dn:
                out["dn"][B] = float(np.mean(mn < B))
        return out

    def current_trimmed_mean(self, prefix, trim=0.2):
        p = np.sort(np.asarray(prefix, float))
        m = len(p)
        lo, hi = int(math.floor(m * trim)), int(math.ceil(m * (1 - trim)))
        return float(p[lo:hi].mean()) if hi > lo else float("nan")
