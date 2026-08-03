"""Fair value for a Kalshi digital ladder, anchored on Deribit, on the business clock.

The chain, once per refresh:

  1. Deribit index + front futures  -> forward F(t) at the settlement instant.
  2. Deribit front option expiries   -> total implied variance w at each expiry.
  3. Business clock                  -> tau(now, expiry) in variance-hours.
     kappa = w / tau is then the variance *rate*, which is the object that is
     actually stable through the day. Multiplying kappa by tau(now, settlement)
     gives the right variance for a contract ending at any hour -- which is the
     whole edge, since a flat-clock quoter uses calendar hours here and is
     wrong by -26% at 11Z and +36% at 15Z.
  4. Deribit smile                   -> skew (beta) and curvature (gamma).
  5. P(S > K) = -dC/dK on that smile, per strike.

Everything is a pure function of a market snapshot, so it is trivially testable
and cannot silently pick up state from a previous loop.
"""
import bisect
import datetime as dt
import math
import sys

import numpy as np

sys.path.insert(0, "/home/user/D2/src")

from fitladder import fit_ladder, model_cdf          # noqa: E402
from scipy.special import ndtr, ndtri                 # noqa: E402
from rnd import TermSurface                          # noqa: E402
from seasonality import Clock                        # noqa: E402
from surface import fwd                              # noqa: E402

YEAR_H = 365.25 * 24.0

# Sharpening coefficients fitted in src/recalib.py on two years of outcomes.
# The raw model prices a distribution wider than reality -- predicted-31%
# events happen 26% of the time -- and this pulls probabilities back toward
# 0/1 by the measured amount. Without it the bot "finds" 4-7c of edge on
# short-dated contracts that is purely its own misspecification.
RECAL_B = {
    "BTC": [(1, 1.215), (2, 1.210), (3, 1.211), (6, 1.196), (12, 1.154), (24, 1.121)],
    "ETH": [(1, 1.267), (2, 1.245), (3, 1.237), (6, 1.198), (12, 1.150), (24, 1.118)],
}


def sharpen_b(asset, cal_h, damp=0.5):
    """Interpolate b in log-horizon; damp it because the Deribit smile already
    supplies part of the kurtosis the lognormal fit was missing."""
    tab = RECAL_B.get(asset) or RECAL_B["BTC"]
    xs = [math.log(h) for h, _ in tab]
    ys = [b for _, b in tab]
    b = float(np.interp(math.log(max(cal_h, 0.25)), xs, ys))
    return 1.0 + damp * (b - 1.0)


def recalibrate(p, b):
    if b is None or abs(b - 1.0) < 1e-9:
        return p
    z = ndtri(min(max(p, 1e-9), 1 - 1e-9))
    return float(ndtr(b * z))


class Pricer:
    """Snapshot pricer. Rebuild it whenever the Deribit surface is refreshed."""

    def __init__(self, deribit_blob, now_ms, clock: Clock, asset: str):
        self.asset = asset
        self.now_ms = now_ms
        self.clock = clock
        self.ts = TermSurface(deribit_blob, now_ms, asset)
        self.index = self.ts.index
        self.anchors = []
        for e, s in sorted(self.ts.exps.items(), key=lambda x: x[1]["T"]):
            tau = clock.tau(now_ms, e)
            if tau <= 0:
                continue
            f = self._fit(s)
            if not f:
                continue
            self.anchors.append({
                "exp_ms": e, "tau": tau, "T": s["T"], "F": s["F"],
                "w": f["sigma_atm"] ** 2 * s["T"],
                "beta": f["beta"], "gamma": f["gamma"],
            })
        self.ok = len(self.anchors) >= 2

    def _fit(self, s):
        d = s["dens"]
        rows = [(float(K), max(d.prob_above(float(K)) - 1e-4, 0.0),
                 min(d.prob_above(float(K)) + 1e-4, 1.0)) for K in d.grid[::8]]
        return fit_ladder(rows, s["T"], s["F"])

    # -------------------------------------------------------------- rates

    def kappa(self, tau_h):
        """Variance per business-hour at horizon tau_h (interpolated in tau)."""
        if not self.anchors:
            return None
        xs = [a["tau"] for a in self.anchors]
        ws = [a["w"] for a in self.anchors]
        if tau_h <= xs[0]:
            # Short of the front expiry the rate is flat in business time --
            # which is exactly the assumption the clock is designed to make safe.
            return ws[0] / xs[0]
        if tau_h >= xs[-1]:
            return ws[-1] / xs[-1]
        return float(np.interp(tau_h, xs, ws)) / tau_h

    def skew(self, tau_h):
        xs = [a["tau"] for a in self.anchors]
        b = [a["beta"] for a in self.anchors]
        g = [a["gamma"] for a in self.anchors]
        return float(np.interp(tau_h, xs, b)), float(np.interp(tau_h, xs, g))

    def forward(self, settle_ms):
        T = max((settle_ms - self.now_ms) / 1000.0 / (365.25 * 86400.0), 0.0)
        return fwd(self.ts.curve, T)

    # -------------------------------------------------------------- prices

    def state(self, settle_ms):
        """All the pricing state for one settlement instant."""
        tau = self.clock.tau(self.now_ms, settle_ms)
        cal_h = (settle_ms - self.now_ms) / 3600_000.0
        if tau <= 0 or cal_h <= 0 or not self.ok:
            return None
        k = self.kappa(tau)
        if not k or k <= 0:
            return None
        w = k * tau
        T = tau / YEAR_H                       # business-time year fraction
        beta, gamma = self.skew(tau)
        return {"tau_h": tau, "cal_h": cal_h, "w": w, "T": T,
                "sigma": math.sqrt(w / T), "beta": beta, "gamma": gamma,
                "b": sharpen_b(self.asset, cal_h),
                "front_tau": self.anchors[0]["tau"] if self.anchors else None,
                "extrapolated": bool(self.anchors and tau < self.anchors[0]["tau"]),
                "F": self.forward(settle_ms),
                # what a flat-clock quoter would use, for edge attribution
                "sigma_flat": math.sqrt(k * cal_h / (cal_h / YEAR_H))}

    def prob_above(self, K, st):
        raw = float(model_cdf([K], st["F"], st["T"], st["sigma"],
                              st["beta"], st["gamma"])[0])
        return recalibrate(raw, st.get("b"))

    def prob_between(self, lo, hi, st):
        a = self.prob_above(lo, st) if lo and lo > 0 else 1.0
        b = self.prob_above(hi, st) if hi and math.isfinite(hi) else 0.0
        return max(a - b, 0.0)

    def delta_per_contract(self, K, st, side="YES"):
        """dP/dS in probability per dollar of underlying -- the hedge ratio."""
        h = max(st["F"] * 2e-4, 1.0)
        d = (self.prob_above(K + h, st) - self.prob_above(K - h, st)) / (2 * h)
        return d if side == "YES" else -d

    def flat_clock_prob(self, K, st):
        """What a calendar-time quoter would price. The gap is our edge."""
        Tf = st["cal_h"] / YEAR_H
        kap = st["w"] / st["tau_h"]
        sf = math.sqrt(kap * st["cal_h"] / Tf)
        return float(model_cdf([K], st["F"], Tf, sf, st["beta"], st["gamma"])[0])
