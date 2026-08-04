"""One-touch / barrier pricing by block-bootstrap Monte Carlo.

Kalshi's largest crypto open interest is not in the digital ladders at all --
it is 15.1 million contracts across the max/min series ("if the price is ever
above $X"). Those are barrier payoffs on the running extremum, and a closed
form Black-Scholes one-touch is the wrong tool: the answer is extremely
sensitive to the volatility assumption and to path behaviour (fat tails, vol
clustering) that a single sigma cannot express.

Method:
  * block-bootstrap hourly returns in 24-hour blocks, so vol clustering and
    autocorrelation survive the resampling;
  * carry each bar's high and low with its close return, so the running maximum
    reflects intra-bar extremes -- monitoring on closes alone materially
    underprices a barrier;
  * scale the whole ensemble so terminal variance matches the market's implied
    variance for that horizon, converting the real-world path shape to the
    risk-neutral level;
  * re-centre so the ensemble is a martingale on the forward.

Crucially the same engine also prices ordinary terminal digitals, so it can be
checked against Deribit and against Kalshi's own (liquid, well-arbitraged)
above/below ladders. A barrier price is only trusted when the terminal
validation passes.
"""
import datetime as dt
import math

import numpy as np

from common import load, save

YEAR_H = 365.25 * 24.0


def bars_arrays(bars):
    c = np.array([b["c"] for b in bars], float)
    h = np.array([b["h"] for b in bars], float)
    lo = np.array([b["l"] for b in bars], float)
    ok = (c > 0) & (h > 0) & (lo > 0)
    c, h, lo = c[ok], h[ok], lo[ok]
    r = np.diff(np.log(c))
    # intra-bar excursions relative to the bar's own close
    up = np.log(h[1:] / c[1:])
    dn = np.log(lo[1:] / c[1:])
    return r, np.maximum(up, 0.0), np.minimum(dn, 0.0)


class BarrierMC:
    """Bootstrapped path ensemble for one asset."""

    def __init__(self, bars, block=24, n_paths=40000, seed=7):
        self.r, self.up, self.dn = bars_arrays(bars)
        self.block = block
        self.n = n_paths
        self.rng = np.random.default_rng(seed)
        self.base_var_per_h = float(np.mean(self.r ** 2))

    def paths(self, hours, target_var=None, drift_to_forward=True):
        """(cum_max_logret, cum_min_logret, terminal_logret) for each path."""
        H = int(math.ceil(hours))
        nb = int(math.ceil(H / self.block))
        N = self.n
        starts = self.rng.integers(0, len(self.r) - self.block - 1, size=(N, nb))
        idx = (starts[:, :, None] + np.arange(self.block)[None, None, :])
        idx = idx.reshape(N, -1)[:, :H]
        R = self.r[idx]
        UP = self.up[idx]
        DN = self.dn[idx]

        if target_var is not None:
            realised = float(np.mean(np.sum(R, axis=1) ** 2))
            if realised > 0:
                s = math.sqrt(target_var / realised)
                R, UP, DN = R * s, UP * s, DN * s

        C = np.cumsum(R, axis=1)
        # running extremes including the intra-bar excursion of each bar
        hi = np.maximum.accumulate(C + UP, axis=1)[:, -1]
        lo = np.minimum.accumulate(C + DN, axis=1)[:, -1]
        term = C[:, -1]
        if drift_to_forward:
            # make the ensemble a martingale: E[exp(term)] = 1
            adj = math.log(float(np.mean(np.exp(term))))
            hi, lo, term = hi - adj, lo - adj, term - adj
        return hi, lo, term

    def cache_paths(self, hours):
        """Unscaled ensemble for a horizon, reusable across vol scenarios."""
        key = round(hours, 3)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key not in self._cache:
            self._cache[key] = self.paths(hours, None, drift_to_forward=False)
        return self._cache[key]

    def scaled(self, hours, target_var):
        """Rescale the cached ensemble to a target variance.

        Multiplying every log-return by a constant is exactly what changing the
        variance target does, so one ensemble serves every volatility scenario
        and the vol sensitivity band costs nothing extra.
        """
        hi, lo, term = self.cache_paths(hours)
        realised = float(np.mean(term ** 2))
        s = math.sqrt(target_var / realised) if realised > 0 else 1.0
        hi, lo, term = hi * s, lo * s, term * s
        adj = math.log(float(np.mean(np.exp(term))))
        return hi - adj, lo - adj, term - adj

    def price(self, S0, hours, target_var, barriers_up=(), barriers_dn=(),
              terminal=()):
        hi, lo, term = self.scaled(hours, target_var)
        out = {"touch_up": {}, "touch_dn": {}, "terminal_above": {}}
        for B in barriers_up:
            out["touch_up"][B] = float(np.mean(hi >= math.log(B / S0)))
        for B in barriers_dn:
            out["touch_dn"][B] = float(np.mean(lo <= math.log(B / S0)))
        for K in terminal:
            out["terminal_above"][K] = float(np.mean(term > math.log(K / S0)))
        return out


def implied_var(asset, hours, deribit_blob=None, now_ms=None, fallback_vol=None):
    """Total implied variance to a horizon, from Deribit where it exists."""
    if deribit_blob is not None:
        from rnd import TermSurface
        from fitladder import fit_ladder
        ts = TermSurface(deribit_blob, now_ms, asset)
        pts = []
        for e, s in sorted(ts.exps.items(), key=lambda x: x[1]["T"]):
            d = s["dens"]
            rows = [(float(K), max(d.prob_above(float(K)) - 1e-4, 0.0),
                     min(d.prob_above(float(K)) + 1e-4, 1.0))
                    for K in d.grid[::8]]
            f = fit_ladder(rows, s["T"], s["F"])
            if f:
                pts.append((s["T"] * YEAR_H, f["sigma_atm"] ** 2 * s["T"]))
        if pts:
            xs = [p[0] for p in pts]
            ws = [p[1] for p in pts]
            if hours <= xs[0]:
                return ws[0] * hours / xs[0]
            if hours >= xs[-1]:
                return ws[-1] * hours / xs[-1]
            return float(np.interp(hours, xs, ws))
    if fallback_vol:
        return fallback_vol ** 2 * hours / YEAR_H
    return None


def realised_vol(bars, window=720):
    c = np.array([b["c"] for b in bars], float)
    r = np.diff(np.log(c[c > 0]))
    return float(np.std(r[-window:]) * math.sqrt(YEAR_H))
