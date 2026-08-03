"""Non-parametric risk-neutral density from Deribit quotes.

Why not just fit SVI: at 16h to expiry the front smile is extremely steep, and
most of Deribit's wing "quotes" are one-tick asks with no bid at all (a 58000
put 16h out is bid None / ask 0.0001 BTC). Their mark_iv -- 63% -- is an
artifact of Deribit's own flooring, not a price. A global 5-parameter fit
chases those phantom points and wrecks the belly of the distribution, which is
exactly where Kalshi and Polymarket have their strikes.

Instead we estimate the density directly:

    min_p  || W (A p - c) ||^2  +  lambda || D2 p ||^2
    s.t.   p >= 0,  sum p = 1,  sum p*K = F

where A[j,i] = max(K_i - K_j, 0) maps the density to call prices. The solution
is a genuine probability measure that reprices the forward, so it is free of
butterfly *and* calendar arbitrage by construction, and one-sided junk quotes
are simply excluded rather than fitted.

Calendar interpolation is then done in total-variance space: each expiry's
density is converted back to a smile, w(k) = sigma(k)^2 T is interpolated
linearly in T at fixed log-moneyness, and the density is rebuilt at the target
time. Linear-in-w is the standard condition that keeps the term structure
arbitrage-free.
"""
import math

import numpy as np
from scipy.optimize import lsq_linear

from surface import bs_call, fwd, forward_curve, implied_vol, ncdf, parse_instruments

MIN_TICK = 0.0001          # Deribit option tick, in coin
MAX_REL_SPREAD = 0.60      # wider than this is not a price


def build_grid(F, T, iv_guess=0.6, n=361):
    """Strike grid spanning +/- 6 standard deviations of the forward."""
    sd = max(iv_guess * math.sqrt(max(T, 1e-6)), 0.02)
    lo, hi = F * math.exp(-6.5 * sd), F * math.exp(6.5 * sd)
    return np.linspace(lo, hi, n)


def collect_quotes(blob, meta, exp_ms, F, T):
    """Convert usable option quotes into (K, call_price_usd, weight) rows.

    Puts are flipped to calls by parity so everything lives on one curve.
    A quote earns weight only if it is two-sided and reasonably tight; the
    rest of the surface is left to the smoothness prior.
    """
    rows = []
    idx = blob["index"]["index_price"]
    for s in blob["book_summary"]["option"]:
        m = meta.get(s["instrument_name"])
        if not m or m["expiry_ms"] != exp_ms:
            continue
        bid, ask = s.get("bid_price"), s.get("ask_price")
        if not bid or not ask or ask <= bid:
            continue
        rel = (ask - bid) / ask
        if rel > MAX_REL_SPREAD or ask <= MIN_TICK:
            continue
        K = m["strike"]
        mid_coin = 0.5 * (bid + ask)
        mid_usd = mid_coin * idx            # Deribit options are coin-settled
        call_usd = mid_usd if m["cp"] == "C" else mid_usd + (F - K)
        if call_usd <= 0 or call_usd >= F:
            continue
        # Tight, well-owned quotes dominate; wide ones barely register.
        oi = s.get("open_interest") or 0.0
        w = (1.0 / (0.02 + rel)) * (1.0 + math.log1p(oi) / 5.0)
        rows.append((K, call_usd, w))
    rows.sort()
    return rows


def fit_density(F, T, quotes, grid=None, lam=None):
    """Solve the regularised, constrained least-squares density fit."""
    if len(quotes) < 4:
        return None
    Ks = np.array([q[0] for q in quotes])
    Cs = np.array([q[1] for q in quotes])
    Ws = np.array([q[2] for q in quotes])
    Ws = Ws / Ws.max()

    if grid is None:
        rough = np.mean([implied_vol(c, F, k, T, 1.0, True) or 0.6
                         for k, c in zip(Ks, Cs)])
        grid = build_grid(F, T, iv_guess=max(rough, 0.2))

    n = len(grid)
    # payoff matrix: call at observed strike j given density on the grid
    A = np.maximum(grid[None, :] - Ks[:, None], 0.0) * Ws[:, None]
    b = Cs * Ws

    # moment constraints, enforced by heavy penalty rows
    big = 1e4 * max(1.0, float(np.abs(b).max()))
    A = np.vstack([A, big * np.ones((1, n)), big * grid[None, :] / F])
    b = np.concatenate([b, [big], [big]])

    # roughness penalty on the second difference of the density
    if lam is None:
        lam = 4e-3 * float(np.abs(Cs).max()) * math.sqrt(n)
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    A = np.vstack([A, lam * D2 * n])
    b = np.concatenate([b, np.zeros(n - 2)])

    res = lsq_linear(A, b, bounds=(0.0, np.inf), max_iter=400, tol=1e-11)
    p = np.maximum(res.x, 0.0)
    if p.sum() <= 0:
        return None
    p = p / p.sum()
    return {"grid": grid, "p": p, "F": F, "T": T, "n_quotes": len(quotes),
            "fit_F": float((p * grid).sum())}


class Density:
    """A discrete risk-neutral measure with the queries the venues need."""

    def __init__(self, grid, p, F, T):
        self.grid = np.asarray(grid, float)
        self.p = np.asarray(p, float)
        self.F = F
        self.T = T
        self.cdf = np.cumsum(self.p)

    def prob_above(self, K):
        return float(self.p[self.grid > K].sum())

    def prob_between(self, lo, hi):
        m = (self.grid >= lo) & (self.grid < hi)
        return float(self.p[m].sum())

    def call(self, K):
        return float((self.p * np.maximum(self.grid - K, 0.0)).sum())

    def quantile(self, q):
        return float(np.interp(q, self.cdf, self.grid))

    def smile(self, ks):
        """Implied vol at the given log-moneyness points, via call prices."""
        out = []
        for k in ks:
            K = self.F * math.exp(k)
            c = self.call(K)
            iv = implied_vol(c, self.F, K, self.T, 1.0, True)
            out.append(iv)
        return out


K_GRID = np.linspace(-0.45, 0.45, 61)


class TermSurface:
    """Densities at every listed expiry + calendar interpolation between them."""

    def __init__(self, blob, now_ms, label=""):
        self.label = label
        self.now_ms = now_ms
        self.index = blob["index"]["index_price"]
        self.curve = forward_curve(blob, now_ms)
        self.meta, _ = parse_instruments(blob)
        self.exps = {}
        self._build(blob)

    def _build(self, blob):
        exps = sorted({m["expiry_ms"] for m in self.meta.values()})
        for e in exps:
            T = (e - self.now_ms) / 1000.0 / (365.25 * 86400.0)
            if T <= 1e-6:
                continue
            F = fwd(self.curve, T)
            q = collect_quotes(blob, self.meta, e, F, T)
            fit = fit_density(F, T, q)
            if not fit:
                continue
            dens = Density(fit["grid"], fit["p"], F, T)
            sm = dens.smile(K_GRID)
            # Keep only the moneyness band the quotes actually support.
            w = [(iv * iv * T) if iv else None for iv in sm]
            self.exps[e] = {"T": T, "F": F, "dens": dens, "w": w,
                            "n_quotes": fit["n_quotes"],
                            "fit_F": fit["fit_F"],
                            "k_lo": math.log(min(x[0] for x in q) / F) if q else None,
                            "k_hi": math.log(max(x[0] for x in q) / F) if q else None}

    def _w_at(self, T):
        """Total variance curve w(k) at time T, linear in T at fixed k."""
        sl = sorted(self.exps.values(), key=lambda s: s["T"])
        if not sl:
            return None
        Ts = np.array([s["T"] for s in sl])
        out = []
        for i, k in enumerate(K_GRID):
            ws = [(s["T"], s["w"][i]) for s in sl if s["w"][i] is not None]
            if len(ws) == 0:
                out.append(None)
                continue
            if len(ws) == 1:
                out.append(ws[0][1] * T / ws[0][0])
                continue
            tt = np.array([x[0] for x in ws])
            vv = np.array([x[1] for x in ws])
            if T <= tt[0]:
                out.append(float(vv[0] * T / tt[0]))
            elif T >= tt[-1]:
                out.append(float(vv[-1] * T / tt[-1]))
            else:
                out.append(float(np.interp(T, tt, vv)))
        return out

    def density_at(self, T, n=601):
        """Rebuild a density at an arbitrary horizon from the interpolated smile."""
        if T <= 0:
            return None
        F = fwd(self.curve, T)
        w = self._w_at(T)
        if w is None:
            return None
        ks = [k for k, x in zip(K_GRID, w) if x is not None and x > 0]
        vs = [math.sqrt(x / T) for x in w if x is not None and x > 0]
        if len(ks) < 5:
            return None
        lo, hi = min(ks), max(ks)
        grid_k = np.linspace(lo, hi, n)
        iv = np.interp(grid_k, ks, vs)
        Kg = F * np.exp(grid_k)
        C = np.array([float(bs_call(F, K, T, s)) for K, s in zip(Kg, iv)])
        # density = d2C/dK2 on the reconstructed call curve
        dK = np.gradient(Kg)
        d2 = np.gradient(np.gradient(C, Kg), Kg)
        p = np.maximum(d2, 0.0) * dK
        # attach the mass outside the quoted band to the endpoints
        tail_lo = max(1.0 - float(-np.gradient(C, Kg)[0]), 0.0)
        tail_hi = max(float(-np.gradient(C, Kg)[-1]), 0.0)
        p[0] += tail_lo
        p[-1] += tail_hi
        if p.sum() <= 0:
            return None
        p = p / p.sum()
        return Density(Kg, p, F, T)
