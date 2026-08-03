"""Parametric fit of a quoted digital ladder to a smooth, skewed distribution.

Reading quantiles by interpolating a coarse ladder linearly is badly biased.
Polymarket lists BTC strikes 2,000 apart while the 16-84 range of a one-day
distribution is about +/-950 -- narrower than a single strike interval -- so
linear interpolation across the gaps inflates the implied spread by ~50% and
makes Polymarket look far wider than Deribit when it is not.

Instead we fit a smooth risk-neutral distribution to the quotes:

    sigma(k) = sigma0 * exp(beta*k + gamma*k^2),   k = ln(K/F)
    C(K)     = Black76(F, K, T, sigma(k))
    P(S>K)   = -dC/dK          (skew-aware, not the flat-vol N(d2))

with (F, sigma0, beta, gamma) fitted by weighted least squares to the quoted
digitals, each weighted by the tightness of its own bid/ask. The same
functional is used for every venue, so the resulting sigma_atm numbers are
directly comparable across Kalshi, Polymarket and Deribit.
"""
import math

import numpy as np
from scipy.optimize import least_squares

from surface import bs_call

YEAR_H = 365.25 * 24.0


def model_cdf(Ks, F, T, s0, beta, gamma):
    """P(S>K) on the fitted smile, by differencing the call curve."""
    Ks = np.asarray(Ks, float)
    h = np.maximum(Ks * 3e-4, 0.25)

    def c(x):
        k = np.log(x / F)
        sig = s0 * np.exp(beta * k + gamma * k * k)
        return bs_call(F, x, T, sig)

    p = -(c(Ks + h) - c(Ks - h)) / (2 * h)
    return np.clip(p, 1e-9, 1 - 1e-9)


def fit_ladder(quotes, T, F0, weight_floor=0.02):
    """quotes = [(K, bid, ask)]; returns fitted params + diagnostics.

    Only quotes that are not pinned at the tick carry information, so a wide
    or one-sided market is downweighted rather than dropped: it still bounds
    the tail, it just should not steer the belly of the fit.
    """
    rows = []
    for K, b, a in quotes:
        b = 0.0 if b is None else b
        a = 1.0 if a is None else a
        if a <= 0 or a > 1 or b < 0 or b > 1 or b > a:
            continue
        mid = 0.5 * (b + a)
        sp = max(a - b, 1e-4)
        # Pinned 0/1c markets carry almost no information about the belly.
        w = 1.0 / (weight_floor + sp)
        if mid < 0.004 or mid > 0.996:
            w *= 0.05
        rows.append((K, mid, w))
    if len(rows) < 4:
        return None
    Ks = np.array([r[0] for r in rows])
    Ps = np.array([r[1] for r in rows])
    Ws = np.array([r[2] for r in rows])
    Ws = Ws / Ws.max()

    def resid(p):
        F, s0, beta, gamma = p
        if s0 <= 0 or F <= 0:
            return np.full(len(Ks), 1e3)
        return Ws * (model_cdf(Ks, F, T, s0, beta, gamma) - Ps)

    best = None
    for s_init in (0.2, 0.35, 0.55, 0.9):
        p0 = [F0, s_init, 0.0, 0.0]
        try:
            r = least_squares(resid, p0, max_nfev=6000,
                              bounds=([F0 * 0.9, 0.02, -4.0, -8.0],
                                      [F0 * 1.1, 6.0, 4.0, 8.0]))
        except Exception:  # noqa: BLE001
            continue
        rmse = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or rmse < best["rmse"]:
            F, s0, beta, gamma = r.x
            best = {"F": F, "sigma0": s0, "beta": beta, "gamma": gamma,
                    "rmse": rmse, "n": len(Ks), "T": T}
    if best is None:
        return None

    F, s0, beta, gamma = best["F"], best["sigma0"], best["beta"], best["gamma"]
    best["sigma_atm"] = s0
    best["cdf"] = lambda K: float(model_cdf([K], F, T, s0, beta, gamma)[0])
    # 25-delta-ish skew, a scale-free description of the fitted asymmetry
    kk = 0.5 * s0 * math.sqrt(max(T, 1e-9))
    best["skew_25d"] = float(s0 * (math.exp(-beta * kk + gamma * kk * kk)
                                   - math.exp(beta * kk + gamma * kk * kk)))
    return best


def quantile(fit, q):
    """Invert the fitted survival function."""
    F, T, s0 = fit["F"], fit["T"], fit["sigma0"]
    lo, hi = F * math.exp(-8 * s0 * math.sqrt(T)), F * math.exp(8 * s0 * math.sqrt(T))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if fit["cdf"](mid) > 1 - q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
