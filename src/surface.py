"""Deribit implied-vol surface -> arbitrage-free risk-neutral distribution.

Pipeline:
  1. Forward curve F(T) from the listed futures (log-linear in T between
     listed expiries, anchored on the index at T=0).
  2. Per-expiry SVI fit of total implied variance w(k) = sigma^2 T against
     log-moneyness k = ln(K/F), fitted to OTM quotes only and weighted by
     quote quality (tight two-sided markets dominate).
  3. Calendar interpolation of w at fixed k -- linear in total variance,
     which is the condition that rules out calendar arbitrage.
  4. Digitals and bucket probabilities by numerically differentiating the
     resulting call curve, so the skew term is handled exactly rather than
     via the flat-vol N(d2) approximation that retail venues implicitly use.

The output is a callable measure P(S_T > K) for *any* T in the listed range,
which is what lets Deribit (08:00Z expiries) price Kalshi (16:00Z / 21:00Z)
and Polymarket (16:00Z) events.
"""
import math
import re

import numpy as np
from scipy.optimize import least_squares

SQ2PI = math.sqrt(2.0 * math.pi)


def ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def npdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQ2PI


def bs_call(F, K, T, sigma, df=1.0):
    """Black-76 call on a forward."""
    F, K = np.asarray(F, float), np.asarray(K, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    T = max(T, 1e-8)
    v = sigma * math.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return df * (F * ncdf(d1) - K * ncdf(d2))


def implied_vol(price, F, K, T, df=1.0, is_call=True):
    """Robust bisection IV; returns None when the quote is outside no-arb bounds."""
    if price is None or price <= 0 or T <= 0:
        return None
    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
    upper = F if is_call else K
    if price <= intrinsic * df * (1 - 1e-9) or price >= upper * df:
        return None
    lo, hi = 1e-4, 8.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        c = float(bs_call(F, K, T, mid, df))
        p = c - df * (F - K)  # put-call parity
        v = c if is_call else p
        if v > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------- SVI

def svi_w(k, a, b, rho, m, sig):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig * sig))


def durrleman(k, a, b, rho, m, sig):
    """Durrleman's g(k); g >= 0 everywhere <=> no butterfly arbitrage."""
    k = np.asarray(k, float)
    w = svi_w(k, a, b, rho, m, sig)
    r = np.sqrt((k - m) ** 2 + sig * sig)
    wp = b * (rho + (k - m) / r)                       # dw/dk
    wpp = b * sig * sig / (r ** 3)                     # d2w/dk2
    return (1 - k * wp / (2 * w)) ** 2 - (wp ** 2 / 4) * (1 / w + 0.25) + wpp / 2


def fit_svi(k, iv, T, weights=None):
    """SVI fitted in *volatility* space, then screened for butterfly arbitrage.

    Fitting raw total variance is badly conditioned at these tenors -- w is
    O(1e-4), so a least-squares fit happily drives sigma to zero and puts a
    kink at the money, which then shows up as an absurd dsigma/dK and a broken
    digital. Working in vol points normalises the residuals, and a floor on
    sigma keeps the slice smooth. Candidate fits that violate Durrleman are
    rejected and retried from a stiffer start.
    """
    k = np.asarray(k, float)
    iv = np.asarray(iv, float)
    if weights is None:
        weights = np.ones_like(iv)
    weights = np.asarray(weights, float)
    if len(k) < 6:
        return None

    iv_atm = float(np.interp(0.0, k, iv))
    kk = np.linspace(min(k.min(), -0.6), max(k.max(), 0.6), 200)

    def model_iv(p):
        a, b, rho, m, sig = p
        return np.sqrt(np.maximum(svi_w(k, a, b, rho, m, sig), 1e-12) / T)

    def resid(p):
        return weights * (model_iv(p) - iv)

    best = None
    w_atm = iv_atm * iv_atm * T
    for sig0, rho0 in ((0.10, -0.1), (0.20, -0.3), (0.05, 0.0), (0.35, -0.5), (0.15, 0.3)):
        p0 = [w_atm * 0.3, max(w_atm, 1e-5) / max(sig0, 1e-3), rho0, 0.0, sig0]
        lo = [-0.5, 1e-9, -0.999, -1.5, 0.015]
        hi = [2.0, 10.0, 0.999, 1.5, 3.0]
        p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)
        except Exception:  # noqa: BLE001
            continue
        a, b, rho, m, sig = r.x
        if a + b * sig * math.sqrt(max(1 - rho * rho, 0)) < -1e-9:
            continue
        if float(durrleman(kk, a, b, rho, m, sig).min()) < -1e-6:
            continue
        rmse = float(np.sqrt(np.mean((model_iv(r.x) - iv) ** 2)))
        if best is None or rmse < best["rmse_vol"]:
            best = {"a": a, "b": b, "rho": rho, "m": m, "sigma": sig,
                    "rmse_vol": rmse,
                    "g_min": float(durrleman(kk, a, b, rho, m, sig).min())}
    return best


# ------------------------------------------------------------- surface build

INAME = re.compile(r"^(?P<ccy>\w+)-(?P<exp>\d+\w+\d+)-(?P<K>\d+(?:\.\d+)?)-(?P<cp>[CP])$")


def parse_instruments(blob):
    """Index Deribit option metadata by name."""
    meta = {}
    for i in blob["instruments"]["option"]:
        m = INAME.match(i["instrument_name"])
        if not m:
            continue
        meta[i["instrument_name"]] = {
            "expiry_ms": i["expiration_timestamp"],
            "strike": float(m.group("K")),
            "cp": m.group("cp"),
            "exp": m.group("exp"),
        }
    futs = {}
    for i in blob["instruments"]["future"]:
        futs[i["instrument_name"]] = i.get("expiration_timestamp")
    return meta, futs


def forward_curve(blob, now_ms):
    """(T_years, F) knots from the listed futures, plus the spot index at T=0."""
    idx = blob["index"]["index_price"]
    pts = [(0.0, idx)]
    summ = {b["instrument_name"]: b for b in blob["book_summary"]["future"]}
    _, futs = parse_instruments(blob)
    for name, exp in futs.items():
        if not exp or name.endswith("PERPETUAL"):
            continue
        s = summ.get(name)
        if not s:
            continue
        mark = s.get("mark_price") or s.get("last")
        if not mark:
            continue
        T = (exp - now_ms) / 1000.0 / (365.25 * 86400.0)
        if T > 0:
            pts.append((T, float(mark)))
    pts.sort()
    return pts


def fwd(curve, T):
    """Log-linear interpolation of the forward curve (flat extrapolation)."""
    Ts = [p[0] for p in curve]
    Fs = [math.log(p[1]) for p in curve]
    return math.exp(float(np.interp(T, Ts, Fs)))


class Surface:
    """Calendar-interpolated SVI surface for one currency."""

    def __init__(self, blob, now_ms):
        self.now_ms = now_ms
        self.index = blob["index"]["index_price"]
        self.curve = forward_curve(blob, now_ms)
        self.meta, _ = parse_instruments(blob)
        self.slices = {}
        self._build(blob)

    def _build(self, blob):
        by_exp = {}
        for s in blob["book_summary"]["option"]:
            name = s["instrument_name"]
            m = self.meta.get(name)
            if not m:
                continue
            by_exp.setdefault(m["expiry_ms"], []).append((s, m))

        for exp_ms, rows in sorted(by_exp.items()):
            T = (exp_ms - self.now_ms) / 1000.0 / (365.25 * 86400.0)
            if T <= 1e-6:
                continue
            F = fwd(self.curve, T)
            ks, ws, wt = [], [], []
            for s, m in rows:
                K = m["strike"]
                k = math.log(K / F)
                # OTM only: those are the liquid, information-bearing quotes.
                if (m["cp"] == "C" and k < -0.02) or (m["cp"] == "P" and k > 0.02):
                    continue
                iv = s.get("mark_iv")
                if not iv or iv <= 0:
                    continue
                iv /= 100.0
                bid, ask = s.get("bid_price"), s.get("ask_price")
                # Weight by quote tightness: a locked two-sided market is worth
                # far more than a mark on an untraded wing.
                if bid and ask and ask > bid:
                    q = 1.0 / (1.0 + 40.0 * (ask - bid) / max(ask, 1e-6))
                else:
                    q = 0.15
                oi = s.get("open_interest") or 0.0
                ks.append(k)
                ws.append(iv)
                wt.append(q * (1.0 + math.log1p(oi) / 6.0))
            if len(ks) < 6:
                continue
            order = np.argsort(ks)
            ks = np.array(ks)[order]
            ws = np.array(ws)[order]
            wt = np.array(wt)[order]
            fit = fit_svi(ks, ws, T, wt)
            if not fit:
                continue
            fit.update({"T": T, "F": F, "exp_ms": exp_ms, "n": len(ks),
                        "k_min": float(ks.min()), "k_max": float(ks.max()),
                        "iv_atm": float(np.interp(0.0, ks, ws))})
            self.slices[exp_ms] = fit

    # ---------------------------------------------------------- interpolation

    def total_var(self, k, T):
        """w(k,T), linear in T between slices at fixed log-moneyness.

        Linear-in-total-variance is exactly the condition that the interpolated
        surface stays calendar-arbitrage-free.
        """
        sl = sorted(self.slices.values(), key=lambda s: s["T"])
        if not sl:
            return None
        Ts = [s["T"] for s in sl]
        wk = [svi_w(np.array([k]), s["a"], s["b"], s["rho"], s["m"], s["sigma"])[0] for s in sl]
        if T <= Ts[0]:
            return max(wk[0] * T / Ts[0], 1e-10)          # flat vol below the front
        if T >= Ts[-1]:
            return max(wk[-1] * T / Ts[-1], 1e-10)
        return max(float(np.interp(T, Ts, wk)), 1e-10)

    def vol(self, K, T):
        F = fwd(self.curve, T)
        w = self.total_var(math.log(K / F), T)
        return math.sqrt(w / T)

    def call(self, K, T, df=1.0):
        F = fwd(self.curve, T)
        return float(bs_call(F, K, T, self.vol(K, T), df))

    def prob_above(self, K, T, h=None):
        """P(S_T > K) = -dC/dK, differentiated on the fitted smile.

        This is the skew-aware digital. The flat-vol N(d2) shortcut misses the
        vega * dsigma/dK term, which at these tenors is worth several cents on
        a wing digital.
        """
        h = h or max(K * 2e-4, 0.5)
        c1 = self.call(K - h, T)
        c2 = self.call(K + h, T)
        p = -(c2 - c1) / (2 * h)
        return float(min(max(p, 0.0), 1.0))

    def prob_between(self, lo, hi, T):
        return max(self.prob_above(lo, T) - self.prob_above(hi, T), 0.0)

    def density(self, K, T):
        """Breeden-Litzenberger second derivative."""
        h = max(K * 5e-4, 1.0)
        return float(max((self.call(K - h, T) - 2 * self.call(K, T)
                          + self.call(K + h, T)) / (h * h), 0.0))
