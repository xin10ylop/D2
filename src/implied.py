"""Extract each venue's implied distribution and put them all on one clock.

A prediction-market digital ladder *is* the risk-neutral CDF -- P(S>K) is
quoted directly, strike by strike -- so no model is needed to read it. The
bucket ladders give the same object as a histogram. That makes Kalshi and
Polymarket directly comparable to a Deribit density, once three adjustments
are made:

  * index basis: Polymarket settles on Binance BTC/USDT, Kalshi and Deribit on
    USD. Polymarket strikes are divided by the USDT/USD peg to move them onto
    the USD axis before anything is compared.
  * forward: each venue's distribution is centred on its own forward, so
    comparisons are done in log-moneyness against that forward.
  * clock: variance is measured per unit of *business time*, not calendar
    time, so an 08:00Z expiry and a 21:00Z settlement can be compared.

The headline statistic per ladder is sigma_eff -- the annualised volatility
implied by the quoted distribution's central quantile spread, which is robust
to the tails being pinned at one tick.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from seasonality import Clock
from surface import ncdf

YEAR_H = 365.25 * 24.0


def iso_ms(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


# ------------------------------------------------------- ladder -> CDF

def kalshi_cdf(markets):
    """(strike, P(S>strike)) from the digital ladder mid prices."""
    pts = []
    for m in markets:
        if m["strike_type"] not in ("greater", "greater_or_equal"):
            continue
        b, a = m["yes_bid"], m["yes_ask"]
        if b is None or a is None or a >= 1.0 and b <= 0.0:
            continue
        mid = 0.5 * (b + a)
        pts.append((m["floor_strike"], mid, b, a))
    pts.sort()
    return pts


def kalshi_pdf(markets):
    """(lo, hi, P) from the bucket ladder mid prices."""
    out = []
    for m in markets:
        b, a = m["yes_bid"], m["yes_ask"]
        if b is None or a is None:
            continue
        mid = 0.5 * (b + a)
        if m["strike_type"] == "between":
            out.append((m["floor_strike"], m["cap_strike"], mid, b, a))
        elif m["strike_type"] in ("less", "less_or_equal"):
            out.append((0.0, m["cap_strike"], mid, b, a))
        elif m["strike_type"] in ("greater", "greater_or_equal"):
            out.append((m["floor_strike"], float("inf"), mid, b, a))
    out.sort()
    return out


def poly_cdf(markets, scale=1.0):
    pts = []
    for m in markets:
        t = (m["group_title"] or "").replace(",", "").replace("$", "")
        try:
            K = float(t)
        except ValueError:
            continue
        b, a = m["best_bid"], m["best_ask"]
        if b is None and a is None:
            continue
        b = 0.0 if b is None else b
        a = 1.0 if a is None else a
        pts.append((K * scale, 0.5 * (b + a), b, a))
    pts.sort()
    return pts


def poly_pdf(markets, scale=1.0):
    import re
    NUM = re.compile(r"[\d,]+")
    out = []
    for m in markets:
        t = (m["group_title"] or "").replace("$", "").strip()
        nums = [float(x.replace(",", "")) for x in NUM.findall(t)]
        if not nums:
            continue
        if t.startswith("<"):
            lo, hi = 0.0, nums[0]
        elif t.startswith(">"):
            lo, hi = nums[0], float("inf")
        elif len(nums) >= 2:
            lo, hi = nums[0], nums[1]
        else:
            continue
        b, a = m["best_bid"], m["best_ask"]
        b = 0.0 if b is None else b
        a = 1.0 if a is None else a
        out.append((lo * scale, hi * scale, 0.5 * (b + a), b, a))
    out.sort()
    return out


# ------------------------------------------------ distribution statistics

def quantiles_from_cdf(pts, qs=(0.16, 0.25, 0.5, 0.75, 0.84)):
    """Invert a survival-function ladder. pts = [(K, P(S>K)), ...] ascending K."""
    K = np.array([p[0] for p in pts], float)
    S = np.array([p[1] for p in pts], float)
    S = np.minimum.accumulate(S)            # enforce monotone survival
    F = 1.0 - S                             # CDF
    out = {}
    for q in qs:
        if F[0] > q or F[-1] < q:
            out[q] = None
            continue
        out[q] = float(np.interp(q, F, K))
    return out


def quantiles_from_pdf(rows, qs=(0.16, 0.25, 0.5, 0.75, 0.84)):
    """rows = [(lo, hi, p), ...] over a partition, mass uniform within a bucket.

    Anchoring the CDF only at bucket tops shifts every quantile by half a
    bucket width and inflates the 16/84 spread -- enough, on Kalshi's 250-wide
    grid, to make the bucket ladder look 15 vol points richer than the digital
    ladder settling on the identical print. Building the CDF as a piecewise
    linear function through *both* edges of each bucket removes the bias.
    """
    tot = sum(r[2] for r in rows)
    if tot <= 0:
        return {q: None for q in qs}
    xs, ys, c = [], [], 0.0
    for lo, hi, p, *_ in rows:
        w = p / tot
        a = lo
        b = hi if math.isfinite(hi) else lo * 1.12
        if not xs:
            xs.append(a)
            ys.append(0.0)
        xs.append(b)
        ys.append(c + w)
        c += w
    xs, ys = np.array(xs), np.array(ys)
    ok = np.diff(xs) > 0
    xs = np.r_[xs[0], xs[1:][ok]]
    ys = np.r_[ys[0], ys[1:][ok]]
    out = {}
    for q in qs:
        out[q] = float(np.interp(q, ys, xs)) if ys[0] <= q <= ys[-1] else None
    return out


def sigma_eff(qd, tau_h, fwd=None):
    """Annualised vol from the 16/84 quantile spread of a lognormal fit.

    The central spread is used deliberately: the wings of every one of these
    ladders are pinned at a one-cent tick and carry no information.
    """
    lo, hi = qd.get(0.16), qd.get(0.84)
    if not lo or not hi or hi <= lo or tau_h <= 0:
        return None
    return float(math.log(hi / lo) / 2.0 / math.sqrt(tau_h / YEAR_H))


def sigma_from_atm_digital(pts, F, tau_h):
    """Vol backed out of the single digital nearest the forward.

    P(S>F) = N(-sigma sqrt(T)/2) for a driftless lognormal, so one at-the-money
    quote pins the vol -- a useful cross-check on the quantile estimate.
    """
    if not pts or tau_h <= 0:
        return None
    K, P = min(pts, key=lambda p: abs(p[0] - F))[:2]
    if not (0.02 < P < 0.98):
        return None
    T = tau_h / YEAR_H
    from scipy.optimize import brentq

    def f(s):
        v = s * math.sqrt(T)
        return float(ncdf((math.log(F / K) - 0.5 * v * v) / v)) - P
    try:
        return float(brentq(f, 1e-3, 6.0))
    except ValueError:
        return None


# ------------------------------------------------------------------ main

def main():
    kal, poly = load("kalshi"), load("polymarket")
    ref, seas = load("ref"), load("seasonality")
    der = load("deribit")
    from rnd import TermSurface

    clock = Clock(seas["clock_hour"], seas["clock_dow"])
    flat = Clock(np.ones(24), np.ones(7))
    now = kal["fetched_ms"]

    # USDT peg: Polymarket strikes live on a USDT axis, everyone else on USD.
    spot = ref["spot"]
    usd = {a: np.mean([spot[f"{v}_{a}"] for v in ("coinbase", "bitstamp", "gemini", "kraken")
                       if spot.get(f"{v}_{a}")]) for a in ("BTC", "ETH")}
    peg = {a: usd[a] / spot[f"binance_{a}"] for a in ("BTC", "ETH")}
    print("index basis (multiply Polymarket strikes by this to reach the USD axis):")
    for a in ("BTC", "ETH"):
        print(f"  {a}: USD composite {usd[a]:,.2f} | Binance USDT {spot[f'binance_{a}']:,.2f} "
              f"| peg {peg[a]:.6f} ({(peg[a]-1)*1e4:+.2f} bp)")

    surf = {a: TermSurface(der["ccy"][a], der["fetched_ms"], a) for a in ("BTC", "ETH")}

    rows = []

    # ---- Kalshi
    for series, evs in kal["series"].items():
        asset = "ETH" if "ETH" in series else "BTC"
        kind = "range" if series in ("KXBTC", "KXETH") else "above"
        for ev, blob in evs.items():
            ms = blob["markets"]
            if not ms:
                continue
            quoted = [m for m in ms if m["yes_ask"] is not None and m["yes_ask"] < 1.0]
            if len(quoted) < 8:
                continue
            t1 = iso_ms(ms[0]["close_time"])
            tau = clock.tau(now, t1)
            cal = (t1 - now) / 3600_000.0
            if cal <= 0:
                continue
            F = surf[asset].index
            if kind == "above":
                pts = kalshi_cdf(quoted)
                qd = quantiles_from_cdf([(p[0], p[1]) for p in pts])
                s_atm = sigma_from_atm_digital([(p[0], p[1]) for p in pts], F, tau)
            else:
                pdf = kalshi_pdf(quoted)
                qd = quantiles_from_pdf([(r[0], r[1], r[2]) for r in pdf])
                s_atm = None
            rows.append({"venue": "kalshi", "asset": asset, "kind": kind, "event": ev,
                         "settle": ms[0]["close_time"], "cal_h": cal, "tau_h": tau,
                         "n_quoted": len(quoted),
                         "q16": qd.get(0.16), "q50": qd.get(0.5), "q84": qd.get(0.84),
                         "sigma_cal": sigma_eff(qd, cal),
                         "sigma_biz": sigma_eff(qd, tau),
                         "sigma_atm_biz": s_atm})

    # ---- Polymarket
    for slug, e in poly["events"].items():
        asset = e["asset"]
        sc = peg[asset]
        t1 = iso_ms(e["end_date"])
        tau = clock.tau(now, t1)
        cal = (t1 - now) / 3600_000.0
        if cal <= 0:
            continue
        F = surf[asset].index
        if e["kind"] == "above":
            pts = poly_cdf(e["markets"], sc)
            qd = quantiles_from_cdf([(p[0], p[1]) for p in pts])
            s_atm = sigma_from_atm_digital([(p[0], p[1]) for p in pts], F, tau)
        else:
            pdf = poly_pdf(e["markets"], sc)
            qd = quantiles_from_pdf([(r[0], r[1], r[2]) for r in pdf])
            s_atm = None
        rows.append({"venue": "polymarket", "asset": asset, "kind": e["kind"],
                     "event": slug, "settle": e["end_date"], "cal_h": cal, "tau_h": tau,
                     "n_quoted": len(e["markets"]),
                     "q16": qd.get(0.16), "q50": qd.get(0.5), "q84": qd.get(0.84),
                     "sigma_cal": sigma_eff(qd, cal),
                     "sigma_biz": sigma_eff(qd, tau),
                     "sigma_atm_biz": s_atm})

    # ---- Deribit (same statistic, from the fitted density)
    for asset in ("BTC", "ETH"):
        for e, s in sorted(surf[asset].exps.items(), key=lambda x: x[1]["T"]):
            d = s["dens"]
            cal = (e - now) / 3600_000.0
            if cal <= 0 or cal > 400:
                continue
            tau = clock.tau(now, e)
            qd = {q: d.quantile(q) for q in (0.16, 0.5, 0.84)}
            rows.append({"venue": "deribit", "asset": asset, "kind": "option",
                         "event": dt.datetime.utcfromtimestamp(e / 1000).strftime("%d%b%y-%HZ"),
                         "settle": dt.datetime.utcfromtimestamp(e / 1000).isoformat() + "Z",
                         "cal_h": cal, "tau_h": tau, "n_quoted": s["n_quotes"],
                         "q16": qd[0.16], "q50": qd[0.5], "q84": qd[0.84],
                         "sigma_cal": sigma_eff(qd, cal),
                         "sigma_biz": sigma_eff(qd, tau),
                         "sigma_atm_biz": None})

    rows.sort(key=lambda r: (r["asset"], r["cal_h"], r["venue"]))
    save("implied", {"rows": rows, "peg": peg, "usd": usd})

    for asset in ("BTC", "ETH"):
        print(f"\n=== {asset}  implied distributions, all venues on one clock")
        print(f"{'venue':11s} {'kind':6s} {'event':32s} {'cal_h':>7s} {'biz_h':>7s} "
              f"{'q16':>10s} {'q50':>10s} {'q84':>10s} {'sig_cal':>8s} {'sig_biz':>8s}")
        for r in rows:
            if r["asset"] != asset or r["cal_h"] > 200:
                continue
            f = lambda x, w=10, d=0: (f"{x:,.{d}f}".rjust(w) if x else "-".rjust(w))  # noqa: E731
            sc = f"{r['sigma_cal']*100:7.2f}%" if r["sigma_cal"] else "      - "
            sb = f"{r['sigma_biz']*100:7.2f}%" if r["sigma_biz"] else "      - "
            print(f"{r['venue']:11s} {r['kind']:6s} {r['event'][:32]:32s} "
                  f"{r['cal_h']:7.2f} {r['tau_h']:7.2f} "
                  f"{f(r['q16'])} {f(r['q50'])} {f(r['q84'])} {sc} {sb}")
    return rows


if __name__ == "__main__":
    main()
