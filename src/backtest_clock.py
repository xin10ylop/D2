"""Does the business clock actually predict better than a flat clock?

This is the load-bearing question for any market-making bot on Kalshi's hourly
ladders. A quoter that scales volatility by sqrt(calendar hours) will
systematically overprice short-dated contracts that span the quiet Asian night
and underprice ones that span the US open. If that is true and measurable, a
bot pricing on the business clock has a real edge against that flow -- not just
the spread.

Test, on two years of hourly closes:
  for every start hour t and horizon h, standardise the realised return by each
  model's predicted sigma. A correctly specified model produces standardised
  residuals with unit variance in EVERY hour-of-day bucket. Systematic
  departure from 1.0 is the model being wrong in a predictable direction, which
  is exactly what is tradable.

The dispersion ratio is then converted into what actually matters: the pricing
error, in cents, on an at-the-money digital.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from seasonality import Clock

YEAR_H = 365.25 * 24.0


def series(bars):
    t = np.array([b["open_ms"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    ok = c > 0
    return t[ok], c[ok]


def build(bars, clock, horizons=(1, 2, 3, 5, 8)):
    """Standardised residuals under both clocks, indexed by settlement hour."""
    t, c = series(bars)
    n = len(c)
    hours = np.array([dt.datetime.utcfromtimestamp(x / 1000).hour for x in t])
    out = []
    for h in horizons:
        r = np.log(c[h:] / c[:-h])
        t0 = t[:-h]
        end_h = hours[h:]
        # each model's *relative* variance for this window
        cal = np.full(len(r), float(h))
        biz = np.array([clock.tau(a, a + h * 3600_000) for a in t0])
        out.append({"h": h, "r": r, "t0": t0, "end_h": end_h,
                    "cal": cal, "biz": biz})
    return out


def dispersion(rows, key, nbucket=24):
    """Std of r / (k * sqrt(model_time)), with k fitted so the pooled std is 1."""
    r = np.concatenate([x["r"] for x in rows])
    m = np.concatenate([x[key] for x in rows])
    eh = np.concatenate([x["end_h"] for x in rows])
    z = r / np.sqrt(np.maximum(m, 1e-9))
    k = float(np.std(z))                       # global vol scale, fitted once
    z = z / k
    per = {}
    for hh in range(nbucket):
        sel = eh == hh
        if sel.sum() > 30:
            per[hh] = float(np.std(z[sel]))
    return {"k": k, "pooled": float(np.std(z)), "per_hour": per,
            "spread": float(max(per.values()) - min(per.values())),
            "rmse_from_1": float(np.sqrt(np.mean(
                [(v - 1.0) ** 2 for v in per.values()])))}


def digital_error(disp_ratio, sigma=0.30, hours=3.0):
    """Cents of error on an ATM digital when vol is misestimated by disp_ratio.

    P(S > K) for a strike one tick above the forward moves with sigma; the
    biggest sensitivity is on a strike ~0.5 sigma out, where dP/dsigma is
    near its maximum.
    """
    T = hours / YEAR_H
    s = sigma * math.sqrt(T)
    from surface import ncdf
    K_off = 0.5 * s                       # half a sigma OTM
    p_true = float(ncdf(-(K_off) / s + 0))
    p_wrong = float(ncdf(-(K_off) / (s * disp_ratio)))
    return abs(p_true - p_wrong)


def main():
    hist = load("hist")
    seas = load("seasonality")
    clock = Clock(seas["clock_hour"], seas["clock_dow"])
    res = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        rows = build(hist["bars"][f"{sym}_1h"], clock)
        flat = dispersion(rows, "cal")
        biz = dispersion(rows, "biz")
        res[asset] = {"flat": flat, "biz": biz}
        print(f"\n=== {asset}: standardised-residual dispersion by settlement hour")
        print("   perfect model = 1.00 in every bucket\n")
        print(f"   {'hour':6s} {'flat clock':>11s} {'business clock':>15s}")
        for hh in sorted(flat["per_hour"]):
            f, b = flat["per_hour"][hh], biz["per_hour"][hh]
            barf = "#" * int(round(f * 22))
            print(f"   {hh:02d}Z    {f:11.3f} {b:15.3f}   {barf}")
        print(f"\n   flat clock : max-min spread {flat['spread']:.3f}, "
              f"RMSE from 1.0 = {flat['rmse_from_1']:.4f}")
        print(f"   business   : max-min spread {biz['spread']:.3f}, "
              f"RMSE from 1.0 = {biz['rmse_from_1']:.4f}")
        imp = (1 - biz["rmse_from_1"] / flat["rmse_from_1"]) * 100
        print(f"   -> business clock cuts the mis-specification by {imp:.1f}%")

        worst_h = max(flat["per_hour"], key=lambda k: abs(flat["per_hour"][k] - 1))
        wr = flat["per_hour"][worst_h]
        err = digital_error(1.0 / wr) * 100
        print(f"   worst hour for a flat quoter: {worst_h:02d}Z at "
              f"{wr:.3f}x -> ~{err:.1f} cents of error on a half-sigma digital")

    save("backtest_clock", res)
    return res


if __name__ == "__main__":
    main()
