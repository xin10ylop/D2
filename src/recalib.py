"""Probability recalibration -- the layer that makes the model safe to trade on.

The calibration test showed both clock models over-predicting moderate moves
and under-predicting staying near spot: predicted 31% events happen 26% of the
time, predicted 69% events happen 74% of the time. That is a distribution
materially more peaked than lognormal, and it is the reason the bot's biggest
apparent edges were on short-dated contracts -- it was pricing a wider
distribution than reality and then "discovering" that the market's tighter
quotes were cheap.

The fix is a one-parameter sharpening in probit space,

    p_cal = Phi( b * Phi^-1(p_model) )

with b > 1 pulling probabilities toward 0 and 1. b is fitted per horizon by
maximum likelihood on two years of outcomes, which is enough data for a single
parameter to be estimated precisely.

Fitted on the model's own predictions, this removes the systematic error
without touching the part of the model that is working.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar

from common import load, save
from howclock import WeekClock
from calibrate import run, ncdf

HORIZONS = (1, 2, 3, 6, 12, 24)


def probit(p, eps=1e-6):
    from scipy.special import ndtri
    return ndtri(np.clip(p, eps, 1 - eps))


def phi(z):
    from scipy.special import ndtr
    return ndtr(z)


def fit_b(p, y):
    """Maximum-likelihood sharpening coefficient."""
    z = probit(p)

    def nll(b):
        q = np.clip(phi(b * z), 1e-9, 1 - 1e-9)
        return -float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    r = minimize_scalar(nll, bounds=(0.4, 3.0), method="bounded")
    return float(r.x), float(r.fun)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main():
    hist = load("hist")
    how = load("howclock")
    out = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        clock = WeekClock(how[asset]["clock"])
        bars = hist["bars"][f"{sym}_1h"]
        out[asset] = {}
        print(f"\n=== {asset}: sharpening coefficient b by horizon")
        print(f"   b>1 means the true distribution is narrower than the model's")
        print(f"   {'horizon':>8s} {'b':>6s} {'sd ratio':>9s} {'Brier raw':>10s} "
              f"{'Brier cal':>10s} {'MAE raw':>8s} {'MAE cal':>8s}")
        for H in HORIZONS:
            rows = run(bars, clock, horizon_h=H)
            sel = [(p, y) for n, p, y, _ in rows if n == "week"]
            p = np.array([x[0] for x in sel])
            y = np.array([x[1] for x in sel])
            b, _ = fit_b(p, y)
            pc = phi(b * probit(p))
            # bucketed absolute calibration error, before and after
            def mae(pp):
                e = []
                for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .6),
                               (.6, .7), (.7, .8), (.8, .9), (.9, 1.0)]:
                    m = (pp >= lo) & (pp < hi)
                    if m.sum() > 200:
                        e.append(abs(pp[m].mean() - y[m].mean()))
                return float(np.mean(e)) if e else float("nan")
            out[asset][H] = {"b": b, "brier_raw": brier(p, y),
                             "brier_cal": brier(pc, y),
                             "mae_raw": mae(p), "mae_cal": mae(pc), "n": len(p)}
            print(f"   {H:6d}h {b:6.3f} {1/b:9.3f} {brier(p,y):10.5f} "
                  f"{brier(pc,y):10.5f} {mae(p)*100:7.2f}c {mae(pc)*100:7.2f}c")
    save("recalib", out)
    print("\n'sd ratio' is how much narrower the real distribution is than the")
    print("model's: at 1h the model must be shrunk by that factor to be honest.")
    return out


if __name__ == "__main__":
    main()
