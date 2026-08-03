"""Is the model actually calibrated? The test that decides whether the bot trades.

The bot's premise is that a digital priced on the business clock is closer to
the truth than one priced on calendar time. That is a testable claim about
predicted probabilities, and it can be tested without any quote history: run
the model over two years, and for every predicted probability bucket compare
the prediction to the frequency the event actually occurred.

Two models, identical except for the clock:
  FLAT  sigma from a trailing realised estimator, scaled by sqrt(calendar hours)
  WEEK  the same estimator, scaled by sqrt(business hours) on the hour-of-week
        variance clock

A well-calibrated model sits on the diagonal. The gap between the two curves,
converted to cents, is the per-contract edge available against a counterparty
quoting on the wrong clock -- and the Brier score says which is better overall.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from howclock import WeekClock, how_index

YEAR_H = 365.25 * 24.0
BUCKETS = [(0.0, .05), (.05, .15), (.15, .25), (.25, .35), (.35, .45),
           (.45, .55), (.55, .65), (.65, .75), (.75, .85), (.85, .95), (.95, 1.0)]


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def run(bars, clock, horizon_h=3, rv_window=168, offsets=(-1.0, -0.5, 0.0, 0.5, 1.0)):
    """Predict P(S_T > K) at several sigma-offsets and record the outcome."""
    t = np.array([b["open_ms"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    ok = c > 0
    t, c = t[ok], c[ok]
    r = np.diff(np.log(c))
    h = int(horizon_h)
    rows = []
    for i in range(rv_window, len(c) - h):
        # trailing realised variance per business hour, so the estimator itself
        # is clock-aware and the only difference between models is the scaling
        w = r[i - rv_window:i]
        tw = clock.tau(t[i - rv_window], t[i])
        if tw <= 0:
            continue
        rate = float(np.sum(w ** 2)) / tw            # variance per business hour
        tau = clock.tau(t[i], t[i + h])
        cal = h * 1.0
        var_week = rate * tau
        var_flat = rate * cal                         # same rate, calendar scaling
        S0, S1 = c[i], c[i + h]
        for off in offsets:
            for name, var in (("week", var_week), ("flat", var_flat)):
                sd = math.sqrt(max(var, 1e-12))
                K = S0 * math.exp(off * sd)
                p = ncdf((math.log(S0 / K) - 0.5 * var) / sd)
                rows.append((name, p, 1.0 if S1 > K else 0.0,
                             how_index(t[i + h])))
    return rows


def summarise(rows, name):
    sel = [(p, y) for n, p, y, _ in rows if n == name]
    p = np.array([x[0] for x in sel])
    y = np.array([x[1] for x in sel])
    brier = float(np.mean((p - y) ** 2))
    out = []
    for lo, hi in BUCKETS:
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() < 200:
            continue
        out.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                    "pred": float(p[m].mean()), "actual": float(y[m].mean())})
    mae = float(np.mean([abs(b["pred"] - b["actual"]) for b in out])) if out else None
    return {"brier": brier, "buckets": out, "mae": mae, "n": len(p)}


def main():
    hist = load("hist")
    how = load("howclock")
    res = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        clock = WeekClock(how[asset]["clock"])
        bars = hist["bars"][f"{sym}_1h"]
        res[asset] = {}
        print(f"\n{'='*70}\n=== {asset}")
        for H in (1, 3, 6, 12, 24):
            rows = run(bars, clock, horizon_h=H)
            wk = summarise(rows, "week")
            fl = summarise(rows, "flat")
            res[asset][H] = {"week": wk, "flat": fl}
            imp = (1 - wk["brier"] / fl["brier"]) * 100
            print(f"\n  horizon {H}h   n={wk['n']:,}   "
                  f"Brier: flat {fl['brier']:.5f} -> week {wk['brier']:.5f} "
                  f"({imp:+.2f}%)   mean |pred-actual|: "
                  f"flat {fl['mae']*100:.2f}c -> week {wk['mae']*100:.2f}c")
            print(f"    {'predicted':>10s} {'n':>7s} {'flat actual':>12s} "
                  f"{'week actual':>12s}   flat err   week err")
            fb = {(b['lo']): b for b in fl["buckets"]}
            for b in wk["buckets"]:
                f = fb.get(b["lo"])
                if not f:
                    continue
                print(f"    {b['pred']*100:9.1f}% {b['n']:7,d} "
                      f"{f['actual']*100:11.1f}% {b['actual']*100:11.1f}%   "
                      f"{(f['actual']-f['pred'])*100:+8.2f}c "
                      f"{(b['actual']-b['pred'])*100:+9.2f}c")
    save("calibrate", res)
    return res


if __name__ == "__main__":
    main()
