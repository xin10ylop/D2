"""Horizon-matched volatility forecast for coins with no listed options.

The barrier scan's first pass priced 150-day one-touch contracts with 30-day
realised volatility and duly "found" 35 cent edges on XRP and DOGE. Those were
the model, not the market: XRP's 30-day realised volatility is 36.8% while its
two-year volatility is 84.4%, and a barrier five months out is governed by the
latter.

Volatility mean-reverts, so the forecast for a horizon H blends the current
short-window estimate toward the long-run level:

    log sigma(H) = a(H) * log sigma_30d + (1 - a(H)) * log sigma_long
    a(H) = exp(-H / tau)

tau is calibrated so the forecast reproduces Deribit's *implied* term structure
for BTC and ETH -- the only two assets where a market-implied answer exists.
An alt-coin forecast is then the same function applied to that coin's own
realised inputs, rather than an assumption pulled from nowhere.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar

from common import load, save

YEAR_H = 365.25 * 24.0


def rv(bars, hours):
    c = np.array([b["c"] for b in bars], float)
    c = c[c > 0]
    r = np.diff(np.log(c))
    w = min(int(hours), len(r))
    return float(np.std(r[-w:]) * math.sqrt(YEAR_H))


def forecast(sig_short, sig_long, H_days, tau_days):
    a = math.exp(-H_days / tau_days)
    return math.exp(a * math.log(sig_short) + (1 - a) * math.log(sig_long))


def deribit_implied_curve(asset, der):
    """(days, implied ATM vol) from the listed surface."""
    from rnd import TermSurface
    from fitladder import fit_ladder
    ts = TermSurface(der["ccy"][asset], der["fetched_ms"], asset)
    out = []
    for e, s in sorted(ts.exps.items(), key=lambda x: x[1]["T"]):
        d = s["dens"]
        rows = [(float(K), max(d.prob_above(float(K)) - 1e-4, 0.0),
                 min(d.prob_above(float(K)) + 1e-4, 1.0)) for K in d.grid[::8]]
        f = fit_ladder(rows, s["T"], s["F"])
        if f:
            out.append((s["T"] * 365.25, f["sigma_atm"]))
    return out


def main():
    coins = load("coins")
    der = load("deribit")

    inputs = {}
    for c, b in coins["bars"].items():
        if not b:
            continue
        inputs[c] = {"s30": rv(b, 720), "s90": rv(b, 2160),
                     "slong": rv(b, len(b))}

    # calibrate tau on BTC and ETH against Deribit's implied curve
    curves = {a: deribit_implied_curve(a, der) for a in ("BTC", "ETH")}

    def err(tau):
        e = []
        for a, cv in curves.items():
            i = inputs[a]
            for days, iv in cv:
                if days < 3 or days > 400:
                    continue
                f = forecast(i["s30"], i["slong"], days, tau)
                e.append((math.log(f) - math.log(iv)) ** 2)
        return float(np.mean(e)) if e else 1e9

    r = minimize_scalar(err, bounds=(10.0, 800.0), method="bounded")
    tau = float(r.x)
    print(f"calibrated mean-reversion tau = {tau:.1f} days "
          f"(rms log error {math.sqrt(r.fun)*100:.1f}%)\n")

    print("validation against Deribit implied ATM vol:")
    print(f"  {'asset':5s} {'days':>6s} {'implied':>8s} {'forecast':>9s} {'err':>7s}")
    for a, cv in curves.items():
        i = inputs[a]
        for days, iv in cv:
            if days < 3 or days > 400:
                continue
            f = forecast(i["s30"], i["slong"], days, tau)
            print(f"  {a:5s} {days:6.1f} {iv*100:7.1f}% {f*100:8.1f}% "
                  f"{(f/iv-1)*100:+6.1f}%")

    print(f"\nforecast volatility by coin and horizon:")
    print(f"  {'coin':6s} {'s30':>7s} {'slong':>7s} {'28d':>7s} {'150d':>7s} {'300d':>7s}")
    out = {"tau_days": tau, "inputs": inputs}
    for c, i in sorted(inputs.items()):
        f28 = forecast(i["s30"], i["slong"], 28, tau)
        f150 = forecast(i["s30"], i["slong"], 150, tau)
        f300 = forecast(i["s30"], i["slong"], 300, tau)
        out[c] = {"f28": f28, "f150": f150, "f300": f300}
        print(f"  {c:6s} {i['s30']*100:6.1f}% {i['slong']*100:6.1f}% "
              f"{f28*100:6.1f}% {f150*100:6.1f}% {f300*100:6.1f}%")
    save("volterm", out)
    return out


if __name__ == "__main__":
    main()
