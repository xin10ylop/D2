"""The weekend variance trade.

Polymarket lists BTC/ETH daily contracts settling 16:00Z every day, including
Saturday (Aug 8) and Sunday (Aug 9). Deribit lists no weekend expiry at all --
it jumps from Friday 7 Aug to Friday 14 Aug. So the weekend contracts are the
only instruments in the entire cross-venue dataset with no listed options
market to discipline them, and Kalshi does not list them either.

That is exactly where a structural mispricing can survive.

This module measures, with no model in between, the realised variance of the
*exact* windows these contracts pay on -- 16:00Z on day D to 16:00Z on day D+1
-- over two years, and compares it with the incremental variance Polymarket is
currently charging between consecutive daily ladders.
"""
import datetime as dt
import math

import numpy as np

from common import load, save

YEAR_H = 365.25 * 24.0
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def close_series(bars):
    t = np.array([b["open_ms"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    ok = c > 0
    return t[ok], c[ok]


def window_returns(bars, anchor_hour=16):
    """Log return over each [anchor_hour on day D -> anchor_hour on day D+1].

    Indexed by the weekday the window *ends* on, which is how the contracts
    are named: 'Bitcoin above X on August 8' covers Fri 16:00Z -> Sat 16:00Z.
    """
    t, c = close_series(bars)
    idx = {}
    for ts, px in zip(t, c):
        u = dt.datetime.utcfromtimestamp(ts / 1000)
        if u.hour == anchor_hour:
            idx[u.date()] = px
    days = sorted(idx)
    out = {i: [] for i in range(7)}
    for i in range(1, len(days)):
        d0, d1 = days[i - 1], days[i]
        if (d1 - d0).days != 1:
            continue
        r = math.log(idx[d1] / idx[d0])
        out[d1.weekday()].append(r)
    return out


def stats(rs, boot=2000, seed=0):
    a = np.array(rs, float)
    if len(a) < 10:
        return None
    v = float(np.mean(a ** 2))
    rng = np.random.default_rng(seed)
    sims = np.array([np.mean(rng.choice(a, len(a)) ** 2) for _ in range(boot)])
    return {"n": len(a), "var": v, "sd_ann": math.sqrt(v * 365.25),
            "var_lo": float(np.percentile(sims, 2.5)),
            "var_hi": float(np.percentile(sims, 97.5)),
            "ann_lo": math.sqrt(float(np.percentile(sims, 2.5)) * 365.25),
            "ann_hi": math.sqrt(float(np.percentile(sims, 97.5)) * 365.25)}


def poly_increments():
    """Incremental variance Polymarket charges between consecutive dailies."""
    rv = load("rv_events")
    out = {}
    for r in rv:
        if r["venue"] != "polymarket":
            continue
        # sigma_atm was fitted against business time, so sigma^2 * tau/YEAR is
        # the contract's TOTAL variance -- a calendar-clock-free quantity that
        # differences cleanly against realised window variance.
        w = r["sigma_atm"] ** 2 * (r["tau_h"] / YEAR_H)
        wc = w
        settle = dt.datetime.utcfromtimestamp(r["t1"] / 1000)
        out.setdefault(r["asset"], []).append(
            {"name": r["name"], "settle": settle.isoformat() + "Z",
             "cal_h": r["cal_h"], "tau_h": r["tau_h"],
             "sigma": r["sigma_atm"], "w": w, "wc": wc,
             "dow": settle.weekday()})
    for a in out:
        out[a].sort(key=lambda x: x["cal_h"])
    return out


def main():
    hist = load("hist")
    res = {}
    print("Realised variance of the EXACT contract window "
          "(16:00Z day D-1 -> 16:00Z day D), two years of hourly closes\n")
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        wr = window_returns(hist["bars"][f"{sym}_1h"], 16)
        st = {i: stats(wr[i]) for i in range(7)}
        base = float(np.mean([st[i]["var"] for i in (1, 2, 3)  # Tue/Wed/Thu
                              if st[i]]))
        res[asset] = {"by_dow": {DOW[i]: st[i] for i in range(7) if st[i]},
                      "midweek_var": base}
        print(f"=== {asset}")
        print(f"{'settles':8s} {'n':>4s} {'ann.vol':>9s}  {'95% CI':>18s}"
              f"  {'x midweek':>10s}")
        for i in range(7):
            s = st[i]
            if not s:
                continue
            print(f"{DOW[i]:8s} {s['n']:4d} {s['sd_ann']*100:8.2f}%  "
                  f"[{s['ann_lo']*100:6.2f}%,{s['ann_hi']*100:6.2f}%]"
                  f"  {s['var']/base:10.2f}")
        print()

    inc = poly_increments()
    print("\nPolymarket's implied incremental variance per day vs what that "
          "day actually realises\n")
    rows = []
    for asset in ("BTC", "ETH"):
        seq = inc.get(asset, [])
        base = res[asset]["midweek_var"]
        print(f"=== {asset}  (midweek realised daily var = {base:.3e}, "
              f"{math.sqrt(base*365.25)*100:.1f}% ann)")
        print(f"{'contract':34s} {'settles':8s} {'sigATM':>7s} {'w_total':>10s} "
              f"{'w_incr':>10s} {'xmidweek':>9s} {'realised x':>11s} {'rich/cheap':>11s}")
        prev = None
        for r in seq:
            lbl = DOW[r["dow"]]
            if prev is None:
                incr = None
            else:
                incr = r["w"] - prev["w"]
            st = res[asset]["by_dow"].get(lbl)
            realx = (st["var"] / base) if st else None
            impx = (incr / base) if incr is not None else None
            ratio = (impx / realx) if (impx is not None and realx) else None
            print(f"{r['name'][:34]:34s} {lbl:8s} {r['sigma']*100:6.2f}% "
                  f"{r['w']:10.3e} "
                  f"{(incr if incr is not None else float('nan')):10.3e} "
                  f"{(impx if impx is not None else float('nan')):9.2f} "
                  f"{(realx or float('nan')):11.2f} "
                  f"{(ratio if ratio else float('nan')):11.2f}")
            if incr is not None and realx:
                rows.append({"asset": asset, "name": r["name"], "dow": lbl,
                             "w_incr": incr, "implied_x": impx,
                             "realised_x": realx, "ratio": ratio,
                             "settle": r["settle"]})
            prev = r
        print()

    print("ratio > 1 = Polymarket charges more variance for that day than it "
          "historically realises (sell that day's wings);")
    print("ratio < 1 = the day is being given away (buy that day's wings).")
    save("weekend", {"realised": res, "rows": rows})
    return res, rows


if __name__ == "__main__":
    main()
