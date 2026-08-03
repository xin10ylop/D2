"""Estimate and stress-test the variance clock, then ask whether Deribit uses it.

Three questions, in order:

  Q1  Is the hour-of-day variance profile real, or 40 noisy observations per
      bucket? Tested by bootstrap CIs, a split-half over two years, and
      independent agreement between BTC and ETH.

  Q2  Is there a day-of-week effect? It matters directly: Polymarket lists
      Aug 8 and Aug 9 contracts that settle on a Saturday and a Sunday, and
      weekend crypto variance is materially lower.

  Q3  Does Deribit price any of this? Its dailies all expire 08:00Z, so
      consecutive expiries are exactly one full cycle apart and cannot
      discriminate. The discriminating observation is the *front stub* -- the
      partial day from now to the first expiry. If Deribit's implied variance
      rate over that stub matches the flat-calendar rate rather than the
      business-time rate, then short-dated options are being priced on the
      wrong clock, and that is a structural, repeatable edge.
"""
import datetime as dt
import math

import numpy as np

from common import load, save


def rets(bars):
    c = np.array([b["c"] for b in bars], float)
    t = np.array([b["open_ms"] for b in bars], float)
    ok = c > 0
    c, t = c[ok], t[ok]
    return t[1:], np.diff(np.log(c))


def _profile(r, hours, n=24):
    base = float(np.mean(r ** 2))
    out = np.ones(n)
    for h in range(n):
        m = hours == h
        if m.sum() > 5:
            out[h] = float(np.mean(r[m] ** 2)) / base
    return out / out.mean()


def hour_profile(bars, boot=400, seed=0):
    t, r = rets(bars)
    hrs = np.array([dt.datetime.utcfromtimestamp(x / 1000).hour for x in t])
    prof = _profile(r, hrs)

    rng = np.random.default_rng(seed)
    sims = np.zeros((boot, 24))
    n = len(r)
    for i in range(boot):
        idx = rng.integers(0, n, n)
        sims[i] = _profile(r[idx], hrs[idx])
    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)

    half = n // 2
    p1 = _profile(r[:half], hrs[:half])
    p2 = _profile(r[half:], hrs[half:])
    return {"profile": prof, "lo": lo, "hi": hi, "first_half": p1,
            "second_half": p2, "split_corr": float(np.corrcoef(p1, p2)[0, 1]),
            "n": n}


def dow_profile(bars):
    t, r = rets(bars)
    d = np.array([dt.datetime.utcfromtimestamp(x / 1000).weekday() for x in t])
    base = float(np.mean(r ** 2))
    out = np.ones(7)
    for i in range(7):
        m = d == i
        if m.sum() > 5:
            out[i] = float(np.mean(r[m] ** 2)) / base
    return out / out.mean()


def smooth_cyc(p, k=(0.2, 0.6, 1.0, 0.6, 0.2)):
    k = np.array(k, float)
    k /= k.sum()
    pad = len(k) // 2
    x = np.r_[p[-pad:], p, p[:pad]]
    return np.convolve(x, k, mode="same")[pad:-pad]


class Clock:
    """Variance clock combining hour-of-day and day-of-week seasonality."""

    def __init__(self, hour_prof, dow_prof=None):
        h = np.asarray(hour_prof, float)
        self.h = h / h.mean()
        d = np.ones(7) if dow_prof is None else np.asarray(dow_prof, float)
        self.d = d / d.mean()

    def rate(self, ms):
        u = dt.datetime.utcfromtimestamp(ms / 1000)
        return self.h[u.hour] * self.d[u.weekday()]

    def tau(self, t0, t1, sub=4):
        """Business hours between two instants (== calendar hours if flat)."""
        if t1 <= t0:
            return 0.0
        step = 3600_000.0 / sub
        acc, a = 0.0, float(t0)
        while a < t1:
            b = min(a + step, t1)
            acc += self.rate(a) * (b - a) / 3600_000.0
            a = b
        return acc


def deribit_stub_test(clock, label="BTC"):
    """Compare Deribit's front-stub variance rate to its full-day rate.

    Under a flat calendar clock the two are equal. Under the estimated
    business clock the stub (which ends 08:00Z, after the quiet overnight)
    should carry markedly *less* variance per calendar hour.
    """
    from rnd import TermSurface
    d = load("deribit")
    ts = TermSurface(d["ccy"][label], d["fetched_ms"], label)
    sl = sorted(ts.exps.values(), key=lambda s: s["T"])[:4]
    from surface import implied_vol

    rows = []
    for s in sl:
        iv = implied_vol(s["dens"].call(s["F"]), s["F"], s["F"], s["T"])
        rows.append({"T_yr": s["T"], "T_h": s["T"] * 365.25 * 24, "iv": iv,
                     "w": (iv or 0) ** 2 * s["T"], "exp_ms": None})
    exps = sorted(ts.exps.items(), key=lambda x: x[1]["T"])[:4]
    for r, (e, _) in zip(rows, exps):
        r["exp_ms"] = e

    now = d["fetched_ms"]
    out = []
    for i, r in enumerate(rows):
        cal = (r["exp_ms"] - now) / 3600_000.0
        biz = clock.tau(now, r["exp_ms"])
        if i == 0:
            dw, dcal, dbiz = r["w"], cal, biz
        else:
            dw = r["w"] - rows[i - 1]["w"]
            dcal = cal - (rows[i - 1]["exp_ms"] - now) / 3600_000.0
            dbiz = biz - clock.tau(now, rows[i - 1]["exp_ms"])
        out.append({"i": i, "T_h": r["T_h"], "iv": r["iv"], "w": r["w"],
                    "cal_h": cal, "biz_h": biz,
                    "fwd_w": dw, "fwd_cal_h": dcal, "fwd_biz_h": dbiz,
                    "var_per_cal_h": dw / dcal if dcal > 0 else None,
                    "var_per_biz_h": dw / dbiz if dbiz > 0 else None})
    return out


def main():
    hist = load("hist")
    res = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        bars = hist["bars"][f"{sym}_1h"]
        hp = hour_profile(bars)
        dp = dow_profile(bars)
        res[asset] = {"hour": hp["profile"].tolist(), "hour_lo": hp["lo"].tolist(),
                      "hour_hi": hp["hi"].tolist(), "dow": dp.tolist(),
                      "split_corr": hp["split_corr"], "n": hp["n"],
                      "hour_smooth": smooth_cyc(hp["profile"]).tolist()}
        print(f"=== {asset}  n={hp['n']} hourly returns "
              f"({hp['n']/24/365.25:.2f} yr), split-half corr={hp['split_corr']:.3f}")
        print("  hour-of-day variance multiplier [95% bootstrap CI]")
        for h in range(24):
            bar = "#" * int(round(hp["profile"][h] * 18))
            print(f"    {h:02d}Z {hp['profile'][h]:5.2f} "
                  f"[{hp['lo'][h]:4.2f},{hp['hi'][h]:4.2f}]  {bar}")
        print("  day-of-week multiplier: " +
              " ".join(f"{n}={v:.2f}" for n, v in
                       zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dp)))

    ch = np.array(res["BTC"]["hour"])
    ce = np.array(res["ETH"]["hour"])
    print(f"\nBTC vs ETH hour-profile correlation: {np.corrcoef(ch, ce)[0,1]:.3f}")

    clock = Clock(res["BTC"]["hour_smooth"], res["BTC"]["dow"])
    flat = Clock(np.ones(24))
    print("\n=== Q3: does Deribit price the intraday clock? (BTC)")
    print("  exp    T(h)    ATMiv    fwd_var/cal_h   fwd_var/biz_h   ratio(cal)  ratio(biz)")
    rows = deribit_stub_test(clock, "BTC")
    base_cal = rows[1]["var_per_cal_h"] if len(rows) > 1 else None
    base_biz = rows[1]["var_per_biz_h"] if len(rows) > 1 else None
    for r in rows:
        rc = r["var_per_cal_h"] / base_cal if base_cal else float("nan")
        rb = r["var_per_biz_h"] / base_biz if base_biz else float("nan")
        tag = " <-- front stub" if r["i"] == 0 else ""
        print(f"   {r['i']}  {r['T_h']:7.2f} {(r['iv'] or 0)*100:7.2f}%  "
              f"{r['var_per_cal_h']:.3e}      {r['var_per_biz_h']:.3e}    "
              f"{rc:8.3f}    {rb:8.3f}{tag}")
    print("\n  A front-stub ratio near 1.0 in the CALENDAR column means Deribit is")
    print("  pricing the stub on a flat clock. The BUSINESS column shows what the")
    print("  same quotes imply once the estimated seasonality is applied.")
    print(f"  clock says now->first expiry: calendar "
          f"{(rows[0]['cal_h']):.2f}h vs business {rows[0]['biz_h']:.2f}h "
          f"(ratio {rows[0]['biz_h']/rows[0]['cal_h']:.3f})")

    res["clock_hour"] = res["BTC"]["hour_smooth"]
    res["clock_dow"] = res["BTC"]["dow"]
    save("seasonality", res)
    return res


if __name__ == "__main__":
    main()
