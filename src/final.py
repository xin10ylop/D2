"""The trade: Polymarket's weekend contracts, priced off Deribit's last expiry.

Construction, in one line: take total implied variance out to Deribit's final
listed expiry (Fri 7 Aug 08:00Z) -- a liquid, arbitraged point -- then extend
it across the weekend using the *directly measured* variance of those exact
clock windows over two years, rescaled so that a midweek window matches what
Deribit itself is charging.

This deliberately uses each source only where it is strong:
  * Deribit sets the volatility LEVEL, because it has a real options market.
  * History supplies only the weekend SHAPE, because no listed instrument
    anywhere in the three venues expires on a Saturday or Sunday.
  * No assumption about realised-vs-implied vol premium is needed: the level
    is taken from implied, not realised.

Everything is then compared with Polymarket's executable quotes, bucket by
bucket, at the size actually resting on the book.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from fitladder import fit_ladder, model_cdf
from rnd import TermSurface
from seasonality import Clock

YEAR_H = 365.25 * 24.0
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def window_var(bars, start_dow, start_h, end_dow, end_h):
    """Realised variance of a repeating weekly clock window, measured directly.

    No multiplicative hour x weekday approximation: the return is taken from
    the actual bar close at the window start to the close at the window end,
    which is precisely what the contracts settle on.
    """
    idx = {}
    for b in bars:
        u = dt.datetime.utcfromtimestamp(b["open_ms"] / 1000)
        if b["c"] > 0:
            idx[(u.date(), u.hour)] = b["c"]
    rs = []
    for (d, h), px in idx.items():
        if d.weekday() != start_dow or h != start_h:
            continue
        span = (end_dow - start_dow) % 7
        if span == 0 and end_h <= start_h:
            span = 7
        d2 = d + dt.timedelta(days=span)
        q = idx.get((d2, end_h))
        if q:
            rs.append(math.log(q / px))
    if len(rs) < 20:
        return None
    a = np.array(rs)
    rng = np.random.default_rng(0)
    sims = np.array([np.mean(rng.choice(a, len(a)) ** 2) for _ in range(2000)])
    return {"n": len(a), "var": float(np.mean(a ** 2)),
            "lo": float(np.percentile(sims, 2.5)),
            "hi": float(np.percentile(sims, 97.5))}


def main():
    poly, ref, hist, seas = load("polymarket"), load("ref"), load("hist"), load("seasonality")
    der = load("deribit")
    now = poly["fetched_ms"]
    spot = ref["spot"]
    usd = {a: float(np.mean([spot[f"{v}_{a}"] for v in
                             ("coinbase", "bitstamp", "gemini", "kraken")
                             if spot.get(f"{v}_{a}")])) for a in ("BTC", "ETH")}
    peg = {a: usd[a] / spot[f"binance_{a}"] for a in ("BTC", "ETH")}
    books = poly.get("books", {})
    clock = Clock(seas["clock_hour"], seas["clock_dow"])

    print(f"snapshot {dt.datetime.utcfromtimestamp(now/1000):%a %d %b %Y %H:%MZ}")
    print("USD composite:", {a: f"{v:,.2f}" for a, v in usd.items()},
          " USDT peg:", {a: f"{(p-1)*1e4:+.2f}bp" for a, p in peg.items()})

    out = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        bars = hist["bars"][f"{sym}_1h"]
        ts = TermSurface(der["ccy"][asset], der["fetched_ms"], asset)

        # --- Deribit anchor: last listed expiry, Fri 7 Aug 08:00Z
        anchors = sorted(ts.exps.items(), key=lambda x: x[1]["T"])
        anch = None
        for e, s in anchors:
            u = dt.datetime.utcfromtimestamp(e / 1000)
            if u.month == 8 and u.day == 7:
                anch = (e, s)
        if anch is None:
            anch = anchors[3]
        e_anchor, s_anchor = anch
        f_anchor = fit_ladder(
            [(float(K), max(s_anchor["dens"].prob_above(float(K)) - 1e-4, 0.0),
              min(s_anchor["dens"].prob_above(float(K)) + 1e-4, 1.0))
             for K in s_anchor["dens"].grid[::8]], s_anchor["T"], s_anchor["F"])
        w_anchor = f_anchor["sigma_atm"] ** 2 * s_anchor["T"]

        # Deribit's own midweek daily variance (Thu 08Z -> Fri 08Z)
        prev = [s for e, s in anchors if e < e_anchor][-1]
        f_prev = fit_ladder(
            [(float(K), max(prev["dens"].prob_above(float(K)) - 1e-4, 0.0),
              min(prev["dens"].prob_above(float(K)) + 1e-4, 1.0))
             for K in prev["dens"].grid[::8]], prev["T"], prev["F"])
        w_prev = f_prev["sigma_atm"] ** 2 * prev["T"]
        der_daily = w_anchor - w_prev

        # --- realised windows, measured directly
        mid = window_var(bars, 3, 8, 4, 8)          # Thu 08Z -> Fri 08Z
        w_sat = window_var(bars, 4, 8, 5, 16)       # Fri 08Z -> Sat 16Z
        w_sun = window_var(bars, 5, 16, 6, 16)      # Sat 16Z -> Sun 16Z
        if not (mid and w_sat and w_sun):
            continue
        scale = der_daily / mid["var"]              # implied-vs-realised level

        print(f"\n=== {asset}")
        print(f"  Deribit anchor  {dt.datetime.utcfromtimestamp(e_anchor/1000):%a %d %b %H:%MZ}"
              f"  total var={w_anchor:.4e}  (ATM {f_anchor['sigma_atm']*100:.2f}%)")
        print(f"  Deribit implied midweek day (Thu08Z->Fri08Z) = {der_daily:.4e}")
        print(f"  realised      midweek day (Thu08Z->Fri08Z) = {mid['var']:.4e}"
              f"  (n={mid['n']})  -> implied/realised level = {scale:.2f}x")
        print(f"  realised Fri08Z->Sat16Z = {w_sat['var']:.4e} "
              f"[{w_sat['lo']:.3e},{w_sat['hi']:.3e}] n={w_sat['n']}"
              f"  = {w_sat['var']/mid['var']:.2f} midweek-days")
        print(f"  realised Sat16Z->Sun16Z = {w_sun['var']:.4e} "
              f"[{w_sun['lo']:.3e},{w_sun['hi']:.3e}] n={w_sun['n']}"
              f"  = {w_sun['var']/mid['var']:.2f} midweek-days")

        fair_w = {8: w_anchor + scale * w_sat["var"],
                  9: w_anchor + scale * (w_sat["var"] + w_sun["var"])}
        fair_lo = {8: w_anchor + scale * w_sat["lo"],
                   9: w_anchor + scale * (w_sat["lo"] + w_sun["lo"])}
        fair_hi = {8: w_anchor + scale * w_sat["hi"],
                   9: w_anchor + scale * (w_sat["hi"] + w_sun["hi"])}

        rows = []
        for day in (8, 9):
            slug = f"{'bitcoin' if asset=='BTC' else 'ethereum'}-price-on-august-{day}-2026"
            ev = poly["events"].get(slug)
            aslug = f"{'bitcoin' if asset=='BTC' else 'ethereum'}-above-on-august-{day}-2026"
            aev = poly["events"].get(aslug)
            if not ev or not aev:
                continue
            t1 = dt.datetime.fromisoformat(ev["end_date"].replace("Z", "+00:00")).timestamp() * 1000
            cal = (t1 - now) / 3600_000.0

            lad = []
            for m in aev["markets"]:
                t = (m["group_title"] or "").replace(",", "").replace("$", "")
                try:
                    lad.append((float(t) * peg[asset], m["best_bid"], m["best_ask"]))
                except ValueError:
                    pass
            mk = fit_ladder(sorted(lad), clock.tau(now, t1) / YEAR_H, usd[asset])
            F = mk["F"]
            mkt_w = mk["sigma_atm"] ** 2 * clock.tau(now, t1) / YEAR_H

            print(f"\n  --- {slug}   settles "
                  f"{dt.datetime.utcfromtimestamp(t1/1000):%a %d %b %H:%MZ} "
                  f"({cal:.0f}h)   implied F={F:,.0f}")
            print(f"      market total var = {mkt_w:.4e}   "
                  f"fair = {fair_w[day]:.4e} [{fair_lo[day]:.3e},{fair_hi[day]:.3e}]"
                  f"   market/fair = {mkt_w/fair_w[day]:.2f}x")

            # per-bucket fair vs executable
            T = clock.tau(now, t1) / YEAR_H
            s_fair = math.sqrt(fair_w[day] / T)
            s_hi = math.sqrt(fair_hi[day] / T)
            import re
            NUM = re.compile(r"[\d,]+")
            print(f"      {'bucket (USDT)':16s} {'bid':>6s} {'ask':>6s} "
                  f"{'mktP':>6s} {'fairP':>6s} {'edge(buy)':>10s} {'edge@hiVol':>11s} "
                  f"{'asksz':>8s}")
            for m in ev["markets"]:
                t = (m["group_title"] or "").replace("$", "").strip()
                nums = [float(x.replace(",", "")) * peg[asset] for x in NUM.findall(t)]
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
                bid, ask = m["best_bid"], m["best_ask"]
                if bid is None or ask is None or ask - bid > 0.05 or ask < 0.05:
                    continue

                def P(sig):
                    a = float(model_cdf([lo], F, T, sig, mk["beta"], mk["gamma"])[0]) \
                        if lo > 0 else 1.0
                    b = float(model_cdf([hi], F, T, sig, mk["beta"], mk["gamma"])[0]) \
                        if math.isfinite(hi) else 0.0
                    return max(a - b, 0.0)

                fp, fp_hi = P(s_fair), P(s_hi)
                bk = books.get(m["slug"], {})
                asz = sum(s for p, s in bk.get("asks", []) if p <= ask + 1e-9)
                edge = fp - ask
                print(f"      {t:16s} {bid:6.3f} {ask:6.3f} "
                      f"{0.5*(bid+ask):6.3f} {fp:6.3f} {edge:+10.3f} "
                      f"{fp_hi-ask:+11.3f} {asz:8,.0f}")
                rows.append({"asset": asset, "day": day, "bucket": t,
                             "lo": lo, "hi": hi, "bid": bid, "ask": ask,
                             "fair": fp, "fair_hivol": fp_hi, "edge_buy": edge,
                             "ask_size": asz, "slug": m["slug"]})
        out[asset] = {"rows": rows, "fair_w": fair_w, "w_anchor": w_anchor,
                      "scale": scale, "der_daily": der_daily,
                      "sat": w_sat, "sun": w_sun, "mid": mid}

    save("final", out)
    return out


if __name__ == "__main__":
    main()
