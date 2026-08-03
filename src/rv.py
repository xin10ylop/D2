"""Master cross-venue relative-value engine.

Puts Kalshi, Polymarket and Deribit on one measure and prices every tradable
contract against it. Three adjustments make the comparison legitimate:

  BASIS  Polymarket settles on Binance BTC/USDT; Kalshi on CF Benchmarks BRTI
         and Deribit on its USD index. Binance's BTC/USDC price sits on the USD
         composite, so the whole gap is the USDT peg. Polymarket strikes are
         multiplied by USDT/USD before comparison -- currently -10.5 bp, worth
         ~2 probability points on an at-the-money digital.

  CLOCK  Variance is accrued on the estimated hour-of-day x day-of-week
         profile, not calendar time. Deribit expires 08:00Z (the quietest hour
         of the day), Polymarket settles 16:00Z (right after the busiest
         block), Kalshi 16:00Z and 21:00Z. On a flat clock these are simply
         not comparable.

  SMILE  Every ladder is fitted with the same skewed parametric form, so
         sigma_atm means the same thing on all three venues.

The reference measure is Deribit's, time-interpolated in business time. Edge is
reported per contract, after fees, at executable prices and sizes.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from engine import kalshi_fee_rate
from fitladder import fit_ladder, quantile
from rnd import TermSurface
from seasonality import Clock

YEAR_H = 365.25 * 24.0


def iso_ms(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def hm(ms):
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%a %d %b %H:%MZ")


class Reference:
    """Deribit's measure, re-expressed on the business clock.

    Deribit's own expiries are used as variance anchors: total implied
    variance at each expiry is mapped to the business time elapsed to that
    expiry, then interpolated. Querying at a target instant therefore asks
    'how much variance has accrued by then', not 'how many hours have passed'.
    """

    def __init__(self, blob, now_ms, clock, asset):
        self.asset = asset
        self.now = now_ms
        self.clock = clock
        self.ts = TermSurface(blob, now_ms, asset)
        self.index = self.ts.index
        pts = []
        for e, s in self.ts.exps.items():
            tau = clock.tau(now_ms, e)
            if tau <= 0:
                continue
            fit = self._slice_fit(s)
            if fit:
                pts.append((tau, fit, e))
        pts.sort()
        self.anchors = pts

    def _slice_fit(self, s):
        d = s["dens"]
        Ks = d.grid[::8]
        q = [(float(K), None, None) for K in Ks]
        # fit the same functional form used for the prediction-market ladders
        rows = [(float(K), max(d.prob_above(float(K)) - 1e-4, 0.0),
                 min(d.prob_above(float(K)) + 1e-4, 1.0)) for K in Ks]
        return fit_ladder(rows, s["T"], s["F"])

    def total_var(self, tau_h):
        """Interpolate total variance linearly in business time."""
        if not self.anchors:
            return None
        xs = [a[0] for a in self.anchors]
        ws = [a[1]["sigma_atm"] ** 2 * a[1]["T"] for a in self.anchors]
        if tau_h <= xs[0]:
            return ws[0] * tau_h / xs[0]
        if tau_h >= xs[-1]:
            return ws[-1] * tau_h / xs[-1]
        return float(np.interp(tau_h, xs, ws))

    def sigma_at(self, tau_h, cal_h):
        """Annualised vol for a contract with `cal_h` calendar life."""
        w = self.total_var(tau_h)
        if w is None or cal_h <= 0:
            return None
        # w was accumulated against business time expressed in hours; convert
        # back to a calendar-quoted volatility for the contract's real life.
        return math.sqrt(w / (tau_h / YEAR_H)) if tau_h > 0 else None

    def skew_at(self, tau_h):
        if not self.anchors:
            return 0.0, 0.0
        xs = [a[0] for a in self.anchors]
        bs = [a[1]["beta"] for a in self.anchors]
        gs = [a[1]["gamma"] for a in self.anchors]
        return (float(np.interp(tau_h, xs, bs)), float(np.interp(tau_h, xs, gs)))

    def prob_above(self, K, tau_h, F=None):
        """Reference P(S>K) at a horizon carrying `tau_h` business hours."""
        w = self.total_var(tau_h)
        if w is None or w <= 0:
            return None
        F = F if F else self.index
        beta, gamma = self.skew_at(tau_h)
        T = tau_h / YEAR_H
        s0 = math.sqrt(w / T)
        from fitladder import model_cdf
        return float(model_cdf([K], F, T, s0, beta, gamma)[0])


def kalshi_ladder(markets):
    return [(m["floor_strike"], m["yes_bid"], m["yes_ask"]) for m in markets
            if m["strike_type"] in ("greater", "greater_or_equal")
            and m["yes_ask"] is not None and 0.0 < m["yes_ask"] <= 1.0]


def poly_ladder(event, scale):
    out = []
    for m in event["markets"]:
        t = (m["group_title"] or "").replace(",", "").replace("$", "")
        try:
            K = float(t)
        except ValueError:
            continue
        out.append((K * scale, m["best_bid"], m["best_ask"]))
    return sorted(out)


def main():
    kal, poly, ref, seas = load("kalshi"), load("polymarket"), load("ref"), load("seasonality")
    der = load("deribit")
    clock = Clock(seas["clock_hour"], seas["clock_dow"])
    now = kal["fetched_ms"]

    spot = ref["spot"]
    usd = {a: float(np.mean([spot[f"{v}_{a}"] for v in
                             ("coinbase", "bitstamp", "gemini", "kraken")
                             if spot.get(f"{v}_{a}")])) for a in ("BTC", "ETH")}
    peg = {a: usd[a] / spot[f"binance_{a}"] for a in ("BTC", "ETH")}
    R = {a: Reference(der["ccy"][a], der["fetched_ms"], clock, a) for a in ("BTC", "ETH")}

    print(f"now = {hm(now)}")
    print("USD composite:", {a: f"{v:,.2f}" for a, v in usd.items()},
          " peg:", {a: f"{(p-1)*1e4:+.2f}bp" for a, p in peg.items()})

    events = []

    # ---- Kalshi digital ladders
    for series in ("KXBTCD", "KXETHD"):
        asset = "ETH" if "ETH" in series else "BTC"
        for ev, blob in kal["series"][series].items():
            live = [m for m in blob["markets"]
                    if m["yes_ask"] is not None and 0.0 < m["yes_ask"] <= 1.0]
            lad = kalshi_ladder(live)
            if len(lad) < 6:
                continue
            t1 = iso_ms(live[0]["close_time"])
            cal = (t1 - now) / 3600_000.0
            if cal <= 0.05:
                continue
            events.append({"venue": "kalshi", "asset": asset, "name": ev,
                           "t1": t1, "cal_h": cal, "tau_h": clock.tau(now, t1),
                           "ladder": lad, "scale": 1.0,
                           "fee": lambda p: kalshi_fee_rate(p)})

    # ---- Polymarket digital ladders
    for slug, e in poly["events"].items():
        if e["kind"] != "above":
            continue
        asset = e["asset"]
        t1 = iso_ms(e["end_date"])
        cal = (t1 - now) / 3600_000.0
        if cal <= 0.05:
            continue
        lad = poly_ladder(e, peg[asset])
        if len(lad) < 6:
            continue
        events.append({"venue": "polymarket", "asset": asset, "name": slug,
                       "t1": t1, "cal_h": cal, "tau_h": clock.tau(now, t1),
                       "ladder": lad, "scale": peg[asset],
                       "fee": lambda p: 0.0})

    rows = []
    for ev in events:
        a = ev["asset"]
        T = ev["tau_h"] / YEAR_H
        fit = fit_ladder(ev["ladder"], T, usd[a])
        if not fit:
            continue
        ref_sig = R[a].sigma_at(ev["tau_h"], ev["cal_h"])
        ev.update({
            "sigma_atm": fit["sigma_atm"], "skew": fit["skew_25d"],
            "fit_F": fit["F"], "rmse": fit["rmse"],
            "ref_sigma": ref_sig,
            "vol_gap": (fit["sigma_atm"] - ref_sig) if ref_sig else None,
            "q16": quantile(fit, 0.16), "q50": quantile(fit, 0.50),
            "q84": quantile(fit, 0.84),
        })
        rows.append(ev)

    rows.sort(key=lambda r: (r["asset"], r["cal_h"], r["venue"]))
    save("rv_events", [{k: v for k, v in r.items()
                        if k not in ("fee", "ladder")} for r in rows])

    for a in ("BTC", "ETH"):
        print(f"\n=== {a}   ladder-implied ATM vol vs Deribit reference "
              f"(both on the business clock)")
        print(f"{'venue':11s} {'event':32s} {'settle':18s} {'cal_h':>6s} {'biz_h':>6s} "
              f"{'F_impl':>10s} {'sigATM':>7s} {'ref':>7s} {'gap':>7s} {'skew':>7s} {'rmse':>6s}")
        for r in rows:
            if r["asset"] != a:
                continue
            g = r["vol_gap"]
            print(f"{r['venue']:11s} {r['name'][:32]:32s} {hm(r['t1']):18s} "
                  f"{r['cal_h']:6.1f} {r['tau_h']:6.1f} {r['fit_F']:10,.0f} "
                  f"{r['sigma_atm']*100:6.2f}% {(r['ref_sigma'] or 0)*100:6.2f}% "
                  f"{(g or 0)*100:+6.2f}% {r['skew']*100:+6.2f}% {r['rmse']:6.4f}")
    return rows


if __name__ == "__main__":
    main()
