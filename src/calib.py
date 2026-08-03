"""Calibrate the two adjustments that dominate any cross-venue comparison.

(1) BUSINESS TIME. Deribit's dailies expire 08:00Z; Polymarket settles 16:00Z;
    Kalshi settles 16:00Z and 21:00Z. Interpolating implied variance linearly
    in calendar time assumes crypto variance accrues uniformly round the clock.
    It does not -- the US session carries materially more of it. We estimate an
    hour-of-day variance profile theta(h) from a year of hourly bars and define
    business time tau(t) = integral of theta. Implied variance is then
    interpolated in tau, not t, which is what makes an 08:00Z option able to
    price a 21:00Z event.

(2) INDEX BASIS. Kalshi settles on CF Benchmarks BRTI, built from USD spot
    venues. Polymarket settles on Binance BTC/USDT. Binance's BTC/USDC price
    sits on top of the USD composite, so essentially the whole gap is the
    USDT peg: Polymarket's print is the Kalshi print divided by USDT/USD.
    That is a level shift of roughly 6 bp today -- worth ~2 probability points
    on an at-the-money digital, i.e. far larger than the edges being hunted.
    We measure its level and, more importantly, its volatility, because that
    volatility is the true residual risk of a Kalshi-vs-Polymarket trade.
"""
import datetime as dt
import math

import numpy as np

from common import load, save


def hourly_returns(bars):
    c = np.array([b["c"] for b in bars], float)
    t = np.array([b["open_ms"] for b in bars], float)
    r = np.diff(np.log(c))
    return t[1:], r


def intraday_profile(bars, min_obs=8):
    """theta(h): variance in hour-of-day h relative to the daily mean."""
    t, r = hourly_returns(bars)
    hours = np.array([dt.datetime.utcfromtimestamp(x / 1000).hour for x in t])
    prof = np.ones(24)
    base = float(np.mean(r ** 2))
    for h in range(24):
        m = hours == h
        if m.sum() >= min_obs:
            prof[h] = float(np.mean(r[m] ** 2)) / base
    # Light smoothing: hour-of-day variance is a smooth seasonal, and the raw
    # per-hour estimates are noisy with ~40 observations each.
    k = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    k = k / k.sum()
    sm = np.convolve(np.r_[prof[-2:], prof, prof[:2]], k, mode="same")[2:-2]
    return sm / sm.mean(), base


class BusinessClock:
    """Maps wall-clock instants to accumulated variance ('business time')."""

    def __init__(self, profile):
        self.profile = np.asarray(profile, float)
        self.profile = self.profile / self.profile.mean()

    def tau(self, t0_ms, t1_ms, steps_per_hour=4):
        """Integral of theta between two instants, in units of *hours*.

        Returns calendar hours when the profile is flat, so it drops straight
        into any formula that currently uses elapsed time.
        """
        if t1_ms <= t0_ms:
            return 0.0
        step_ms = 3600_000.0 / steps_per_hour
        n = int(math.ceil((t1_ms - t0_ms) / step_ms))
        acc = 0.0
        for i in range(n):
            a = t0_ms + i * step_ms
            b = min(a + step_ms, t1_ms)
            h = dt.datetime.utcfromtimestamp(a / 1000).hour
            acc += self.profile[h] * (b - a) / 3600_000.0
        return acc


def align(a, b, key="open_ms"):
    """Inner-join two bar series on timestamp."""
    da = {int(x[key]): x["c"] for x in a}
    db = {int(x[key]): x["c"] for x in b}
    ks = sorted(set(da) & set(db))
    return np.array(ks), np.array([da[k] for k in ks]), np.array([db[k] for k in ks])


def basis_stats(ref, asset="BTC"):
    """Distribution of log(Binance USDT print / USD-composite print)."""
    bn = ref["basis"][f"binance_{asset}USDT"]
    kr_key = "kraken_XXBTZUSD" if asset == "BTC" else "kraken_XETHZUSD"
    kr = ref["basis"].get(kr_key, [])
    cb = ref["basis"].get(f"coinbase_{asset}-USD", [])

    out = {}
    for name, other in (("kraken", kr), ("coinbase", cb)):
        if not other:
            continue
        ts, pb, po = align(bn, other)
        if len(ts) < 30:
            continue
        lb = np.log(pb / po) * 1e4  # basis points
        d = np.diff(lb)
        out[name] = {
            "n": int(len(lb)),
            "mean_bp": float(lb.mean()),
            "median_bp": float(np.median(lb)),
            "sd_bp": float(lb.std()),
            "p05_bp": float(np.percentile(lb, 5)),
            "p95_bp": float(np.percentile(lb, 95)),
            "last_bp": float(lb[-1]),
            # AR(1) on the level: how fast does a dislocation decay?
            "ar1": float(np.corrcoef(lb[:-1], lb[1:])[0, 1]),
            "sd_1h_change_bp": float(d.std()),
            "sd_24h_change_bp": float(np.std(lb[24:] - lb[:-24])) if len(lb) > 30 else None,
            "sd_120h_change_bp": float(np.std(lb[120:] - lb[:-120])) if len(lb) > 150 else None,
        }
    return out


def realised_vol(bars, hours=24):
    t, r = hourly_returns(bars)
    ann = math.sqrt(365.25 * 24)
    out = {}
    for w in (24, 72, 168, 336, 720):
        if len(r) > w:
            out[f"rv_{w}h"] = float(np.std(r[-w:]) * ann)
    out["rv_all"] = float(np.std(r) * ann)
    return out


def main():
    ref = load("ref")
    res = {}
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        bars = ref["bars"][f"{sym}_1h"]
        prof, base = intraday_profile(bars)
        res[asset] = {
            "intraday_profile": prof.tolist(),
            "realised": realised_vol(bars),
            "basis": basis_stats(ref, asset),
        }
        print(f"=== {asset}  ({len(bars)} hourly bars)")
        print("  realised vol:", {k: f"{v*100:.1f}%" for k, v in res[asset]["realised"].items()})
        print("  intraday variance multiplier by UTC hour:")
        for row in range(0, 24, 8):
            print("   ", " ".join(f"{h:02d}h:{prof[h]:.2f}" for h in range(row, row + 8)))
        for k, v in res[asset]["basis"].items():
            print(f"  basis vs {k}: mean={v['mean_bp']:+.2f}bp sd={v['sd_bp']:.2f}bp "
                  f"last={v['last_bp']:+.2f}bp ar1={v['ar1']:.3f} "
                  f"sd(1h d)={v['sd_1h_change_bp']:.2f}bp "
                  f"sd(24h d)={(v['sd_24h_change_bp'] or 0):.2f}bp "
                  f"sd(120h d)={(v['sd_120h_change_bp'] or 0):.2f}bp")

    # Effective variance share: how much of a calendar window each venue's
    # settlement hour actually captures.
    clk = BusinessClock(res["BTC"]["intraday_profile"])
    now = ref["fetched_ms"]
    print("\n  business-time vs calendar hours from now:")
    for lbl, iso in (("Kalshi 16:00Z Aug3", "2026-08-03T16:00:00"),
                     ("Kalshi 21:00Z Aug3", "2026-08-03T21:00:00"),
                     ("Deribit 08:00Z Aug4", "2026-08-04T08:00:00"),
                     ("Poly 16:00Z Aug4", "2026-08-04T16:00:00"),
                     ("Deribit 08:00Z Aug7", "2026-08-07T08:00:00"),
                     ("Poly 16:00Z Aug7", "2026-08-07T16:00:00"),
                     ("Kalshi 21:00Z Aug7", "2026-08-07T21:00:00")):
        t1 = dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp() * 1000
        cal = (t1 - now) / 3600_000.0
        bt = clk.tau(now, t1)
        print(f"    {lbl:22s} calendar={cal:7.2f}h  business={bt:7.2f}h  ratio={bt/cal:.4f}")

    save("calib", res)
    return res


if __name__ == "__main__":
    main()
