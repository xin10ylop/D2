"""Build and stress-test the weekend forward-variance trade.

A digital/bucket ladder spans a complete set of states, so any payoff can be
synthesised from it. In particular the variance payoff

    g(S) = ln(S/F)^2

is replicated by buying each bucket i with weight w_i = ln(K_i/F)^2, at a cost
of sum_i w_i * price_i. The expected payoff of that strip is exactly the
contract's total implied variance, so differencing two consecutive days'
strips isolates the *forward* variance over the day between them.

That is the tradable object here. Polymarket lists a ladder for every day
including the weekend; Deribit lists no weekend expiry and Kalshi lists no
weekend event, so the Saturday and Sunday windows have no listed hedge
anywhere in the three-venue universe.

Everything below is priced at executable levels -- the strip is bought at asks
and sold at bids, walking the book -- and every implied number carries a
bootstrap band from resampling each quote inside its own bid/ask, because
differencing two fitted volatilities on thin books amplifies noise and a
signal that does not survive that is not a signal.
"""
import datetime as dt
import math

import numpy as np

from common import load, save
from fitladder import fit_ladder

YEAR_H = 365.25 * 24.0
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def buckets(event, scale):
    """(lo, hi, slug, bid, ask) for a range ladder, on the USD axis."""
    import re
    NUM = re.compile(r"[\d,]+")
    out = []
    for m in event["markets"]:
        t = (m["group_title"] or "").replace("$", "").strip()
        nums = [float(x.replace(",", "")) * scale for x in NUM.findall(t)]
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
        out.append((lo, hi, m["slug"], m["best_bid"], m["best_ask"]))
    return sorted(out)


def strip_cost(bkts, books, F, side="buy", tail_w=1.25):
    """Cost and weights of the ln(K/F)^2 variance strip, walking the book.

    Open-ended tail buckets get a representative point one bucket width beyond
    the edge; their weight is large, so the number is reported alongside the
    tail's share of total cost and should be read with that in mind.
    """
    if not bkts:
        return None
    widths = [b[1] - b[0] for b in bkts if math.isfinite(b[1]) and b[0] > 0]
    wd = float(np.median(widths)) if widths else 1.0
    legs, tot, tail = [], 0.0, 0.0
    for lo, hi, slug, bid, ask in bkts:
        if lo <= 0:
            rep = hi - tail_w * wd
            is_tail = True
        elif not math.isfinite(hi):
            rep = lo + tail_w * wd
            is_tail = True
        else:
            rep = 0.5 * (lo + hi)
            is_tail = False
        if rep <= 0:
            continue
        w = math.log(rep / F) ** 2
        bk = books.get(slug, {})
        if side == "buy":
            lv = sorted(bk.get("asks", []), key=lambda x: x[0])
            px = lv[0][0] if lv else (ask if ask else 1.0)
        else:
            lv = sorted(bk.get("bids", []), key=lambda x: -x[0])
            px = lv[0][0] if lv else (bid if bid else 0.0)
        c = w * px
        tot += c
        if is_tail:
            tail += c
        legs.append({"slug": slug, "lo": lo, "hi": hi, "w": w, "px": px,
                     "cost": c, "tail": is_tail})
    return {"cost": tot, "legs": legs, "tail_share": tail / tot if tot else 0.0}


def boot_fit(ladder, T, F0, n=200, seed=0):
    """Refit the ladder with each quote resampled inside its own bid/ask."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lad = []
        for K, b, a in ladder:
            b = 0.0 if b is None else b
            a = 1.0 if a is None else a
            if b > a:
                b, a = a, b
            p = rng.uniform(b, a)
            lad.append((K, p, p))
        f = fit_ladder(lad, T, F0)
        if f:
            out.append(f["sigma_atm"] ** 2 * T)   # total variance
    return np.array(out)


def main(nboot=200):
    poly, ref, hist = load("polymarket"), load("ref"), load("hist")
    wk = load("weekend")
    rv = {(_r["venue"], _r["asset"], _r["name"]): _r for _r in load("rv_events")}

    spot = ref["spot"]
    usd = {a: float(np.mean([spot[f"{v}_{a}"] for v in
                             ("coinbase", "bitstamp", "gemini", "kraken")
                             if spot.get(f"{v}_{a}")])) for a in ("BTC", "ETH")}
    peg = {a: usd[a] / spot[f"binance_{a}"] for a in ("BTC", "ETH")}
    books = poly.get("books", {})

    # Current vol regime: 30-day realised, so the day-shape is applied to the
    # level the market is actually in rather than a two-year average.
    regime = {}
    for a, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        c = np.array([b["c"] for b in hist["bars"][f"{sym}_1h"][-720:]], float)
        r = np.diff(np.log(c))
        regime[a] = float(np.std(r) * math.sqrt(365.25 * 24))
    print("current 30d realised vol:", {a: f"{v*100:.1f}%" for a, v in regime.items()})

    out = {}
    for asset in ("BTC", "ETH"):
        days = {}
        for slug, e in poly["events"].items():
            if e["asset"] != asset or e["kind"] != "above":
                continue
            key = rv.get(("polymarket", asset, slug))
            if not key:
                continue
            lad = []
            for m in e["markets"]:
                t = (m["group_title"] or "").replace(",", "").replace("$", "")
                try:
                    lad.append((float(t) * peg[asset], m["best_bid"], m["best_ask"]))
                except ValueError:
                    pass
            T = key["tau_h"] / YEAR_H
            bs = boot_fit(sorted(lad), T, usd[asset], nboot)
            settle = dt.datetime.utcfromtimestamp(key["t1"] / 1000)
            days[e["day"]] = {"slug": slug, "settle": settle, "dow": settle.weekday(),
                              "tau_h": key["tau_h"], "cal_h": key["cal_h"],
                              "w": key["sigma_atm"] ** 2 * T,
                              "w_boot": bs, "sigma": key["sigma_atm"]}
        # range-ladder strips, for the executable version of the same object
        for slug, e in poly["events"].items():
            if e["asset"] != asset or e["kind"] != "range":
                continue
            if e["day"] in days:
                bk = buckets(e, peg[asset])
                days[e["day"]]["buy"] = strip_cost(bk, books, usd[asset], "buy")
                days[e["day"]]["sell"] = strip_cost(bk, books, usd[asset], "sell")

        ks = sorted(days)
        base_var = wk["realised"][asset]["midweek_var"]
        scale = (regime[asset] ** 2 / 365.25) / base_var   # regime rescaling
        print(f"\n=== {asset}: forward variance per day, implied vs realised "
              f"(realised rescaled x{scale:.2f} to the current regime)")
        print(f"{'window ends':12s} {'dow':4s} {'impl fwd var':>13s} "
              f"{'95% boot band':>24s} {'realised fwd':>13s} {'impl/real':>10s} "
              f"{'strip buy':>10s} {'strip sell':>10s}")
        rows = []
        for i in range(1, len(ks)):
            a0, a1 = days[ks[i - 1]], days[ks[i]]
            fwd = a1["w"] - a0["w"]
            nb = min(len(a0["w_boot"]), len(a1["w_boot"]))
            if nb < 20:
                continue
            d = a1["w_boot"][:nb] - a0["w_boot"][:nb]
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            lbl = DOW[a1["dow"]]
            st = wk["realised"][asset]["by_dow"].get(lbl)
            real = st["var"] * scale if st else None
            ratio = fwd / real if real else None
            sb = a1.get("buy", {}).get("cost")
            ss = a1.get("sell", {}).get("cost")
            print(f"{a1['settle'].strftime('%a %d %b'):12s} {lbl:4s} "
                  f"{fwd:13.3e} [{lo:10.3e},{hi:10.3e}] {(real or 0):13.3e} "
                  f"{(ratio or 0):10.2f} "
                  f"{(sb if sb else float('nan')):10.4f} "
                  f"{(ss if ss else float('nan')):10.4f}")
            rows.append({"asset": asset, "ends": a1["settle"].isoformat(),
                         "dow": lbl, "fwd_impl": fwd, "boot_lo": float(lo),
                         "boot_hi": float(hi), "fwd_real": real, "ratio": ratio,
                         "day_prev": a0["slug"], "day": a1["slug"],
                         "strip_buy": sb, "strip_sell": ss,
                         "tail_share": a1.get("buy", {}).get("tail_share")})
        out[asset] = rows

    print("\n--- signal survives the bootstrap only where the band excludes the "
          "realised level ---")
    for asset in ("BTC", "ETH"):
        for r in out[asset]:
            if not r["fwd_real"]:
                continue
            rich = r["boot_lo"] > r["fwd_real"]
            cheap = r["boot_hi"] < r["fwd_real"]
            if rich or cheap:
                print(f"  {asset} {r['dow']} window ending {r['ends'][:10]}: "
                      f"{'RICH -> sell variance' if rich else 'CHEAP -> buy variance'}"
                      f"  implied={r['fwd_impl']:.3e} "
                      f"band=[{r['boot_lo']:.3e},{r['boot_hi']:.3e}] "
                      f"realised={r['fwd_real']:.3e} ({r['ratio']:.2f}x)")
    save("trade", out)
    return out


if __name__ == "__main__":
    main()
