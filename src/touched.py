"""Has the barrier already been touched? The check the Monte Carlo cannot make.

Kalshi's one-touch markets resolve on the extreme reached "after issuance and
through" the close -- and issuance is often weeks in the past. The Monte Carlo
prices only the remaining path from today's spot, which is the right object for
an untouched barrier and completely wrong for one that has already been hit: a
touched contract is worth exactly $1 whatever happens next.

So before any model runs, walk each market's realised high and low from its own
open_time to now and check the barrier directly. A touched barrier still quoted
below $1 is not a modelling opinion; it is a settled outcome that the book has
not repriced.

The comparison uses Binance USDT bars while Kalshi settles on its own USD
reference, so a touch is only reported when the realised extreme clears the
barrier by more than the index basis plus a safety margin.
"""
import datetime as dt

import numpy as np

from common import http_json, load, pmap, save, now_ms

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
MARGIN = 0.004        # 40 bp: covers the USDT/USD basis and venue dispersion


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def iso_ms(s):
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def extremes(bars, t0_ms, t1_ms=None):
    t1_ms = t1_ms or now_ms()
    hi, lo = None, None
    for b in bars:
        if b["open_ms"] < t0_ms or b["open_ms"] > t1_ms:
            continue
        hi = b["h"] if hi is None else max(hi, b["h"])
        lo = b["l"] if lo is None else min(lo, b["l"])
    return hi, lo


def main():
    from scan_barrier import series_coin
    uni = load("universe_kalshi")
    coins = load("coins")
    touch = [r for r in uni if r["kind"] in ("onetouch_max", "onetouch_min")]
    print(f"checking {len(touch)} one-touch events for already-touched barriers\n")

    rows = []
    for ev in touch:
        coin = series_coin(ev["series"])
        if not coin or coin not in coins["bars"]:
            continue
        try:
            d = http_json(f"{KALSHI}/markets",
                          {"event_ticker": ev["event"], "limit": 200})
        except Exception:                                # noqa: BLE001
            continue
        bars = coins["bars"][coin]
        for m in d.get("markets", []):
            if m.get("status") != "active":
                continue
            t0 = iso_ms(m.get("open_time"))
            if not t0:
                continue
            hi, lo = extremes(bars, t0)
            if hi is None:
                continue
            st = m.get("strike_type")
            if st in ("greater", "greater_or_equal"):
                B = fnum(m.get("floor_strike"))
                if B is None:
                    continue
                hit = hi >= B * (1.0 + MARGIN)
                reach = hi / B - 1.0
            elif st in ("less", "less_or_equal"):
                B = fnum(m.get("cap_strike"))
                if B is None:
                    continue
                hit = lo <= B * (1.0 - MARGIN)
                reach = B / lo - 1.0
            else:
                continue
            bid = fnum(m.get("yes_bid_dollars")) or 0.0
            ask = fnum(m.get("yes_ask_dollars"))
            rows.append({
                "event": ev["event"], "series": ev["series"], "coin": coin,
                "ticker": m["ticker"], "dir": "up" if "greater" in (st or "") else "dn",
                "barrier": B, "since": m.get("open_time"),
                "days_open": (now_ms() - t0) / 86400_000.0,
                "hi": hi, "lo": lo, "touched": bool(hit), "reach_pct": reach * 100,
                "bid": bid, "ask": ask,
                "oi": fnum(m.get("open_interest_fp")) or 0,
                "free_edge": (1.0 - (ask + 0.07 * ask * (1 - ask))) if (hit and ask) else None,
            })

    save("touched", rows)
    hits = [r for r in rows if r["touched"]]
    print(f"markets checked: {len(rows)}   already touched: {len(hits)}\n")
    if hits:
        hits.sort(key=lambda r: -(r["free_edge"] or 0))
        print(f"{'series':17s} {'coin':5s} {'d':2s} {'barrier':>11s} {'extreme':>11s} "
              f"{'past by':>8s} {'open':>6s} {'bid':>5s} {'ask':>5s} {'edge':>7s} {'OI':>9s}")
        for r in hits:
            ext = r["hi"] if r["dir"] == "up" else r["lo"]
            print(f"{r['series'][:17]:17s} {r['coin']:5s} {r['dir']:2s} "
                  f"{r['barrier']:11,.5g} {ext:11,.5g} {r['reach_pct']:7.2f}% "
                  f"{r['days_open']:5.1f}d {r['bid']:5.2f} "
                  f"{(r['ask'] if r['ask'] else 0):5.2f} "
                  f"{(r['free_edge'] or 0):+7.3f} {r['oi']:9,.0f}")
    else:
        print("none — every live barrier is still untouched since its issuance")

    # near misses are worth knowing: they are the ones most likely to resolve
    near = [r for r in rows if not r["touched"] and -3.0 < r["reach_pct"] < 0]
    near.sort(key=lambda r: -r["reach_pct"])
    if near:
        print(f"\nwithin 3% of touching ({len(near)}):")
        for r in near[:15]:
            ext = r["hi"] if r["dir"] == "up" else r["lo"]
            print(f"  {r['series'][:17]:17s} {r['coin']:5s} {r['dir']:2s} "
                  f"barrier={r['barrier']:,.5g} extreme={ext:,.5g} "
                  f"({r['reach_pct']:+.2f}%) bid={r['bid']:.2f} ask={r['ask']}")
    return rows


if __name__ == "__main__":
    main()
