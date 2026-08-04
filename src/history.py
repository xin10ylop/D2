"""Kalshi quote history -- the data that turns a snapshot into a strategy.

Everything so far priced these contracts as hold-to-settlement. They are not:
they trade continuously, so a position can be closed the moment the dislocation
converges, and capital velocity -- not the edge per trade -- is what sets the
return.

Kalshi publishes bid/ask candlesticks per market, which lets us ask the two
questions a bot actually needs answered:

  * how long has this dislocation been open?  If it has persisted for weeks it
    will probably persist to settlement, and the position must be sized as a
    28-day hold. If it opens and closes daily, the same capital earns the edge
    many times over.
  * what exit prices were actually available?  The exit is the other half of
    the trade and it has to be measured, not assumed.
"""
import datetime as dt
import time

import numpy as np

from common import http_json, pmap, save

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def fnum(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def candles(series, ticker, days=30, interval=60):
    """(ts, bid, ask, volume, oi) history for one market."""
    now = int(time.time())
    out = []
    # the endpoint caps the span, so walk backwards in chunks
    end = now
    while end > now - days * 86400:
        start = max(end - 5000 * interval * 60, now - days * 86400)
        try:
            d = http_json(f"{KALSHI}/series/{series}/markets/{ticker}/candlesticks",
                          {"start_ts": start, "end_ts": end,
                           "period_interval": interval})
        except Exception:                                # noqa: BLE001
            break
        cs = d.get("candlesticks", [])
        if not cs:
            break
        for c in cs:
            out.append({
                "ts": c["end_period_ts"],
                "bid": fnum(c, "yes_bid", "close_dollars"),
                "ask": fnum(c, "yes_ask", "close_dollars"),
                "vol": fnum(c, "volume_fp") or 0.0,
                "oi": fnum(c, "open_interest_fp") or 0.0,
            })
        if start <= now - days * 86400:
            break
        end = start - 1
    seen, ded = set(), []
    for r in sorted(out, key=lambda x: x["ts"]):
        if r["ts"] not in seen:
            seen.add(r["ts"])
            ded.append(r)
    return ded


def lock_history(near, far, days=30):
    """Edge of the locked pair through time: 1 - (1-near_bid) - far_ask - fees."""
    a = {r["ts"]: r for r in candles(near[0], near[1], days)}
    b = {r["ts"]: r for r in candles(far[0], far[1], days)}
    ts = sorted(set(a) & set(b))
    rows = []
    for t in ts:
        nb, fa = a[t]["bid"], b[t]["ask"]
        if nb is None or fa is None or nb <= 0 or fa <= 0:
            continue
        cost_no = 1.0 - nb
        edge = 1.0 - (cost_no + 0.07 * cost_no * (1 - cost_no)
                      + fa + 0.07 * fa * (1 - fa))
        rows.append({"ts": t, "near_bid": nb, "far_ask": fa,
                     "edge": edge,
                     "when": dt.datetime.utcfromtimestamp(t).isoformat()})
    return rows


def main():
    pairs = [
        ("KXBNBMAXMON", "KXBNBMAXMON-BNB-26AUG31-65000",
         "KXBNBMAXY", "KXBNBMAXY-BNB-26DEC31-65000", "BNB @650"),
        ("KXBNBMAXMON", "KXBNBMAXMON-BNB-26AUG31-64000",
         "KXBNBMAXY", "KXBNBMAXY-BNB-26DEC31-64000", "BNB @640"),
        ("KXBNBMAXMON", "KXBNBMAXMON-BNB-26AUG31-66000",
         "KXBNBMAXY", "KXBNBMAXY-BNB-26DEC31-66000", "BNB @660"),
    ]
    out = {}
    for ns, nt, fs, ft, label in pairs:
        rows = lock_history((ns, nt), (fs, ft), days=30)
        if not rows:
            print(f"{label}: no overlapping history")
            continue
        e = np.array([r["edge"] for r in rows])
        pos = e > 0.005
        # longest unbroken run of a positive edge, in hours
        runs, cur = [], 0
        for v in pos:
            cur = cur + 1 if v else 0
            runs.append(cur)
        out[label] = rows
        print(f"\n=== {label}   {len(rows)} hourly observations "
              f"({(rows[-1]['ts']-rows[0]['ts'])/86400:.1f} days)")
        print(f"   edge now      {e[-1]:+.3f}")
        print(f"   edge mean     {e.mean():+.3f}   min {e.min():+.3f}   max {e.max():+.3f}")
        print(f"   hours with a positive edge: {int(pos.sum())} of {len(pos)} "
              f"({pos.mean()*100:.0f}%)")
        print(f"   longest continuous open window: {max(runs)} hours")
        print(f"   best entry seen: edge {e.max():+.3f} at "
              f"{rows[int(np.argmax(e))]['when'][:16]}")
    save("lock_history", out)
    return out


if __name__ == "__main__":
    main()
