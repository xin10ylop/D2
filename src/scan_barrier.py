"""Price every live Kalshi one-touch market and compare with the quotes.

These carry 15.1M contracts of open interest -- an order of magnitude more than
the digital ladders that the first pass looked at -- and they are the only
contract family on the venue whose value depends on the whole path rather than
the settlement print.

Volatility handling:
  BTC / ETH  taken from Deribit's surface, the correct risk-neutral level.
  others     no listed options anywhere, so trailing realised volatility is
             scaled by the implied/realised ratio measured on BTC and ETH at
             the same horizon. That ratio is the variance risk premium, and
             borrowing it is far better than assuming implied = realised.

Every price is reported with a volatility sensitivity band, because a barrier
is much more vol-sensitive than a digital and an edge that disappears under a
+/-20% vol shift is not an edge.
"""
import datetime as dt
import math
import re

import numpy as np

from common import http_json, load, pmap, save, now_ms
from onetouch import BarrierMC, implied_var, realised_vol

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
YEAR_H = 365.25 * 24.0

COIN_RE = re.compile(r"^KX(?P<coin>[A-Z]+?)(MAXMON|MINMON|MAXY|MINY|MAX|MIN)")
ALIAS = {"BITCOIN": "BTC", "ETHEREUM": "ETH"}


def series_coin(series):
    m = COIN_RE.match(series)
    if m:
        c = m.group("coin")
        return ALIAS.get(c, c)
    for c in ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "BCH", "BNB",
              "LINK", "LTC", "DOT", "SHIB", "PEPE", "TRX", "TON", "SUI", "APT"):
        if c in series:
            return c
    return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def iso_ms(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def main(min_edge=0.01):
    uni = load("universe_kalshi")
    coins = load("coins")
    der = load("deribit")
    now = now_ms()

    touch = [r for r in uni if r["kind"] in ("onetouch_max", "onetouch_min")]
    print(f"one-touch events open: {len(touch)}", flush=True)

    # --- variance risk premium, from the two assets that have listed options
    vrp = []
    for a in ("BTC", "ETH"):
        rv = realised_vol(coins["bars"][a], 720)
        w = implied_var(a, 720.0, der["ccy"][a], der["fetched_ms"])
        if w:
            iv = math.sqrt(w / (720.0 / YEAR_H))
            vrp.append(iv / rv)
            print(f"  {a}: 30d realised {rv*100:.1f}%  implied(30d) {iv*100:.1f}%"
                  f"  ratio {iv/rv:.3f}", flush=True)
    VRP = float(np.mean(vrp)) if vrp else 1.0
    print(f"  -> premium applied to coins without options: x{VRP:.3f}\n", flush=True)
    # Horizon-matched forecast (src/volterm.py), calibrated so it reproduces
    # Deribit's implied term structure on BTC/ETH. A flat 30-day realised vol
    # extrapolated to 150 days is what produced the first pass's fictitious
    # 35-cent edges on XRP and DOGE.
    vt = load("volterm")
    from volterm import forecast as vol_forecast
    TAU = vt["tau_days"]

    # Group by coin so only one asset's parsed history is resident at a time.
    by_coin = {}
    for ev in touch:
        c = series_coin(ev["series"])
        if c and c in coins["bars"]:
            by_coin.setdefault(c, []).append(ev)
    print(f"coins with one-touch markets: {sorted(by_coin)}\n", flush=True)

    rows = []
    for coin, evs in sorted(by_coin.items()):
        mc = BarrierMC(coins["bars"][coin], n_paths=20000)
        coins["bars"][coin] = None                  # release the parsed bars
        S0 = coins["spot"][coin]
        print(f"--- {coin} ({len(evs)} events) spot={S0:,.6g}", flush=True)

        for ev in evs:
            try:
                d = http_json(f"{KALSHI}/markets",
                              {"event_ticker": ev["event"], "limit": 200})
            except Exception as e:                  # noqa: BLE001
                print(f"    {ev['event']}: {e}", flush=True)
                continue
            ms = [m for m in d.get("markets", [])
                  if m.get("status") == "active"
                  and fnum(m.get("yes_ask_dollars")) is not None]
            if not ms:
                continue
            H = (iso_ms(ms[0]["close_time"]) - now) / 3600_000.0
            if H <= 2 or H > 24 * 400:
                continue
            if coin in ("BTC", "ETH"):
                w = implied_var(coin, H, der["ccy"][coin], der["fetched_ms"])
            else:
                inp = vt["inputs"][coin]
                sig = vol_forecast(inp["s30"], inp["slong"], H / 24.0, TAU) * VRP
                w = sig ** 2 * H / YEAR_H
            if not w or w <= 0:
                continue

            ups = sorted({fnum(m["floor_strike"]) for m in ms
                          if m.get("strike_type") in ("greater", "greater_or_equal")
                          and fnum(m.get("floor_strike"))})
            dns = sorted({fnum(m["cap_strike"]) for m in ms
                          if m.get("strike_type") in ("less", "less_or_equal")
                          and fnum(m.get("cap_strike"))})

            # one ensemble, three volatility scenarios
            pr = {tag: mc.price(S0, H, w * mult * mult,
                                barriers_up=ups, barriers_dn=dns)
                  for tag, mult in (("lo", 0.8), ("mid", 1.0), ("hi", 1.2))}
            print(f"    {ev['event']:30s} H={H/24:7.1f}d  "
                  f"vol={math.sqrt(w/(H/YEAR_H))*100:5.1f}%  "
                  f"strikes={len(ups)+len(dns)}", flush=True)

            for m in ms:
                stype = m.get("strike_type")
                if stype in ("greater", "greater_or_equal"):
                    B, key = fnum(m.get("floor_strike")), "touch_up"
                elif stype in ("less", "less_or_equal"):
                    B, key = fnum(m.get("cap_strike")), "touch_dn"
                else:
                    continue
                if B is None or B not in pr["mid"][key]:
                    continue
                fair = pr["mid"][key][B]
                f_lo, f_hi = pr["lo"][key][B], pr["hi"][key][B]
                bid = fnum(m.get("yes_bid_dollars")) or 0.0
                ask = fnum(m.get("yes_ask_dollars"))
                if ask is None or not (0 < ask <= 1):
                    continue
                fee_b = 0.07 * ask * (1 - ask)
                fee_s = 0.07 * (1 - bid) * bid
                rows.append({
                    "series": ev["series"], "event": ev["event"], "coin": coin,
                    "ticker": m["ticker"],
                    "dir": "up" if key == "touch_up" else "dn",
                    "barrier": B, "S0": S0, "moneyness": B / S0 - 1.0,
                    "H_days": H / 24.0, "bid": bid, "ask": ask, "fair": fair,
                    "fair_lo": min(f_lo, f_hi), "fair_hi": max(f_lo, f_hi),
                    "buy_edge": fair - (ask + fee_b),
                    "sell_edge": ((1 - fair) - ((1 - bid) + fee_s)) if bid > 0 else None,
                    "oi": fnum(m.get("open_interest_fp")) or 0,
                    "vol24": fnum(m.get("volume_24h_fp")) or 0,
                })
        del mc

    save("barrier_scan", rows)
    print(f"\npriced {len(rows)} one-touch contracts\n", flush=True)

    # Only count an edge that survives the whole volatility band.
    sells = [r for r in rows
             if r["sell_edge"] is not None and r["sell_edge"] > min_edge
             and (1 - r["fair_hi"]) - ((1 - r["bid"])
                                       + 0.07 * (1 - r["bid"]) * r["bid"]) > 0]
    buys = [r for r in rows
            if r["buy_edge"] > min_edge
            and r["fair_lo"] - (r["ask"] + 0.07 * r["ask"] * (1 - r["ask"])) > 0]
    sells.sort(key=lambda r: -r["sell_edge"])
    buys.sort(key=lambda r: -r["buy_edge"])

    def show(name, rs, field):
        print(f"=== {name}: {len(rs)} contracts survive a +/-20% vol shift")
        if not rs:
            print("   none\n")
            return
        print(f"   {'series':17s} {'coin':5s} {'d':2s} {'barrier':>11s} {'move':>7s} "
              f"{'days':>6s} {'bid':>5s} {'ask':>5s} {'fair':>6s} {'band':>15s} "
              f"{'edge':>6s} {'OI':>9s}")
        for r in rs[:25]:
            print(f"   {r['series'][:17]:17s} {r['coin']:5s} {r['dir']:2s} "
                  f"{r['barrier']:11,.5g} {r['moneyness']*100:+6.1f}% "
                  f"{r['H_days']:6.1f} {r['bid']:5.2f} {r['ask']:5.2f} "
                  f"{r['fair']:6.3f} [{r['fair_lo']:6.3f},{r['fair_hi']:6.3f}] "
                  f"{r[field]:+6.3f} {r['oi']:9,.0f}")
        print()

    show("SELL the touch (buy NO)", sells, "sell_edge")
    show("BUY the touch (buy YES)", buys, "buy_edge")
    return rows


if __name__ == "__main__":
    main()
