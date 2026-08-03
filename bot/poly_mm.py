"""Market maker for Polymarket range ladders -- the only book the map says pays.

The opportunity map (src/mmmap.py) is unambiguous: after charging the model's
1c error and adverse selection, Polymarket's ETH range ladders clear +1.45c per
contract at a 30% adverse-selection assumption, on ~3,000 contracts of resting
depth per bucket. Every other book in the three-venue universe is negative or
marginal -- Kalshi's BTC digital ladder carries 2 million contracts a day and a
1c spread, and quoting it loses 0.9c per fill.

Two edges stack on the same instrument:
  SPREAD   7c median on ETH range, quoted two-sided around a calibrated fair.
  WEEKEND  the Sat->Sun forward variance is priced 2.46x what it realises
           (src/final.py), so the Sunday ladder's fair distribution is
           narrower than the market's -- body buckets are cheap, wings rich.

Live execution needs an EIP-712 signed order via py-clob-client and a funded
Polygon wallet; this module produces the exact quote plan and runs paper by
default.
"""
import datetime as dt
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/user/D2/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import http_json, load                      # noqa: E402
from howclock import WeekClock                          # noqa: E402

from config import CFG                                  # noqa: E402
from pricing import Pricer, recalibrate, sharpen_b      # noqa: E402
from venues import Deribit                              # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
NUM = re.compile(r"[\d,]+")
YEAR_H = 365.25 * 24.0

# Realised variance of each 16:00Z->16:00Z window as a multiple of a midweek
# day, measured over two years in src/weekend.py. Used to tilt the fair
# distribution on days Deribit does not list.
DOW_VAR = {"BTC": {"Mon": 0.96, "Tue": 1.09, "Wed": 0.97, "Thu": 0.95,
                   "Fri": 0.78, "Sat": 0.24, "Sun": 0.39},
           "ETH": {"Mon": 1.21, "Tue": 1.05, "Wed": 0.96, "Thu": 0.99,
                   "Fri": 0.97, "Sat": 0.43, "Sun": 0.60}}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def bucket_bounds(title, scale=1.0):
    t = (title or "").replace("$", "").strip()
    nums = [float(x.replace(",", "")) * scale for x in NUM.findall(t)]
    if not nums:
        return None
    if t.startswith("<"):
        return 0.0, nums[0]
    if t.startswith(">"):
        return nums[0], float("inf")
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


def peg(asset):
    """USDT/USD: Polymarket strikes are on a USDT axis, the model on USD."""
    usd = []
    for u, f in (("https://api.exchange.coinbase.com/products/"
                  f"{'BTC' if asset=='BTC' else 'ETH'}-USD/ticker", "price"),):
        try:
            usd.append(float(http_json(u)[f]))
        except Exception:                                # noqa: BLE001
            pass
    try:
        b = float(http_json("https://data-api.binance.vision/api/v3/ticker/price",
                            {"symbol": f"{asset}USDT"})["price"])
    except Exception:                                    # noqa: BLE001
        return 1.0
    return (sum(usd) / len(usd) / b) if usd else 1.0


def event(slug):
    d = http_json(f"{GAMMA}/events", {"slug": slug})
    return d[0] if d else None


def book(token_id):
    return http_json(f"{CLOB}/book", {"token_id": token_id})


class PolyMM:
    def __init__(self, cfg=CFG, assets=("ETH", "BTC")):
        self.cfg = cfg
        self.assets = assets
        how = load("howclock")
        self.clocks = {a: WeekClock(how[a]["clock"]) for a in ("BTC", "ETH")}
        self.deribit = Deribit(cfg)
        self.pricers = {}

    def refresh(self):
        now = int(dt.datetime.utcnow().timestamp() * 1000)
        for a in self.assets:
            blob = self.deribit.snapshot(a)
            p = Pricer(blob, now, self.clocks[a], a)
            if p.ok:
                self.pricers[a] = p
        return now

    def weekend_tilt(self, asset, settle_dt, st):
        """Scale total variance by the realised day-shape past Deribit's last expiry.

        Deribit lists nothing on a Saturday or Sunday, so beyond its final
        Friday expiry the model has no market anchor. Rather than extrapolate a
        flat rate -- which is what the market appears to be doing -- carry
        forward the measured variance of each weekday window.
        """
        anchors = self.pricers[asset].anchors
        if not anchors:
            return st["w"], 1.0
        settle_ms = settle_dt.replace(tzinfo=dt.timezone.utc).timestamp() * 1000
        before = [a for a in anchors if a["exp_ms"] <= settle_ms]
        if not before:
            return st["w"], 1.0
        last = before[-1]
        # Only day-shape across a genuine listing gap. Deribit lists dailies
        # Mon-Fri then jumps a week, so a settlement more than ~20h past the
        # last listed expiry is in unhedged territory.
        if (settle_ms - last["exp_ms"]) / 3600_000.0 < 20.0:
            return st["w"], 1.0
        # variance up to the last listed expiry, then day-shaped beyond it
        w_anchor = last["w"]
        rate_mid = w_anchor / max(last["tau"], 1e-9)      # per business hour
        end = dt.datetime.utcfromtimestamp(last["exp_ms"] / 1000)
        add, cur = 0.0, end
        # a midweek day of business time, for scaling the day multipliers
        day_tau = 24.0
        while cur < settle_dt:
            nxt = min(cur + dt.timedelta(days=1), settle_dt)
            frac = (nxt - cur).total_seconds() / 86400.0
            mult = DOW_VAR[asset].get(DOW[nxt.weekday()], 1.0)
            add += rate_mid * day_tau * mult * frac
            cur = nxt
        w = w_anchor + add
        return w, w / max(st["w"], 1e-12)

    def plan(self, asset, day, verbose=True):
        pr = self.pricers.get(asset)
        if not pr:
            return []
        slug = f"{'bitcoin' if asset=='BTC' else 'ethereum'}-price-on-august-{day}-2026"
        ev = event(slug)
        if not ev:
            return []
        sc = peg(asset)
        settle_dt = dt.datetime.fromisoformat(ev["endDate"].replace("Z", "+00:00")).replace(tzinfo=None)
        settle_ms = settle_dt.replace(tzinfo=dt.timezone.utc).timestamp() * 1000
        st = pr.state(settle_ms)
        if not st:
            return []
        w, tilt = self.weekend_tilt(asset, settle_dt, st)
        st = dict(st, w=w, sigma=math.sqrt(w / st["T"]))

        rows = []
        for m in ev.get("markets", []):
            if m.get("closed"):
                continue
            bd = bucket_bounds(m.get("groupItemTitle"), sc)
            if not bd:
                continue
            lo, hi = bd
            fair = pr.prob_between(lo, hi, st)
            bid, ask = m.get("bestBid"), m.get("bestAsk")
            if bid is None or ask is None or ask <= bid:
                continue
            half = (ask - bid) / 2.0
            if not (0.03 <= 0.5 * (bid + ask) <= 0.97) or half < 0.012:
                continue
            # quote inside the touch, but never through fair
            our_bid = round(min(bid + 0.01, fair - 0.012), 3)
            our_ask = round(max(ask - 0.01, fair + 0.012), 3)
            our_bid = max(our_bid, 0.01)
            our_ask = min(our_ask, 0.99)
            if our_bid >= our_ask or our_bid >= fair or our_ask <= fair:
                continue
            rows.append({
                "slug": m.get("slug"), "bucket": m.get("groupItemTitle"),
                "mkt_bid": bid, "mkt_ask": ask, "fair": fair,
                "our_bid": our_bid, "our_ask": our_ask,
                "edge_bid": fair - our_bid, "edge_ask": our_ask - fair,
                "tilt": tilt, "settle": settle_dt.strftime("%a %d %b %H:%MZ"),
            })
        if verbose and rows:
            print(f"\n  {slug}   settles {rows[0]['settle']}   "
                  f"variance tilt vs flat extrapolation: x{tilt:.2f}")
            print(f"  {'bucket':16s} {'mkt':>13s} {'fair':>7s} "
                  f"{'our bid':>8s} {'our ask':>8s} {'edge b':>7s} {'edge a':>7s}")
            for r in rows:
                print(f"  {str(r['bucket']):16s} "
                      f"{r['mkt_bid']:6.3f}/{r['mkt_ask']:6.3f} {r['fair']:7.3f} "
                      f"{r['our_bid']:8.3f} {r['our_ask']:8.3f} "
                      f"{r['edge_bid']:+7.3f} {r['edge_ask']:+7.3f}")
        return rows


def main():
    mm = PolyMM()
    mm.refresh()
    print("Polymarket range-ladder quote plan "
          f"(paper; live needs py-clob-client + funded Polygon wallet)")
    total = 0.0
    for asset in ("ETH", "BTC"):
        print(f"\n{'='*76}\n=== {asset}")
        for day in (5, 6, 7, 8, 9):
            rows = mm.plan(asset, day)
            for r in rows:
                total += max(r["edge_bid"], 0) + max(r["edge_ask"], 0)
    print(f"\nsum of two-sided edge across all quotable buckets: "
          f"${total:,.2f} per contract-pair")
    print("Realistic capture is a fraction of that -- you are only filled on")
    print("one side at a time, and only when someone crosses to you.")


if __name__ == "__main__":
    main()
