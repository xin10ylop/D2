"""Main loop. Paper by default; live needs BOT_MODE=live AND KALSHI_LIVE=yes.

Each cycle:
  refresh Deribit -> rebuild the pricer -> pull open Kalshi events and books ->
  price every strike on the business clock -> emit take/make orders -> hedge
  net delta on the perpetual -> enforce limits.

Runs against live market data in paper mode, so the model and the signal can be
validated for as long as you like before any real order is sent.
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/user/D2/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load                                   # noqa: E402
from seasonality import Clock                             # noqa: E402

from config import CFG                                    # noqa: E402
from pricing import Pricer                                # noqa: E402
from risk import Book, hedge_required                     # noqa: E402
from strategy import evaluate, kalshi_taker_fee           # noqa: E402
from venues import Deribit, Kalshi                        # noqa: E402

SURFACE_REFRESH = 45.0     # seconds between Deribit surface rebuilds


def iso_ms(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def log(*a):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}]", *a, flush=True)


class Bot:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        seas = load("seasonality")
        self.clock = Clock(seas["clock_hour"], seas["clock_dow"])
        self.kalshi = Kalshi(cfg)
        self.deribit = Deribit(cfg)
        self.book = Book()
        self.pricers = {}
        self.surface_at = 0.0
        self.cycles = 0

    # ------------------------------------------------------------ surface
    def refresh_surface(self):
        now = time.time()
        if now - self.surface_at < SURFACE_REFRESH and self.pricers:
            return
        for a in self.cfg.assets:
            try:
                blob = self.deribit.snapshot(a)
                p = Pricer(blob, int(now * 1000), self.clock, a)
                if p.ok:
                    self.pricers[a] = p
            except Exception as e:                        # noqa: BLE001
                log(f"surface {a}: {e}")
        self.surface_at = now
        if self.pricers:
            log("surface refreshed:", {a: f"{p.index:,.0f}" for a, p in self.pricers.items()})

    # -------------------------------------------------------------- cycle
    def cycle(self, dry=True):
        self.refresh_surface()
        if not self.pricers:
            log("no pricer; skipping")
            return []

        breached = self.book.check(self.cfg)
        if breached:
            log("LIMIT BREACH -> quoting halted:", "; ".join(breached))
            return []

        now_ms = int(time.time() * 1000)
        found = []
        deltas = {}
        for series in self.cfg.series:
            asset = "ETH" if "ETH" in series else "BTC"
            pricer = self.pricers.get(asset)
            if not pricer:
                continue
            try:
                events = self.kalshi.open_events(series)
            except Exception as e:                        # noqa: BLE001
                log(f"events {series}: {e}")
                continue
            for ev in events:
                res = self.handle_event(pricer, asset, ev, now_ms, deltas, dry)
                found += res
        # hedge
        nd = self.book.net_delta_usd(deltas)
        need = hedge_required(nd, self.cfg)
        for asset, usd in need.items():
            if self.cfg.hedge_enabled:
                r = self.deribit.hedge(asset, usd)
                log(f"HEDGE {asset} ${usd:,.0f} -> {r}")
                self.book.hedge_usd[asset] += usd
        self.cycles += 1
        return found

    def handle_event(self, pricer, asset, ev, now_ms, deltas, dry):
        tkr = ev["event_ticker"]
        try:
            mkts = self.kalshi.markets(tkr)
        except Exception as e:                            # noqa: BLE001
            log(f"markets {tkr}: {e}")
            return []
        live = [m for m in mkts
                if m.get("status") == "active"
                and m.get("yes_ask_dollars") is not None]
        if not live:
            return []
        settle = iso_ms(live[0]["close_time"])
        secs = (settle - now_ms) / 1000.0
        if not (self.cfg.min_seconds_to_settle < secs < self.cfg.max_seconds_to_settle):
            return []
        st = pricer.state(settle)
        if not st:
            return []
        # Stand down where the smile would have to be extrapolated below
        # Deribit's front expiry -- the model is not validated there.
        if st.get("front_tau") and st["tau_h"] < self.cfg.min_frac_of_front_anchor * st["front_tau"]:
            return []

        # only fetch books for strikes near the money -- the wings are pinned
        band = 3.2 * st["sigma"] * math.sqrt(st["T"]) * st["F"]
        near = [m for m in live
                if m.get("floor_strike") is not None
                and abs(float(m["floor_strike"]) - st["F"]) < band]
        if not near:
            return []
        books = self.kalshi.books([m["ticker"] for m in near])

        out = []
        for m in near:
            bk = books.get(m["ticker"])
            if not bk:
                continue
            mm = {"ticker": m["ticker"], "subtitle": m.get("subtitle"),
                  "strike_type": m.get("strike_type"),
                  "floor_strike": float(m["floor_strike"]) if m.get("floor_strike") else None,
                  "cap_strike": float(m["cap_strike"]) if m.get("cap_strike") else None}
            r = evaluate(pricer, st, mm, bk, self.book.pos[m["ticker"]])
            if not r:
                continue
            if mm["floor_strike"]:
                deltas[m["ticker"]] = (asset,
                                       pricer.delta_per_contract(mm["floor_strike"], st))
            r.update({"event": tkr, "asset": asset, "secs": secs,
                      "tau_h": st["tau_h"], "cal_h": st["cal_h"],
                      "sigma": st["sigma"], "F": st["F"]})
            if r["actions"]:
                out.append(r)
                self.act(r, asset, dry)
        return out

    def act(self, r, asset, dry):
        for a in r["actions"]:
            if a["kind"] == "take":
                fee = kalshi_taker_fee(a["price"]) * a["qty"]
                log(f"TAKE {r['event']} {r['desc']} {a['side']} "
                    f"{a['qty']}@{a['price']:.2f} fair={r['fair']:.3f} "
                    f"edge={a['edge']:+.3f} clock={r['clock_edge']:+.3f} "
                    f"exp=${a['exp_profit']:.2f}")
                if not dry:
                    self.kalshi.place(r["ticker"],
                                      "yes" if a["side"] == "YES" else "no",
                                      "buy", a["qty"], int(round(a["price"] * 100)),
                                      post_only=False)
                self.book.fill(r["ticker"], a["side"], a["qty"], a["price"], fee,
                               asset, {"kind": "take", "edge": a["edge"]})
            else:
                log(f"QUOTE {r['event']} {r['desc']} {a['side']} "
                    f"{a['qty']}@{a['price']:.2f} (fair={r['fair']:.3f})")
                if not dry:
                    self.kalshi.place(r["ticker"],
                                      "yes" if a["side"] == "YES" else "no",
                                      "buy", a["qty"], int(round(a["price"] * 100)),
                                      post_only=True)

    # --------------------------------------------------------------- loop
    def run(self, cycles=None, dry=True):
        log(f"mode={'LIVE' if self.cfg.live and not dry else 'PAPER'} "
            f"kalshi_trade={self.kalshi.can_trade} deribit_hedge={self.deribit.can_trade}")
        n = 0
        try:
            while cycles is None or n < cycles:
                t0 = time.time()
                try:
                    found = self.cycle(dry=dry)
                    if found:
                        tot = sum(a["exp_profit"] for r in found
                                  for a in r["actions"] if a["kind"] == "take")
                        if tot:
                            log(f"cycle {n}: {len(found)} actionable, "
                                f"expected ${tot:,.2f}, gross={self.book.gross()}")
                except Exception as e:                    # noqa: BLE001
                    log(f"cycle error: {e}")
                n += 1
                time.sleep(max(0.0, self.cfg.poll_secs - (time.time() - t0)))
        except KeyboardInterrupt:
            log("stopped")
        self.report()

    def report(self):
        log(f"cycles={self.cycles} fills={len(self.book.fills)} "
            f"gross={self.book.gross()} fees=${self.book.fees:,.2f} "
            f"cash=${self.book.cash:,.2f}")
        exp = sum(f.get("edge", 0) * f["qty"] for f in self.book.fills)
        log(f"expected edge captured (model): ${exp:,.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--live", action="store_true",
                    help="send real orders (also needs BOT_MODE=live, KALSHI_LIVE=yes)")
    args = ap.parse_args()
    dry = not (args.live and CFG.live)
    if args.live and not CFG.live:
        log("--live ignored: set BOT_MODE=live and KALSHI_LIVE=yes to arm")
    Bot().run(cycles=args.cycles, dry=dry)
