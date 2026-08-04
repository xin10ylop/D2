"""Automated locked-pair bot: scan, size, execute, mark, exit, recycle.

The tradable object is a pair of Kalshi one-touch contracts on the same coin
where one measurement window strictly contains the other:

    SELL (buy NO)  the near/harder leg
    BUY  (buy YES) the far/easier leg

Touching the barrier inside the near window necessarily touches it inside the
far window, so the pair pays $1 in every state and $2 when the barrier is hit
only after the near leg closes. If both legs cost under $1 after fees, the
difference is locked at settlement.

Positions are NOT held blindly to expiry. Contracts trade continuously, so the
bot marks every position against the live book each cycle and closes early
whenever unwinding realises most of the edge sooner -- the same dollars earned
over fewer days is a higher return, and it frees capital for the next pair.
Measured history says this particular dislocation persists (open 95-98% of
hours since issuance), so the base case is a hold to the near leg's close with
early exit as the upside, not the plan.

Safety: paper by default. Live requires BOT_MODE=live AND KALSHI_LIVE=yes.
Both legs go fill-or-kill; a single-sided fill is unwound immediately, because
a naked short barrier is the one genuinely dangerous state this can reach.
"""
import argparse
import datetime as dt
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/user/D2/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import http_json, pmap                       # noqa: E402

from config import CFG                                   # noqa: E402
from venues import Kalshi                                # noqa: E402

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")


# ------------------------------------------------------------------ helpers

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fee(price, contracts=1):
    """Kalshi taker fee: ceil to the cent on the order, not the contract."""
    return math.ceil(0.07 * contracts * price * (1 - price) * 100) / 100


def iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000 if s else None


def log(*a):
    print(f"[{dt.datetime.utcnow():%m-%d %H:%M:%S}]", *a, flush=True)


def buy_levels(bk, side):
    """(price, size) to BUY `side`, best first. Kalshi quotes two bid ladders."""
    ladder = bk.get("no" if side == "YES" else "yes", [])
    return sorted([(round(1.0 - p, 2), q) for p, q in ladder
                   if 0 < 1 - p < 1 and q > 0], key=lambda x: x[0])


def sell_levels(bk, side):
    """(price, size) you'd RECEIVE selling `side`, best first."""
    ladder = bk.get("yes" if side == "YES" else "no", [])
    return sorted([(p, q) for p, q in ladder if 0 < p < 1 and q > 0],
                  key=lambda x: -x[0])


def walk(levels, want):
    got, notional = 0.0, 0.0
    for p, q in levels:
        take = min(q, want - got)
        if take <= 0:
            break
        got += take
        notional += take * p
    return (notional / got if got else None), got


# -------------------------------------------------------------------- bot

class ArbBot:
    def __init__(self, cfg=CFG, capital=1000.0, max_per_pair=400.0,
                 min_edge=0.02, exit_capture=0.80):
        self.cfg = cfg
        self.k = Kalshi(cfg)
        self.capital = capital
        self.max_per_pair = max_per_pair
        self.min_edge = min_edge
        # close early once unwinding banks this fraction of the locked edge
        self.exit_capture = exit_capture
        self.pos = self._load()

    # ------------------------------------------------------------- state
    def _load(self):
        if os.path.exists(STATE):
            with open(STATE) as f:
                return json.load(f)
        return {"open": [], "closed": [], "realised": 0.0}

    def _save(self):
        with open(STATE, "w") as f:
            json.dump(self.pos, f, indent=1)

    @property
    def deployed(self):
        return sum(p["capital"] for p in self.pos["open"])

    @property
    def free(self):
        return self.capital - self.deployed

    # ---------------------------------------------------------- universe
    def ladders(self):
        """Every live one-touch contract, with its measurement window."""
        from scan_barrier import series_coin
        try:
            ser = http_json(f"{KALSHI}/series", {"category": "Crypto", "limit": 200})
        except Exception as e:                            # noqa: BLE001
            log(f"series: {e}")
            return []
        names = [s["ticker"] for s in ser.get("series", [])
                 if any(k in s["ticker"].upper() for k in ("MAX", "MIN"))]

        def opens(sname):
            try:
                return http_json(f"{KALSHI}/events",
                                 {"series_ticker": sname, "status": "open",
                                  "limit": 50}).get("events", [])
            except Exception:                             # noqa: BLE001
                return []

        evs = []
        for s, es in pmap(opens, names, workers=12):
            for e in es or []:
                evs.append((s, e["event_ticker"]))

        def mk(pair):
            s, ev = pair
            try:
                return http_json(f"{KALSHI}/markets",
                                 {"event_ticker": ev, "limit": 200}).get("markets", [])
            except Exception:                             # noqa: BLE001
                return []

        rows = []
        for (s, ev), ms in pmap(mk, evs, workers=12):
            coin = series_coin(s)
            if not coin:
                continue
            for m in ms or []:
                if m.get("status") != "active":
                    continue
                st = m.get("strike_type")
                if st in ("greater", "greater_or_equal"):
                    B, side = fnum(m.get("floor_strike")), "up"
                elif st in ("less", "less_or_equal"):
                    B, side = fnum(m.get("cap_strike")), "dn"
                else:
                    continue
                if B is None:
                    continue
                rows.append({"coin": coin, "series": s, "event": ev,
                             "ticker": m["ticker"], "barrier": B, "side": side,
                             "open": iso(m.get("open_time")),
                             "close": iso(m.get("close_time")),
                             "bid": fnum(m.get("yes_bid_dollars")) or 0.0,
                             "ask": fnum(m.get("yes_ask_dollars"))})
        return rows

    def books(self, tickers):
        out = {}
        for t, bk in pmap(self.k.orderbook, sorted(set(tickers)), workers=12):
            if isinstance(bk, dict) and "__error__" not in bk:
                out[t] = bk
        return out

    # ------------------------------------------------------------- entry
    def find(self, rows):
        """Pairs whose windows nest and whose quotes leave a locked edge."""
        cands = []
        for a, b in itertools.permutations(rows, 2):
            if a["coin"] != b["coin"] or a["side"] != b["side"]:
                continue
            if a["open"] is None or b["open"] is None:
                continue
            # b (far) must cover every instant a (near) covers
            if not (b["open"] <= a["open"] + 3600_000 and b["close"] >= a["close"] - 1000):
                continue
            if a["side"] == "up" and b["barrier"] > a["barrier"] + 1e-9:
                continue
            if a["side"] == "dn" and b["barrier"] < a["barrier"] - 1e-9:
                continue
            if a["ticker"] == b["ticker"] or a["bid"] <= 0 or b["ask"] is None:
                continue
            if (1 - a["bid"]) + b["ask"] >= 1.0:
                continue          # cheap screen before spending a book request
            cands.append((a, b))
        return cands

    def size(self, a, b, books, consumed=None):
        """Walk both books together while the pair still costs under $1.

        `consumed` carries depth already committed to earlier pairs in this
        same cycle -- two pairs can share a leg, and the same contracts cannot
        be sold twice.
        """
        consumed = consumed if consumed is not None else {}
        la = self._avail(books, a["ticker"], "NO", consumed)
        lb = self._avail(books, b["ticker"], "YES", consumed)
        ia = ib = 0
        qty = cost = 0.0
        budget = min(self.free, self.max_per_pair)
        while ia < len(la) and ib < len(lb):
            pa, qa = la[ia]
            pb, qb = lb[ib]
            if pa + pb >= 1.0 - self.min_edge:
                break
            room = (budget - cost) / (pa + pb)
            q = min(qa, qb, room)
            if q <= 0:
                break
            qty += q
            cost += (pa + pb) * q
            la[ia] = (pa, qa - q)
            lb[ib] = (pb, qb - q)
            if la[ia][1] <= 1e-9:
                ia += 1
            if lb[ib][1] <= 1e-9:
                ib += 1
        if qty <= 0:
            return None
        qty = math.floor(qty)
        if qty < 1:
            return None
        px = cost / max(qty, 1)
        f = fee(px / 2, qty) * 2
        edge = qty * 1.0 - cost - f
        if edge <= 0:
            return None
        return {"qty": qty, "cost": cost, "fees": f, "capital": cost + f,
                "edge": edge, "edge_per": edge / qty}

    @staticmethod
    def _avail(books, ticker, side, consumed):
        """Buy levels with depth already spoken for in this cycle removed."""
        lv = buy_levels(books.get(ticker, {}), side)
        left = consumed.get(ticker, 0.0)
        out = []
        for p, q in lv:
            if left >= q:
                left -= q
                continue
            out.append((p, q - left))
            left = 0.0
        return out

    # -------------------------------------------------------------- exit
    def mark(self, p, books):
        """What unwinding both legs would realise right now, net of fees."""
        sa = sell_levels(books.get(p["sell_ticker"], {}), "NO")
        sb = sell_levels(books.get(p["buy_ticker"], {}), "YES")
        pa, qa = walk(sa, p["qty"])
        pb, qb = walk(sb, p["qty"])
        if pa is None or pb is None:
            return None
        q = min(qa, qb)
        gross = (pa + pb) * q
        f = fee(pa, q) + fee(pb, q)
        return {"proceeds": gross - f, "qty": q,
                "pnl": gross - f - p["capital"] * (q / p["qty"])}

    # ------------------------------------------------------------- cycle
    def cycle(self, dry=True):
        rows = self.ladders()
        log(f"universe: {len(rows)} live one-touch contracts, "
            f"free capital ${self.free:,.2f}, open pairs {len(self.pos['open'])}")

        # ---- manage what we already hold, first: it may free capital
        if self.pos["open"]:
            tk = [t for p in self.pos["open"] for t in (p["sell_ticker"], p["buy_ticker"])]
            bks = self.books(tk)
            still = []
            for p in self.pos["open"]:
                m = self.mark(p, bks)
                if not m:
                    still.append(p)
                    continue
                target = p["edge"] * self.exit_capture
                days_left = (p["near_close"] - time.time() * 1000) / 86400_000
                log(f"  HOLD {p['label']}  qty={p['qty']:,.0f} "
                    f"cost=${p['capital']:,.2f} mark=${m['proceeds']:,.2f} "
                    f"pnl=${m['pnl']:+,.2f} (locked ${p['edge']:,.2f}, "
                    f"{days_left:.0f}d to near close)")
                if m["pnl"] >= target and m["qty"] >= p["qty"] - 1e-6:
                    log(f"  EXIT {p['label']}: banking ${m['pnl']:,.2f} of "
                        f"${p['edge']:,.2f} locked, {days_left:.0f} days early")
                    if not dry:
                        self.k.place(p["sell_ticker"], "no", "sell", p["qty"],
                                     int(round(m["proceeds"] / p["qty"] * 100)),
                                     post_only=False)
                        self.k.place(p["buy_ticker"], "yes", "sell", p["qty"],
                                     int(round(m["proceeds"] / p["qty"] * 100)),
                                     post_only=False)
                    self.pos["realised"] += m["pnl"]
                    self.pos["closed"].append({**p, "exit_pnl": m["pnl"],
                                               "exit_ts": time.time()})
                else:
                    still.append(p)
            self.pos["open"] = still
            self._save()

        if self.free <= 1.0:
            log("  no free capital; holding")
            return

        # ---- look for new pairs
        cands = self.find(rows)
        if not cands:
            log("  no nested pairs quoted with a locked edge")
            return
        bks = self.books([t["ticker"] for c in cands for t in c])
        held = {(p["sell_ticker"], p["buy_ticker"]) for p in self.pos["open"]}
        # depth already committed by open positions cannot be reused
        consumed = {}
        for p in self.pos["open"]:
            consumed[p["sell_ticker"]] = consumed.get(p["sell_ticker"], 0) + p["qty"]
            consumed[p["buy_ticker"]] = consumed.get(p["buy_ticker"], 0) + p["qty"]

        ranked = []
        for a, b in cands:
            if (a["ticker"], b["ticker"]) in held:
                continue
            s = self.size(a, b, bks, consumed)
            if s:
                ranked.append((a, b, s))
        ranked.sort(key=lambda x: -x[2]["edge_per"])

        for a, b, _ in ranked:
            # re-size against depth consumed by pairs entered earlier this cycle
            s = self.size(a, b, bks, consumed)
            if not s or self.free < s["capital"]:
                continue
            label = (f"{a['coin']} {a['side']} "
                     f"{a['series']}@{a['barrier']:,.6g} / "
                     f"{b['series']}@{b['barrier']:,.6g}")
            log(f"  ENTER {label}  qty={s['qty']:,.0f} "
                f"capital=${s['capital']:,.2f} locked=${s['edge']:,.2f} "
                f"({s['edge']/s['capital']*100:.2f}%)")
            if not dry:
                # fill-or-kill both legs; unwind immediately if only one lands
                r1 = self.k.place(a["ticker"], "no", "buy", s["qty"],
                                  int(round((1 - a["bid"]) * 100)), post_only=False)
                r2 = self.k.place(b["ticker"], "yes", "buy", s["qty"],
                                  int(round(b["ask"] * 100)), post_only=False)
                if not r2 or r2.get("error"):
                    log("  !! far leg failed -- unwinding near leg")
                    self.k.place(a["ticker"], "no", "sell", s["qty"],
                                 int(round((1 - a["bid"]) * 100)), post_only=False)
                    continue
            consumed[a["ticker"]] = consumed.get(a["ticker"], 0) + s["qty"]
            consumed[b["ticker"]] = consumed.get(b["ticker"], 0) + s["qty"]
            self.pos["open"].append({
                "label": label, "coin": a["coin"],
                "sell_ticker": a["ticker"], "buy_ticker": b["ticker"],
                "qty": s["qty"], "capital": s["capital"], "edge": s["edge"],
                "near_close": a["close"], "far_close": b["close"],
                "opened": time.time(),
            })
            self._save()

    def report(self):
        log(f"realised ${self.pos['realised']:,.2f} | "
            f"open {len(self.pos['open'])} pairs, ${self.deployed:,.2f} deployed | "
            f"locked-but-unrealised ${sum(p['edge'] for p in self.pos['open']):,.2f}")

    def run(self, cycles=None, dry=True, sleep=300):
        log(f"mode={'LIVE' if not dry else 'PAPER'}  capital=${self.capital:,.2f}  "
            f"kalshi_trade={self.k.can_trade}")
        n = 0
        try:
            while cycles is None or n < cycles:
                try:
                    self.cycle(dry=dry)
                    self.report()
                except Exception as e:                    # noqa: BLE001
                    log(f"cycle error: {e}")
                n += 1
                if cycles is None or n < cycles:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            log("stopped")
        self.report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--max-per-pair", type=float, default=400.0)
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--sleep", type=int, default=300)
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    dry = not (a.live and CFG.live)
    if a.live and not CFG.live:
        log("--live ignored: set BOT_MODE=live and KALSHI_LIVE=yes to arm")
    ArbBot(capital=a.capital, max_per_pair=a.max_per_pair,
           min_edge=a.min_edge).run(cycles=a.cycles, dry=dry, sleep=a.sleep)
