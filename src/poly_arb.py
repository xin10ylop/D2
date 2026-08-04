"""Full-universe arbitrage scan on Polymarket, run the way the Kalshi scan was.

Two engines, because Polymarket's crypto markets split into two kinds of claim
and they need different machinery:

  1. TERMINAL PARTITION.  Every `above`, `range` and daily `up-or-down` market
     on one asset that settles on the SAME Binance 1-minute candle is a claim
     about one number. Those form a partition of the real line, so the whole
     set goes into the linear program from engine.py, which finds monotonicity
     breaks, bucket-sum breaks and the digital-vs-bucket relation at once,
     rather than pattern by pattern.

     The up/down market is the piece the old scan could not see. "Bitcoin Up or
     Down on August 4" resolves on the same candle as the August 4 strike
     ladder, against a reference close that is already public -- so it is the
     ladder's digital struck at that number, and it belongs in the same LP.

  2. PATH IMPLICATION.  One-touch ladders span windows, so they do not share a
     partition with anything; they are handled pairwise. A => B means
     P(A) <= P(B), so if B's ask is below A's bid the pair is locked. This
     covers weekly-inside-monthly, monthly-inside-before-2027, and terminal =>
     touch, which is where all of the Kalshi money came from.

Polymarket charges no maker or taker fee, so the whole gap is kept. Sizing is
against real CLOB depth, level by level -- never top of book, never a blended
average.
"""
import itertools

from common import http_json, load, pmap, save
from engine import Leg, solve_arb, state_grid, yes_payoff

CLOB = "https://clob.polymarket.com"


def book(token):
    try:
        b = http_json(f"{CLOB}/book", {"token_id": token})
        return {"bids": sorted([(float(x["price"]), float(x["size"]))
                                for x in b.get("bids", [])], key=lambda z: -z[0]),
                "asks": sorted([(float(x["price"]), float(x["size"]))
                                for x in b.get("asks", [])], key=lambda z: z[0])}
    except Exception:                                    # noqa: BLE001
        return None


def fetch_books(tokens):
    out = {}
    for t, b in pmap(book, sorted({t for t in tokens if t}), workers=12):
        if b and (b["bids"] or b["asks"]):
            out[t] = b
    return out


def levels(ladder, invert):
    """Taker levels as (cost, size), best first. One entry per book level."""
    out = []
    src = sorted(ladder, key=lambda x: -x[0]) if invert else sorted(ladder, key=lambda x: x[0])
    for p, s in src:
        px = (1.0 - p) if invert else p
        if 0.0 < px < 1.0 and s > 0:
            out.append((px, s))
    return out


def claim_legs(c, bk):
    """Both prepaid sides of one claim, one Leg per book level, no fee."""
    if c["kind"] == "above":
        yp = yes_payoff("greater", c["K"], None)
    elif c["kind"] == "range":
        yp = yes_payoff("half_open", c["lo"], c["hi"])
    else:
        return []
    npf = lambda S, f=yp: 1.0 - f(S)                     # noqa: E731
    out = []
    for side, lv, pay in (("YES", levels(bk.get("asks", []), False), yp),
                          ("NO", levels(bk.get("bids", []), True), npf)):
        for i, (px, sz) in enumerate(lv):
            out.append(Leg(key=f"{c['market']}|{side}|L{i}", venue="polymarket",
                           label=f"{c['title']} [{c['series']}] {side} @{px:.3f} x{sz:,.0f}",
                           price=px, fee=0.0, depth=sz, payoff=pay,
                           meta={"claim": c, "side": side}))
    return out


# --------------------------------------------------------- 1. terminal partition

def partitions(cl):
    """Group terminal claims by (asset, settlement instant)."""
    g = {}
    for c in cl:
        if c["kind"] not in ("above", "range"):
            continue
        g.setdefault((c["asset"], round(c["end"] / 60000)), []).append(c)
    return {k: v for k, v in g.items() if len(v) >= 2}


def scan_partitions(cl, books, capital=None):
    hits = []
    for (asset, tmin), cs in sorted(partitions(cl).items()):
        legs = []
        for c in cs:
            bk = books.get(c["yes"])
            if bk:
                legs += claim_legs(c, bk)
        if len(legs) < 2:
            continue
        bnds = []
        for c in cs:
            bnds += [c["K"]] if c["kind"] == "above" else [c["lo"], c["hi"]]
        bnds = [b for b in bnds if b and b != float("inf")]
        st = state_grid(bnds)
        if not st:
            continue
        r = solve_arb(legs, st, capital=capital, min_edge=1e-6)
        if r and r.get("profit", 0) > 1e-6 and r["legs"]:
            has_ud = any(lg.meta["claim"].get("from_updown") for lg, _ in r["legs"])
            hits.append({"asset": asset, "tmin": tmin, "res": r,
                         "n_claims": len(cs), "uses_updown": has_ud})
    return hits


# --------------------------------------------------------- 2. path implications

def implications(cl):
    """(A, B) with A => B, per asset. Path facts only -- no model, no vol."""
    out = []
    for asset in sorted({c["asset"] for c in cl}):
        cs = [c for c in cl if c["asset"] == asset]
        tu = [c for c in cs if c["kind"] == "touch_up"]
        td = [c for c in cs if c["kind"] == "touch_dn"]
        ab = [c for c in cs if c["kind"] == "above"]
        rg = [c for c in cs if c["kind"] == "range"]

        def inside(t, b):
            return b["start"] is not None and b["start"] - 3600_000 <= t <= b["end"] + 1000

        # terminal above K at instant t  =>  the high over any window containing
        # t reached K' for every K' <= K
        for a in ab:
            for b in tu:
                if b["K"] <= a["K"] + 1e-9 and inside(a["end"], b):
                    out.append((a, b, "above => touch_up"))
        # terminal inside [lo, hi) => touched up at lo and down at hi
        for a in rg:
            for b in tu:
                if a["lo"] > 0 and b["K"] <= a["lo"] + 1e-9 and inside(a["end"], b):
                    out.append((a, b, "range => touch_up"))
            for b in td:
                if a["hi"] < float("inf") and b["K"] >= a["hi"] - 1e-9 and inside(a["end"], b):
                    out.append((a, b, "range => touch_dn"))
        # touch nesting: shorter window, harder barrier => longer window, easier
        for a, b in itertools.permutations(tu, 2):
            if b["K"] <= a["K"] + 1e-9 and b["start"] <= a["start"] + 3600_000 \
                    and b["end"] >= a["end"] - 1000 and a["market"] != b["market"]:
                out.append((a, b, "touch_up nesting"))
        for a, b in itertools.permutations(td, 2):
            if b["K"] >= a["K"] - 1e-9 and b["start"] <= a["start"] + 3600_000 \
                    and b["end"] >= a["end"] - 1000 and a["market"] != b["market"]:
                out.append((a, b, "touch_dn nesting"))
    return out


def walk(la, lb, min_edge):
    """Consume both ladders while the pair still costs under a dollar."""
    ia = ib = 0
    qty = cost = 0.0
    la, lb = [list(x) for x in la], [list(x) for x in lb]
    while ia < len(la) and ib < len(lb):
        pa, qa = la[ia]
        pb, qb = lb[ib]
        if pa + pb >= 1.0 - min_edge:
            break
        q = min(qa, qb)
        qty += q
        cost += (pa + pb) * q
        la[ia][1] -= q
        lb[ib][1] -= q
        if la[ia][1] <= 1e-9:
            ia += 1
        if lb[ib][1] <= 1e-9:
            ib += 1
    return qty, cost


def scan_implications(cl, books, min_edge=0.005):
    """Every implication priced off the live CLOB, with no cached pre-screen.

    Gamma's bestBid/bestAsk lag the book, and screening on them can only ever
    discard pairs -- a stale quote that looks consistent hides a real crossing.
    There are only a few thousand pairs and the books are already in memory, so
    every pair is walked against real depth.
    """
    imps = implications(cl)
    out = []
    for a, b, why in imps:
        ba, bb = books.get(a["yes"]), books.get(b["yes"])
        if not ba or not bb:
            continue
        # sell A = hit A's YES bids; buy B = lift B's YES asks
        la = levels(ba.get("bids", []), True)
        lb = levels(bb.get("asks", []), False)
        if not la or not lb or la[0][0] + lb[0][0] >= 1.0 - 1e-3:
            continue
        qty, cost = walk(la, lb, 1e-3)
        if qty <= 0 or qty - cost <= min_edge:
            continue
        out.append({"A": a, "B": b, "why": why, "edge": qty / max(qty, 1) - cost / qty,
                    "qty": qty, "cost": cost, "profit": qty - cost})
    out.sort(key=lambda r: -r["profit"])
    return len(imps), out


# ------------------------------------------------------------------------ main

def main(capital=None, min_edge=0.005):
    cl = load("poly_universe")
    if not cl:
        import poly_universe
        cl = poly_universe.main()
    print(f"claims: {len(cl)}")

    books = fetch_books([c["yes"] for c in cl])
    print(f"order books fetched: {len(books)}\n")

    print("=" * 100)
    print("1. TERMINAL PARTITION LP  (above + range + daily up/down on one candle)")
    print("=" * 100)
    ph = scan_partitions(cl, books, capital=capital)
    parts = partitions(cl)
    print(f"partitions with 2+ claims: {len(parts)}   arbitrage found: {len(ph)}\n")
    tot_p = tot_c = 0.0
    for h in sorted(ph, key=lambda r: -r["res"]["profit"]):
        r = h["res"]
        print(f"  {h['asset']:4s}  {h['n_claims']:3d} claims  "
              f"profit ${r['profit']:8,.2f}  outlay ${r['outlay']:9,.2f}  "
              f"roi {(r['roi'] or 0)*100:6.2f}%"
              f"{'   [uses up/down]' if h['uses_updown'] else ''}")
        for lg, x in sorted(r["legs"], key=lambda z: -z[1])[:8]:
            print(f"        {x:9,.1f} x  {lg.label}")
        tot_p += r["profit"]
        tot_c += r["outlay"]
    if ph:
        print(f"\n  partition total: ${tot_p:,.2f} locked on ${tot_c:,.2f}")
    else:
        print("  none -- every terminal ladder is internally consistent")

    print("\n" + "=" * 100)
    print("2. PATH IMPLICATION  (touch nesting, terminal => touch)")
    print("=" * 100)
    n_imp, ih = scan_implications(cl, books, min_edge)
    print(f"implications tested: {n_imp:,}   violations at top of book: "
          f"{sum(1 for _ in ih)}\n")
    if ih:
        print(f"  {'why':20s} {'ast':4s} {'SELL A':34s} {'BUY B':34s} "
              f"{'edge':>6s} {'qty':>8s} {'profit':>9s}")
        for h in sorted(ih, key=lambda r: -r["profit"]):
            a, b = h["A"], h["B"]
            print(f"  {h['why']:20s} {a['asset']:4s} "
                  f"{(a['slug'][-20:] + ' ' + str(a['title']))[:34]:34s} "
                  f"{(b['slug'][-20:] + ' ' + str(b['title']))[:34]:34s} "
                  f"{h['edge']:+6.3f} {h['qty']:8,.0f} ${h['profit']:8,.2f}")
        print(f"\n  path total: ${sum(h['profit'] for h in ih):,.2f} locked on "
              f"${sum(h['cost'] for h in ih):,.2f}")
    else:
        print("  none")

    res = {"partition": [{"asset": h["asset"], "profit": h["res"]["profit"],
                          "outlay": h["res"]["outlay"], "uses_updown": h["uses_updown"],
                          "legs": [[lg.label, x] for lg, x in h["res"]["legs"]]}
                         for h in ph],
           "path": [{"why": h["why"], "asset": h["A"]["asset"],
                     "sell": h["A"]["market"], "buy": h["B"]["market"],
                     "qty": h["qty"], "cost": h["cost"], "profit": h["profit"]}
                    for h in ih]}
    save("poly_arb", res)
    grand = tot_p + sum(h["profit"] for h in ih)
    cap = tot_c + sum(h["cost"] for h in ih)
    print(f"\nGRAND TOTAL: ${grand:,.2f} locked on ${cap:,.2f} of capital"
          f"{f'  ({grand/cap*100:.1f}%)' if cap else ''}")
    return res


if __name__ == "__main__":
    main()
