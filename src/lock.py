"""Locked one-touch arbitrage at full book depth, cross-strike and cross-tenor.

The relation is pure logic. For up-barriers, touching B1 during window W1
implies touching B2 during W2 whenever

    B2 <= B1     and     W1 is contained in W2

so a portfolio of NO on (B1, W1) and YES on (B2, W2) pays at least $1 in every
state of the world:

    touched B1 in W1  -> B2 also touched in W2  -> YES pays          $1
    not touched B1    -> NO pays, YES may too                     $1 or $2

If the two legs cost less than $1 after fees, the difference is locked. No
volatility, no distribution, no view. Down-barriers mirror it with B2 >= B1.

Kalshi lists monthly and annual one-touch ladders on the same coin in separate
order books, and the annual ladder is the stale one, so the relation binds.
Depth is walked level by level rather than taken from the top of book.
"""
import datetime as dt
import itertools

from common import http_json, load, save

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fee(p):
    return 0.07 * p * (1 - p)


def iso_ms(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000 if s else None


def book(ticker):
    d = http_json(f"{KALSHI}/markets/{ticker}/orderbook", {"depth": 100})
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    return {"yes": [[float(p), float(q)] for p, q in (ob.get("yes_dollars") or [])],
            "no": [[float(p), float(q)] for p, q in (ob.get("no_dollars") or [])]}


def buy_levels(bk, side):
    """Executable (price, size) to BUY `side`, best first."""
    ladder = bk.get("no" if side == "YES" else "yes", [])
    return sorted([(1.0 - p, q) for p, q in ladder if 0 < 1 - p < 1 and q > 0],
                  key=lambda x: x[0])


def walk_pair(lv_a, lv_b, budget_edge=0.0):
    """Match two buy-ladders level by level while the pair costs under $1."""
    ia = ib = 0
    ra = list(lv_a)
    rb = list(lv_b)
    total_qty = 0.0
    total_edge = 0.0
    fills = []
    while ia < len(ra) and ib < len(rb):
        pa, qa = ra[ia]
        pb, qb = rb[ib]
        cost = pa + fee(pa) + pb + fee(pb)
        edge = 1.0 - cost
        if edge <= budget_edge:
            break
        q = min(qa, qb)
        fills.append((pa, pb, q, edge))
        total_qty += q
        total_edge += edge * q
        ra[ia] = (pa, qa - q)
        rb[ib] = (pb, qb - q)
        if ra[ia][1] <= 1e-9:
            ia += 1
        if rb[ib][1] <= 1e-9:
            ib += 1
    return total_qty, total_edge, fills


def main(min_edge=0.005):
    from scan_barrier import series_coin
    uni = load("universe_kalshi")
    touch = [r for r in uni if r["kind"] in ("onetouch_max", "onetouch_min")]

    ladders = []
    for ev in touch:
        coin = series_coin(ev["series"])
        if not coin:
            continue
        try:
            d = http_json(f"{KALSHI}/markets", {"event_ticker": ev["event"], "limit": 200})
        except Exception:                                # noqa: BLE001
            continue
        for m in d.get("markets", []):
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
            ladders.append({
                "coin": coin, "series": ev["series"], "event": ev["event"],
                "ticker": m["ticker"], "barrier": B, "side": side,
                "open": iso_ms(m.get("open_time")),
                "close": iso_ms(m.get("close_time")),
                "oi": fnum(m.get("open_interest_fp")) or 0,
            })
    print(f"one-touch contracts loaded: {len(ladders)}", flush=True)

    # candidate implication pairs, before spending any book requests
    pairs = []
    for a, b in itertools.permutations(ladders, 2):
        if a["coin"] != b["coin"] or a["side"] != b["side"]:
            continue
        if a["open"] is None or b["open"] is None:
            continue
        # a is the NEAR/HARDER leg (we sell it), b is the FAR/EASIER leg (we buy)
        if not (b["open"] <= a["open"] + 3600_000 and b["close"] >= a["close"] - 1000):
            continue
        if a["side"] == "up" and not (b["barrier"] <= a["barrier"] + 1e-9):
            continue
        if a["side"] == "dn" and not (b["barrier"] >= a["barrier"] - 1e-9):
            continue
        if a["ticker"] == b["ticker"]:
            continue
        pairs.append((a, b))
    print(f"logically implied pairs: {len(pairs)}", flush=True)

    need = {t for p in pairs for t in (p[0]["ticker"], p[1]["ticker"])}
    print(f"fetching {len(need)} order books...", flush=True)
    books = {}
    from common import pmap
    for t, bk in pmap(book, sorted(need), workers=12):
        if isinstance(bk, dict) and "__error__" not in bk:
            books[t] = bk

    found = []
    for a, b in pairs:
        ba, bb = books.get(a["ticker"]), books.get(b["ticker"])
        if not ba or not bb:
            continue
        qty, edge, fills = walk_pair(buy_levels(ba, "NO"), buy_levels(bb, "YES"),
                                     budget_edge=min_edge)
        if qty > 0 and edge > 0:
            found.append({
                "coin": a["coin"], "side": a["side"],
                "sell_series": a["series"], "sell_ticker": a["ticker"],
                "sell_barrier": a["barrier"],
                "buy_series": b["series"], "buy_ticker": b["ticker"],
                "buy_barrier": b["barrier"],
                "qty": qty, "profit": edge, "avg_edge": edge / qty,
                "capital": sum((pa + pb) * q for pa, pb, q, _ in fills),
                "fills": fills[:6],
            })

    found.sort(key=lambda r: -r["profit"])
    save("lock", found)

    print(f"\n=== LOCKED ARBITRAGE (no model, no volatility view): "
          f"{len(found)} pairs\n")
    if not found:
        print("   none")
        return found
    print(f"{'coin':5s} {'d':2s} {'SELL (near/harder)':28s} {'BUY (far/easier)':28s} "
          f"{'qty':>7s} {'edge':>7s} {'capital':>10s} {'profit':>9s} {'return':>7s}")
    tot_p = tot_c = 0.0
    for r in found:
        tot_p += r["profit"]
        tot_c += r["capital"]
        print(f"{r['coin']:5s} {r['side']:2s} "
              f"{r['sell_series'][:15]+' @'+format(r['sell_barrier'],',.5g'):28s} "
              f"{r['buy_series'][:15]+' @'+format(r['buy_barrier'],',.5g'):28s} "
              f"{r['qty']:7,.0f} {r['avg_edge']:+7.3f} ${r['capital']:9,.0f} "
              f"${r['profit']:8,.0f} {r['profit']/max(r['capital'],1)*100:6.2f}%")
    print(f"\n  TOTAL: ${tot_p:,.0f} locked on ${tot_c:,.0f} of capital "
          f"({tot_p/max(tot_c,1)*100:.2f}%)")
    print("  Legs overlap across pairs, so the aggregate is an upper bound;")
    print("  the per-pair rows are individually executable.")
    return found


if __name__ == "__main__":
    main()
