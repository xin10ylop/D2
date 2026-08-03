"""Signal -> orders. Two books of business, one model.

TAKE  -- lift or hit a resting quote that the model says is wrong by more than
         `take_edge` after fees. This is where the clock edge cashes: a quoter
         scaling volatility by calendar hours misprices a half-sigma digital by
         2-6c depending on which hours the contract spans, and that error is a
         deterministic function of the settlement time.

MAKE  -- rest two-sided quotes around fair. Pays the spread, but only survives
         if the fair value is right, because every fill is by definition
         someone taking the other side.

Both are gated by the same sanity rule: if the market disagrees with the model
by more than `max_model_disagreement`, the market probably knows something the
model does not, so stand down on that strike rather than betting against it.
"""
import math

from config import CFG


def kalshi_taker_fee(price, mult=None):
    m = CFG.kalshi_taker_mult if mult is None else mult
    return m * price * (1.0 - price)


def yes_levels(book):
    """Executable levels for BUYING yes: the no-ladder, inverted."""
    return sorted([(1.0 - p, s) for p, s in book.get("no", []) if 0 < 1 - p < 1],
                  key=lambda x: x[0])


def no_levels(book):
    """Executable levels for BUYING no: the yes-ladder, inverted."""
    return sorted([(1.0 - p, s) for p, s in book.get("yes", []) if 0 < 1 - p < 1],
                  key=lambda x: x[0])


def best(book):
    y, n = book.get("yes", []), book.get("no", [])
    yes_bid = max((p for p, _ in y), default=None)
    yes_ask = (1.0 - max((p for p, _ in n), default=None)) if n else None
    return yes_bid, yes_ask


def contract_bounds(m):
    st, fl, cap = m["strike_type"], m.get("floor_strike"), m.get("cap_strike")
    if st in ("greater", "greater_or_equal"):
        return fl, float("inf")
    if st in ("less", "less_or_equal"):
        return 0.0, cap
    if st == "between":
        return fl, cap
    return None


def evaluate(pricer, state, market, book, inventory=0):
    """Return the model's view of one contract plus any actionable order."""
    b = contract_bounds(market)
    if not b:
        return None
    lo, hi = b
    fair = pricer.prob_between(lo, hi, state)
    if not (0.0 <= fair <= 1.0):
        return None
    flat = (pricer.flat_clock_prob(lo, state) if math.isinf(hi)
            else max(pricer.flat_clock_prob(lo, state)
                     - pricer.flat_clock_prob(hi, state), 0.0))

    yes_bid, yes_ask = best(book)
    mid = None
    if yes_bid is not None and yes_ask is not None:
        mid = 0.5 * (yes_bid + yes_ask)
        if yes_ask - yes_bid > CFG.max_spread_to_quote:
            mid = None

    out = {"ticker": market["ticker"], "desc": market.get("subtitle"),
           "lo": lo, "hi": hi, "fair": fair, "flat_clock": flat,
           "clock_edge": fair - flat, "yes_bid": yes_bid, "yes_ask": yes_ask,
           "mid": mid, "actions": []}

    # --- sanity gate: never fight a market that disagrees violently
    if mid is not None and abs(mid - fair) > CFG.max_model_disagreement:
        out["stood_down"] = "model disagrees with market beyond gate"
        return out

    # --- TAKE
    for side, levels, tgt in (("YES", yes_levels(book), fair),
                              ("NO", no_levels(book), 1.0 - fair)):
        for px, sz in levels[:4]:
            cost = px + kalshi_taker_fee(px)
            edge = tgt - cost
            if edge < CFG.take_edge:
                break
            room = CFG.max_contracts_per_strike - abs(inventory)
            qty = int(min(sz, max(room, 0)))
            if qty <= 0:
                break
            out["actions"].append({
                "kind": "take", "side": side, "price": px, "qty": qty,
                "edge": edge, "exp_profit": edge * qty})

    # --- MAKE (only where the model is confident and the book is sane)
    if mid is not None and not out["actions"]:
        m = CFG.quote_margin + CFG.kalshi_maker_fee
        skew = 0.0
        if CFG.max_contracts_per_strike:
            skew = 0.5 * m * (inventory / CFG.max_contracts_per_strike)
        bid = round(max(fair - m - skew, 0.01), 2)
        ask = round(min(fair + m - skew, 0.99), 2)
        if bid < ask:
            # only post if it improves the book; never cross it
            if yes_bid is None or bid > yes_bid:
                out["actions"].append({"kind": "make", "side": "YES",
                                       "price": bid, "qty": 50})
            if yes_ask is None or ask < yes_ask:
                out["actions"].append({"kind": "make", "side": "NO",
                                       "price": round(1.0 - ask, 2), "qty": 50})
    return out
