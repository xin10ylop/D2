"""Core pricing / execution / arbitrage machinery.

The central abstraction is a *Leg*: one directional, cash-collateralised claim
that costs `price + fee` today and pays `payoff(S)` at settlement, available in
size `depth`.  Every Kalshi and Polymarket contract decomposes into two legs
(buy YES, buy NO), which is convenient because on both venues *both* sides are
fully prepaid -- there is no margin, so "sell YES" is literally "buy NO".

Given a set of legs that all settle on the *same* print, a guaranteed-profit
portfolio is the solution of

    max_x  min_s  sum_j x_j * (payoff_j(s) - cost_j)      0 <= x_j <= depth_j

which is a linear program.  Solving it finds every static arbitrage in the
ladder simultaneously -- monotonicity breaks, bucket-sum breaks, and the
digital-vs-bucket cross-book relation -- without having to enumerate patterns
by hand.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------- fee models

def kalshi_fee(price, contracts=1.0, multiplier=0.07):
    """Kalshi taker fee: ceil_to_cent(m * C * P * (1-P)).

    Charged on entry only; settlement is free. Per-contract this converges to
    m*P*(1-P) for any order big enough that the cent-rounding washes out.
    """
    raw = multiplier * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def kalshi_fee_rate(price, multiplier=0.07):
    """Marginal (large-order) fee per contract, in dollars."""
    return multiplier * price * (1.0 - price)


def poly_fee_rate(price):
    """Polymarket CLOB charges no maker/taker fee; gas is relayed."""
    return 0.0


def deribit_option_fee(index_price, premium_coin, is_ccy_btc=True):
    """0.03% of the underlying, capped at 12.5% of the option premium."""
    return min(0.0003 * 1.0, 0.125 * premium_coin) if premium_coin > 0 else 0.0003


DERIBIT_FUT_TAKER = 0.0005  # 5 bp of notional


# ------------------------------------------------------------------- legs

@dataclass
class Leg:
    key: str
    venue: str
    label: str
    price: float            # clean price paid per contract, in dollars
    fee: float              # marginal fee per contract, in dollars
    depth: float            # contracts available at (or better than) `price`
    payoff: object          # callable S -> {0,1} payoff in dollars
    meta: dict = field(default_factory=dict)

    @property
    def cost(self):
        return self.price + self.fee


def yes_payoff(strike_type, floor, cap):
    """Indicator function for the YES side of a laddered contract."""
    if strike_type in ("greater", "greater_or_equal"):
        return lambda S, f=floor: 1.0 if S > f else 0.0
    if strike_type in ("less", "less_or_equal"):
        return lambda S, c=cap: 1.0 if S < c else 0.0
    if strike_type == "between":
        return lambda S, a=floor, b=cap: 1.0 if (a <= S <= b) else 0.0
    if strike_type == "half_open":  # Polymarket buckets are [lo, hi)
        return lambda S, a=floor, b=cap: 1.0 if (a <= S < b) else 0.0
    return None


def _levels(ladder, invert, max_size):
    """Normalise a counterparty bid ladder into (cost, size) taker levels.

    One Leg per price level, never a blended VWAP: the LP has to be free to
    take two cents of a cheap level and stop, rather than being forced to
    average in the junk sitting behind it.
    """
    out = []
    left = max_size if max_size else float("inf")
    # Best level first: for a bid ladder we are lifting, that is the top bid.
    for p, s in sorted(ladder, key=lambda x: -x[0]) if invert else \
            sorted(ladder, key=lambda x: x[0]):
        if left <= 0:
            break
        px = (1.0 - p) if invert else p
        if not (0.0 < px < 1.0) or s <= 0:
            continue
        take = min(s, left)
        out.append((px, take))
        left -= take
    return out


def kalshi_legs(market, book=None, fee_mult=0.07, max_size=None):
    """Every executable taker level on both prepaid sides of a Kalshi market.

    Kalshi publishes two bid ladders. A resting NO bid at q is an offer to sell
    YES at 1-q, so buying YES means lifting the NO ladder from the top down.
    """
    fl, cap, st = market["floor_strike"], market["cap_strike"], market["strike_type"]
    yp = yes_payoff(st, fl, cap)
    if yp is None:
        return []
    np_ = lambda S, f=yp: 1.0 - f(S)  # noqa: E731

    if book:
        sides = [("YES", _levels(book.get("no", []), True, max_size), yp),
                 ("NO", _levels(book.get("yes", []), True, max_size), np_)]
    else:
        sides = []
        if market["no_bid"]:
            sides.append(("YES", [(1.0 - market["no_bid"], market["yes_ask_size"] or 0)], yp))
        if market["yes_bid"]:
            sides.append(("NO", [(1.0 - market["yes_bid"], market["yes_bid_size"] or 0)], np_))

    out = []
    for side, levels, pay in sides:
        for i, (px, sz) in enumerate(levels):
            out.append(Leg(
                key=f"{market['ticker']}|{side}|L{i}",
                venue="kalshi",
                label=f"{market['ticker']} {side} @{px:.4f} x{sz:,.0f}",
                price=px,
                fee=kalshi_fee_rate(px, fee_mult),
                depth=sz,
                payoff=pay,
                meta={"ticker": market["ticker"], "side": side, "level": i,
                      "subtitle": market["subtitle"], "floor": fl, "cap": cap,
                      "strike_type": st, "oi": market["open_interest"]},
            ))
    return out


def poly_legs(market, book, kind, bounds, max_size=None):
    """Every executable taker level on both sides of one Polymarket outcome.

    The CLOB book is quoted on the YES token; buying NO means selling YES into
    the bid ladder at (1 - price), which is what the UI's "Buy No" does.
    """
    lo, hi = bounds
    yp = yes_payoff("greater" if kind == "above" else "half_open", lo, hi)
    np_ = lambda S, f=yp: 1.0 - f(S)  # noqa: E731

    sides = [("YES", _levels(book.get("asks", []), False, max_size), yp),
             ("NO", _levels(book.get("bids", []), True, max_size), np_)]

    out = []
    for side, levels, pay in sides:
        for i, (px, sz) in enumerate(levels):
            out.append(Leg(
                key=f"{market['slug']}|{side}|L{i}",
                venue="polymarket",
                label=f"{market['slug']} {side} @{px:.4f} x{sz:,.0f}",
                price=px,
                fee=0.0,
                depth=sz,
                payoff=pay,
                meta={"slug": market["slug"], "side": side, "level": i,
                      "group": market["group_title"], "lo": lo, "hi": hi, "kind": kind},
            ))
    return out


# --------------------------------------------------------------- state space

def state_grid(boundaries, pad=0.05):
    """Representative settlement prices, one strictly inside each partition cell.

    Sampling just above and just below every boundary is deliberately
    redundant: duplicate states only duplicate LP constraints, whereas a
    missed cell would let a fake arbitrage through.
    """
    b = sorted(set(float(x) for x in boundaries if x is not None))
    if not b:
        return []
    pts = [b[0] - max(pad, b[0] * 0.02)]
    for i, v in enumerate(b):
        pts += [v - 0.005, v, v + 0.005]
        if i + 1 < len(b):
            pts.append(0.5 * (v + b[i + 1]))
    pts.append(b[-1] + max(pad, b[-1] * 0.02))
    return sorted(set(round(p, 4) for p in pts if p > 0))


def payoff_matrix(legs, states):
    A = np.zeros((len(states), len(legs)))
    for j, lg in enumerate(legs):
        for i, s in enumerate(states):
            A[i, j] = lg.payoff(s)
    return A


# ------------------------------------------------------------------- the LP

def solve_arb(legs, states, capital=None, min_edge=1e-9):
    """Maximise the worst-case profit of a prepaid portfolio.

    Variables are x (contracts per leg, >= 0) plus a scalar t = the profit
    floor. Constraints:  t - sum_j x_j (A[s,j] - cost_j) <= 0  for every state
    s, plus optional capital and per-leg depth caps.  A strictly positive t*
    is money that exists regardless of where the underlying prints.
    """
    if not legs:
        return None
    A = payoff_matrix(legs, states)
    cost = np.array([lg.cost for lg in legs])
    cap = np.array([lg.depth for lg in legs], dtype=float)
    n = len(legs)

    # net payoff per state, per unit of each leg
    N = A - cost[None, :]

    # maximise t  ->  minimise -t
    c = np.zeros(n + 1)
    c[-1] = -1.0

    A_ub = np.hstack([-N, np.ones((len(states), 1))])
    b_ub = np.zeros(len(states))

    if capital:
        row = np.zeros((1, n + 1))
        row[0, :n] = cost
        A_ub = np.vstack([A_ub, row])
        b_ub = np.concatenate([b_ub, [capital]])

    bounds = [(0.0, float(cp)) for cp in cap] + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    x = res.x[:n]
    t = res.x[-1]
    if t <= min_edge:
        return {"profit": float(t), "legs": [], "outlay": 0.0, "roi": 0.0,
                "min_payoff": 0.0, "max_payoff": 0.0, "x": x}

    picks = [(legs[j], x[j]) for j in range(n) if x[j] > 1e-6]
    outlay = float((x * cost).sum())
    pay = A @ x
    return {
        "profit": float(t),
        "outlay": outlay,
        "roi": float(t / outlay) if outlay > 0 else None,
        "legs": picks,
        "min_payoff": float(pay.min()),
        "max_payoff": float(pay.max()),
        "x": x,
    }
