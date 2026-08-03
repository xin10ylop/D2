"""Inventory, limits and delta hedging.

Every Kalshi digital carries a delta in probability-per-dollar-of-underlying.
Summed across the ladder that is a real dollar exposure, and near settlement it
gets very large (a digital's delta blows up as time to expiry goes to zero),
which is why `min_seconds_to_settle` exists and why the hedge runs on the
Deribit perpetual rather than being ignored.
"""
import time
from collections import defaultdict

from config import CFG


class Book:
    def __init__(self):
        self.pos = defaultdict(int)          # ticker -> signed contracts (YES +)
        self.cash = 0.0                      # realised cash flow, dollars
        self.fees = 0.0
        self.fills = []
        self.hedge_usd = defaultdict(float)  # asset -> perp notional held
        self.start = time.time()
        self.day_pnl = 0.0

    # ---------------------------------------------------------- accounting
    def fill(self, ticker, side, qty, price, fee, asset, meta=None):
        signed = qty if side == "YES" else -qty
        self.pos[ticker] += signed
        self.cash -= qty * price          # both sides are prepaid on Kalshi
        self.fees += fee
        self.day_pnl -= fee
        self.fills.append({"t": time.time(), "ticker": ticker, "side": side,
                           "qty": qty, "price": price, "fee": fee,
                           "asset": asset, **(meta or {})})

    def gross(self):
        return sum(abs(v) for v in self.pos.values())

    def event_exposure(self, tickers):
        return sum(abs(self.pos[t]) for t in tickers)

    # ------------------------------------------------------------- greeks
    def net_delta_usd(self, deltas):
        """deltas: ticker -> dP/dS. Position delta in dollars per $1 of spot."""
        d = defaultdict(float)
        for tk, q in self.pos.items():
            if tk in deltas:
                d[deltas[tk][0]] += q * deltas[tk][1]
        return d

    # -------------------------------------------------------------- limits
    def check(self, cfg=CFG):
        """Returns a list of breached limits; empty means clear to trade."""
        bad = []
        if self.gross() > cfg.max_gross_contracts:
            bad.append(f"gross {self.gross()} > {cfg.max_gross_contracts}")
        if -self.cash > cfg.max_notional_usd:
            bad.append(f"notional ${-self.cash:,.0f} > ${cfg.max_notional_usd:,.0f}")
        if self.day_pnl < -cfg.daily_loss_limit_usd:
            bad.append(f"daily loss ${self.day_pnl:,.0f}")
        return bad


def mark_to_model(book, fair_by_ticker):
    """Unrealised P&L against the model's own fair values."""
    v = 0.0
    for tk, q in book.pos.items():
        f = fair_by_ticker.get(tk)
        if f is None:
            continue
        v += q * f if q > 0 else abs(q) * (1.0 - f) * -1 + 0.0
    return v


def portfolio_value(book, fair_by_ticker):
    """Model value of open positions (YES positive, NO negative convention)."""
    v = 0.0
    for tk, q in book.pos.items():
        f = fair_by_ticker.get(tk)
        if f is None:
            continue
        v += (q * f) if q >= 0 else (q * f)   # a short YES is worth -q*(1-f) - q
    return v


def hedge_required(net_delta_usd, cfg=CFG):
    """How much perp notional to trade, respecting the dead-band."""
    out = {}
    for asset, d in net_delta_usd.items():
        if abs(d) > cfg.hedge_band_usd:
            out[asset] = -d           # trade the opposite of the exposure
    return out
