"""Venue adapters. Reads are public; anything that places an order is gated.

Kalshi authentication is RSA-PSS request signing: an API key ID plus a private
key, set as KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH. Without them the adapter
is read-only and the bot runs in paper mode, which is the default.
"""
import base64
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, "/home/user/D2/src")
from common import http_json, pmap                     # noqa: E402

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
DERIBIT = "https://www.deribit.com/api/v2"


# ------------------------------------------------------------------ Kalshi

class Kalshi:
    def __init__(self, cfg):
        self.cfg = cfg
        self.key_id = os.getenv("KALSHI_KEY_ID")
        self.pkey = None
        p = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if p and os.path.exists(p):
            try:
                from cryptography.hazmat.primitives.serialization import load_pem_private_key
                with open(p, "rb") as f:
                    self.pkey = load_pem_private_key(f.read(), password=None)
            except Exception as e:                      # noqa: BLE001
                print(f"[kalshi] private key not loaded: {e}")

    @property
    def can_trade(self):
        return bool(self.key_id and self.pkey and self.cfg.live)

    def _sign(self, method, path):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        msg = (ts + method + path).encode()
        sig = self.pkey.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    # ---- reads (public)
    def open_events(self, series):
        d = http_json(f"{KALSHI}/events",
                      {"series_ticker": series, "status": "open", "limit": 200})
        return d.get("events", [])

    def markets(self, event_ticker):
        d = http_json(f"{KALSHI}/markets",
                      {"event_ticker": event_ticker, "limit": 500})
        return d.get("markets", [])

    def orderbook(self, ticker, depth=20):
        d = http_json(f"{KALSHI}/markets/{ticker}/orderbook", {"depth": depth})
        ob = d.get("orderbook_fp") or d.get("orderbook") or {}
        return {"yes": [[float(p), float(q)] for p, q in (ob.get("yes_dollars") or [])],
                "no": [[float(p), float(q)] for p, q in (ob.get("no_dollars") or [])]}

    def books(self, tickers, workers=12):
        out = {}
        for t, b in pmap(self.orderbook, tickers, workers=workers):
            if isinstance(b, dict) and "__error__" not in b:
                out[t] = b
        return out

    # ---- writes (gated)
    def place(self, ticker, side, action, count, price_cents, post_only=True):
        """side: yes|no, action: buy|sell, price in integer cents."""
        if not self.can_trade:
            return {"paper": True, "ticker": ticker, "side": side,
                    "action": action, "count": count, "price": price_cents}
        path = "/trade-api/v2/portfolio/orders"
        body = {"ticker": ticker, "action": action, "side": side,
                "count": int(count), "type": "limit",
                f"{side}_price": int(price_cents),
                "time_in_force": "fill_or_kill" if not post_only else None,
                "post_only": bool(post_only),
                "client_order_id": f"bot-{int(time.time()*1000)}-{ticker[-8:]}"}
        body = {k: v for k, v in body.items() if v is not None}
        req = urllib.request.Request(
            KALSHI + "/portfolio/orders", data=json.dumps(body).encode(),
            headers={**self._sign("POST", path), "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def cancel(self, order_id):
        if not self.can_trade:
            return {"paper": True, "cancel": order_id}
        path = f"/trade-api/v2/portfolio/orders/{order_id}"
        req = urllib.request.Request(KALSHI + f"/portfolio/orders/{order_id}",
                                     method="DELETE", headers=self._sign("DELETE", path))
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def positions(self):
        if not self.can_trade:
            return {"paper": True, "positions": []}
        path = "/trade-api/v2/portfolio/positions"
        req = urllib.request.Request(KALSHI + "/portfolio/positions",
                                     headers=self._sign("GET", path))
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())


# ----------------------------------------------------------------- Deribit

class Deribit:
    """Public market data + (optionally) the perpetual hedge leg."""

    NEAR = {"option_expiries": 3}

    def __init__(self, cfg):
        self.cfg = cfg
        self.token = None
        cid, csec = os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET")
        if cid and csec and cfg.live:
            try:
                d = http_json(f"{DERIBIT}/public/auth",
                              {"grant_type": "client_credentials",
                               "client_id": cid, "client_secret": csec})
                self.token = d["result"]["access_token"]
            except Exception as e:                       # noqa: BLE001
                print(f"[deribit] auth failed, hedging disabled: {e}")

    @property
    def can_trade(self):
        return bool(self.token)

    def snapshot(self, ccy):
        """The blob shape src/rnd.TermSurface expects."""
        blob = {"index": http_json(f"{DERIBIT}/public/get_index_price",
                                   {"index_name": f"{ccy.lower()}_usd"})["result"],
                "instruments": {}, "book_summary": {}}
        for kind in ("option", "future"):
            blob["instruments"][kind] = http_json(
                f"{DERIBIT}/public/get_instruments",
                {"currency": ccy, "kind": kind, "expired": "false"})["result"]
            blob["book_summary"][kind] = http_json(
                f"{DERIBIT}/public/get_book_summary_by_currency",
                {"currency": ccy, "kind": kind})["result"]
        return blob

    def perp_price(self, ccy):
        r = http_json(f"{DERIBIT}/public/ticker",
                      {"instrument_name": self.cfg.hedge_instrument[ccy]})["result"]
        return r["mark_price"]

    def hedge(self, ccy, usd_notional):
        """Market order on the perpetual to neutralise delta. Sign = direction."""
        if not self.can_trade:
            return {"paper": True, "ccy": ccy, "usd": usd_notional}
        inst = self.cfg.hedge_instrument[ccy]
        # Deribit perps are quoted in $10 contracts for BTC, $1 for ETH
        lot = 10 if ccy == "BTC" else 1
        amt = int(abs(usd_notional) // lot) * lot
        if amt < lot:
            return {"skipped": "below lot size"}
        side = "buy" if usd_notional > 0 else "sell"
        url = f"{DERIBIT}/private/{side}"
        req = urllib.request.Request(
            url + "?" + urllib.parse.urlencode(
                {"instrument_name": inst, "amount": amt, "type": "market"}),
            headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
