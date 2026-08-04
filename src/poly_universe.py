"""Series-driven discovery of every live Polymarket crypto price market.

The earlier Polymarket work guessed slugs -- `{asset}-above-on-august-{d}-2026`
and friends -- for four assets. That was the mistake. On Kalshi I enumerated the
series registry and found 272 series; on Polymarket I hand-wrote a dozen slug
templates and never saw the rest of the exchange. The single most valuable
market on Kalshi (the BNB annual one-touch ladder) has an exact Polymarket twin,
`what-price-will-bnb-hit-before-2027`, and no slug template I wrote could have
produced it.

Gamma exposes /series, which is the direct counterpart of Kalshi's series list,
and /events?series_slug=... expands one. This module walks that registry and
normalises every live market into a predicate over a single settlement instant
or window, so the arbitrage scanner can reason about them together.

Families found and what they resolve on (all Binance {COIN}/USDT, which matters
-- claims that settle on different sources are not comparable):

    *-multi-strikes-*   above(K)   1m close at 12:00 ET on date D
    *-neg-risk-*        range      same candle, same instant
    *-up-or-down-*      above(P0)  same candle, vs a KNOWN earlier close P0
    *-hit-price-*       touch      1m high/low over a window

The third one is the link the old scanner could not see: "Bitcoin Up or Down on
August 4" is not a separate kind of question, it is the above-ladder contract
struck at the August 3 noon close. Once P0 is fetched it joins the same
partition as the strike ladders.
"""
import datetime as dt
import json
import re

from common import http_json, pmap, save

GAMMA = "https://gamma-api.polymarket.com"
BINANCE = "https://data-api.binance.vision/api/v3"

NUM = re.compile(r"[\d,]+(?:\.\d+)?")

# coin token appearing in a series slug -> canonical asset
COINS = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "xrp": "XRP",
    "dogecoin": "DOGE", "doge": "DOGE",
    "bnb": "BNB",
    "hype": "HYPE",
    "cardano": "ADA", "ada": "ADA",
    "litecoin": "LTC", "ltc": "LTC",
    "chainlink": "LINK", "link": "LINK",
    "avalanche": "AVAX", "avax": "AVAX",
    "shiba": "SHIB", "pepe": "PEPE", "sui": "SUI", "aptos": "APT",
    "toncoin": "TON", "tron": "TRX", "stellar": "XLM", "monero": "XMR",
}
# ratio and index series: one leg is not a coin, so they get their own asset id
# and never join a coin's partition
PAIRS = ("ethbtc", "soleth", "dominance")

FAMILIES = ("hit-price", "multi-strikes", "neg-risk", "up-or-down", "weeklies")


def ms(s):
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000


def nums(t):
    return [float(x.replace(",", "")) for x in NUM.findall(t or "")]


def series_asset(slug):
    """Canonical asset for a series slug, or None if it is not a single coin."""
    if any(p in slug for p in PAIRS):
        return None
    toks = slug.split("-")
    for t in toks:
        if t in COINS:
            return COINS[t]
    return None


def all_series():
    out, off = [], 0
    while True:
        d = http_json(f"{GAMMA}/series", {"limit": 100, "offset": off})
        if not d:
            break
        out += d
        off += 100
        if off > 5000:
            break
    return out


def crypto_series():
    """Series slugs that quote a coin's price, plus the ones the registry omits.

    The registry is not complete -- `bitcoin-neg-risk-weekly` is live and
    carries the daily range ladders, but does not come back from /series. Any
    series slug seen on an event we fetch is folded back in.
    """
    reg = {s["slug"] for s in all_series()}
    seeds = {"bitcoin-neg-risk-weekly", "ethereum-neg-risk-weekly",
             "solana-neg-risk-weekly", "xrp-neg-risk-weekly",
             "btc-multi-strikes-weekly", "ethereum-multi-strikes-weekly",
             "solana-multi-strikes-weekly", "xrp-multi-strikes-weekly",
             "bitcoin-hit-price-monthly", "ethereum-hit-price-monthly",
             "solana-hit-price-monthly", "bnb-hit-price-monthly",
             "xrp-hit-price-weekly", "bitcoin-hit-price-weekly",
             "ethereum-hit-price-weekly", "solana-hit-price-weekly"}
    cand = {s for s in reg | seeds
            if any(f in s for f in FAMILIES)
            and (series_asset(s) or any(p in s for p in PAIRS))}
    return sorted(cand)


def events_of(slug):
    try:
        return http_json(f"{GAMMA}/events",
                         {"series_slug": slug, "closed": "false", "limit": 100})
    except Exception:                                    # noqa: BLE001
        return []


def kind_of(series_slug):
    if "hit-price" in series_slug or "weeklies" in series_slug:
        return "touch"
    if "up-or-down" in series_slug:
        return "updown"
    if "multi-strikes" in series_slug:
        return "above"
    if "neg-risk" in series_slug:
        return "range"
    return None


REF_TIME = re.compile(
    r"candle for ([A-Z0-9]+)/USDT\s+([A-Z][a-z]{2} \d{1,2} '\d{2} \d{1,2}:\d{2})")


def ref_close(symbol, when_ms):
    """Binance 1m close at a given instant -- the up/down market's strike."""
    try:
        k = http_json(f"{BINANCE}/klines",
                      {"symbol": f"{symbol}USDT", "interval": "1m",
                       "startTime": int(when_ms), "limit": 1})
        return float(k[0][4]) if k else None
    except Exception:                                    # noqa: BLE001
        return None


def updown_strike(m, asset):
    """Reference close P0 for an up/down market, from its own description.

    'the Close price for the Binance 1 minute candle for BTC/USDT Aug 3 '26
    12:00 in the ET timezone (noon) is lower than the final Close price for the
    Aug 4 '26 12:00 ET candle'  ->  Up  <=>  close(Aug 4 noon ET) > P0.
    """
    d = (m.get("description") or "").replace("’", "'")
    hit = REF_TIME.search(d)
    if not hit:
        return None, None
    sym, stamp = hit.group(1), hit.group(2)
    try:
        t = dt.datetime.strptime(stamp, "%b %d '%y %H:%M")
    except ValueError:
        return None, None
    # ET timezone; 12:00 ET = 16:00Z in DST, 17:00Z otherwise. August is DST.
    off = 4 if 3 <= t.month <= 10 else 5
    t = t.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=off)
    return sym, t.timestamp() * 1000


def claims():
    """Every live crypto claim on Polymarket, normalised to a predicate."""
    ser = crypto_series()
    got = {}
    for s, evs in pmap(events_of, ser, workers=12):
        for e in evs or []:
            got[e["slug"]] = (s, e)
    print(f"crypto series queried: {len(ser)}   live events: {len(got)}")

    out, updown = [], []
    for s, e in got.values():
        kind = kind_of(s)
        asset = series_asset(s)
        if kind is None or asset is None:
            continue
        t0, t1 = ms(e.get("startDate")), ms(e.get("endDate"))
        if t1 is None:
            continue
        for m in e.get("markets", []):
            if m.get("closed"):
                continue
            bid, ask = m.get("bestBid"), m.get("bestAsk")
            if bid is None and ask is None:
                continue
            toks = m.get("clobTokenIds")
            try:
                toks = json.loads(toks) if isinstance(toks, str) else toks
            except Exception:                            # noqa: BLE001
                toks = None
            title = (m.get("groupItemTitle") or "").strip()
            base = {"asset": asset, "series": s, "slug": e["slug"],
                    "market": m.get("slug"), "title": title or m.get("question"),
                    "start": t0, "end": t1, "bid": bid or 0.0, "ask": ask,
                    "yes": toks[0] if toks else None,
                    "no": toks[1] if toks and len(toks) > 1 else None}
            if kind == "updown":
                sym, when = updown_strike(m, asset)
                if when is None:
                    continue
                updown.append({**base, "kind": "updown", "sym": sym, "ref_ms": when})
                continue
            n = nums(title)
            if not n:
                continue
            if kind == "touch":
                up = title.startswith("↑")
                dn = title.startswith("↓")
                if not (up or dn):
                    continue
                out.append({**base, "kind": "touch_up" if up else "touch_dn", "K": n[0]})
            elif kind == "above":
                out.append({**base, "kind": "above", "K": n[0]})
            else:
                if title.startswith("<"):
                    lo, hi = 0.0, n[0]
                elif title.startswith(">"):
                    lo, hi = n[0], float("inf")
                elif len(n) >= 2:
                    lo, hi = n[0], n[1]
                else:
                    continue
                out.append({**base, "kind": "range", "lo": lo, "hi": hi})

    # resolve every up/down market's strike from Binance, then fold it in as an
    # `above` claim on exactly the settlement instant its ladder uses
    keys = sorted({(u["sym"], u["ref_ms"]) for u in updown})
    px = {}
    for k, v in pmap(lambda kk: ref_close(kk[0], kk[1]), keys, workers=10):
        if v:
            px[k] = v
    for u in updown:
        p0 = px.get((u["sym"], u["ref_ms"]))
        if not p0:
            continue
        out.append({**u, "kind": "above", "K": p0, "from_updown": True})
    print(f"up/down markets converted to strike claims: "
          f"{sum(1 for c in out if c.get('from_updown'))}")
    return out


def main():
    cl = claims()
    kinds, assets = {}, {}
    for c in cl:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        assets[c["asset"]] = assets.get(c["asset"], 0) + 1
    print(f"\nlive claims: {len(cl)}")
    print("  by kind:  " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print("  by asset: " + "  ".join(f"{k}={v}" for k, v in sorted(assets.items())))
    save("poly_universe", cl)
    return cl


if __name__ == "__main__":
    main()
