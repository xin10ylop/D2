"""Shared HTTP + time helpers for the multi-venue crypto data pull."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (compatible; quant-research/1.0)"
DATA = "/home/user/D2/data"


def http_json(url, params=None, headers=None, retries=4, timeout=30):
    """GET a JSON endpoint with exponential backoff on transient failures."""
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    hdrs = {"User-Agent": UA, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    delay = 1.0
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            last = f"HTTP {e.code} {body}"
            if e.code in (400, 404):  # not transient
                break
            time.sleep(delay)
            delay *= 2
        except Exception as e:  # noqa: BLE001 - network layer is best-effort
            last = repr(e)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed {url}: {last}")


def pmap(fn, items, workers=8):
    """Thread-pooled map that returns (item, result_or_None) pairs."""
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for item, res in zip(items, ex.map(lambda i: _safe(fn, i), items)):
            out.append((item, res))
    return out


def _safe(fn, i):
    try:
        return fn(i)
    except Exception as e:  # noqa: BLE001
        return {"__error__": repr(e)}


def save(name, obj):
    path = f"{DATA}/{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    return path


def load(name):
    with open(f"{DATA}/{name}.json") as f:
        return json.load(f)


def now_ms():
    return int(time.time() * 1000)
