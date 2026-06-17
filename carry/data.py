"""
OKX keyless market-data adapter for the carry harvester.

Why OKX: Binance (451) and Bybit (403) are geo-blocked from here; OKX public data
(spot ticker, perp ticker, funding rate) works without an API key. We only ever
READ — no auth, no orders. One snapshot() call returns everything a token's carry
decision needs: spot price, perp mark, the current funding rate, the funding
interval (8h/4h — derived, not assumed), and the annualized funding APR.
"""
import time
import logging
import ccxt

logger = logging.getLogger("carry.data")
_ex = None


def get_okx() -> ccxt.Exchange:
    global _ex
    if _ex is None:
        _ex = ccxt.okx({"enableRateLimit": True})
    return _ex


def _retry(fn, tries: int = 3):
    """Resilient read: never let one flaky HTTP call kill a harvest cycle."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:        # noqa: BLE001 — deliberately broad; we retry
            last = e
            time.sleep(0.8 * (i + 1))
    raise last


def snapshot(token: str, quote: str = "USDT") -> dict:
    """
    Return a normalized market snapshot for one token, or raise on total failure.

      {token, spot, mark, rate, interval_h, apr, next_ts}

    rate       = funding rate for the current interval (fraction; +ve → shorts earn)
    interval_h = funding interval in hours (derived from the settlement timestamps)
    apr        = rate annualized = rate * (24/interval_h) * 365
    """
    ex = get_okx()
    spot_sym = f"{token}/{quote}"
    perp_sym = f"{token}/{quote}:{quote}"

    spot_t = _retry(lambda: ex.fetch_ticker(spot_sym))
    perp_t = _retry(lambda: ex.fetch_ticker(perp_sym))
    fr = _retry(lambda: ex.fetch_funding_rate(perp_sym))

    spot = float(spot_t.get("last") or spot_t.get("close"))
    mark = float(perp_t.get("last") or perp_t.get("mark") or perp_t.get("close"))
    rate = float(fr["fundingRate"])

    info = fr.get("info") or {}
    cur_ts = fr.get("fundingTimestamp") or info.get("fundingTime")
    next_ts = fr.get("nextFundingTimestamp") or info.get("nextFundingTime")
    cur_ts = int(cur_ts) if cur_ts else None
    next_ts = int(next_ts) if next_ts else None

    interval_h = 8                                  # OKX default; refine if we can
    if cur_ts and next_ts and next_ts > cur_ts:
        interval_h = max(1, round((next_ts - cur_ts) / 3_600_000))

    periods = (24.0 / interval_h) * 365.0
    return {
        "token": token, "spot": spot, "mark": mark, "rate": rate,
        "interval_h": interval_h, "apr": rate * periods, "next_ts": next_ts,
    }
