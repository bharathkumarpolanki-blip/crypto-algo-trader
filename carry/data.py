"""
Market-data adapter for the carry harvester. Venue-selectable via
config.CARRY_DATA_VENUE (read-only, no API keys, no orders):

  • "hyperliquid" (default) — the Phase-2 TARGET venue. Perp funding + mark come
    from Hyperliquid (ccxt, keyless, HOURLY funding); the long-spot-leg price
    comes from Coinbase public (USD). This models the REAL cross-venue book —
    long spot on a spot venue + short perp on Hyperliquid — so the basis residual
    is genuine, not assumed away. Bonus: Coinbase AND Hyperliquid are both
    reachable from US AWS regions, so this sidesteps OKX's US geo-block on EC2.

  • "okx" — both legs on OKX (BTC/USDT spot + BTC/USDT:USDT perp), 8h funding.
    Single-venue, but OKX is US-geo-restricted → use only from a non-US host.

One snapshot() returns everything a token's carry decision needs: spot, perp mark,
funding rate, the funding interval (derived, not assumed), and annualized APR.
"""
import time
import logging
import ccxt
import config

logger = logging.getLogger("carry.data")
_clients: dict = {}


def _client(name: str) -> ccxt.Exchange:
    if name not in _clients:
        _clients[name] = getattr(ccxt, name)({"enableRateLimit": True})
    return _clients[name]


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


def _coinbase_spot(token: str) -> float | None:
    """Real spot price for the long leg from Coinbase public (keyless)."""
    try:
        from exchange.market_data import fetch_ticker
        t = fetch_ticker(f"{token}/USD")
        px = t.get("last") or t.get("close")
        return float(px) if px else None
    except Exception:
        return None


def _interval_hours(fr: dict, default: int) -> int:
    """Derive the funding interval from the settlement timestamps (8h/4h/1h)."""
    info = fr.get("info") or {}
    cur = fr.get("fundingTimestamp") or info.get("fundingTime")
    nxt = fr.get("nextFundingTimestamp") or info.get("nextFundingTime")
    try:
        cur, nxt = int(cur), int(nxt)
        if nxt > cur:
            return max(1, round((nxt - cur) / 3_600_000))
    except (TypeError, ValueError):
        pass
    return default


def _result(token, spot, mark, rate, interval_h, next_ts, venue) -> dict:
    periods = (24.0 / interval_h) * 365.0
    return {"token": token, "spot": float(spot), "mark": float(mark),
            "rate": float(rate), "interval_h": interval_h,
            "apr": float(rate) * periods, "next_ts": next_ts, "venue": venue}


def _snap_hyperliquid(token: str) -> dict:
    hl = _client("hyperliquid")
    perp = f"{token}/USDC:USDC"
    pt = _retry(lambda: hl.fetch_ticker(perp))
    mark = pt.get("last") or pt.get("mark") or pt.get("close")
    fr = _retry(lambda: hl.fetch_funding_rate(perp))
    rate = fr["fundingRate"]
    interval_h = _interval_hours(fr, default=1)          # Hyperliquid funds hourly
    spot = _coinbase_spot(token) or mark                 # real cross-venue spot; else proxy
    return _result(token, spot, mark, rate, interval_h,
                   fr.get("nextFundingTimestamp"), "hyperliquid")


def _snap_okx(token: str) -> dict:
    ex = _client("okx")
    spot_t = _retry(lambda: ex.fetch_ticker(f"{token}/USDT"))
    perp_t = _retry(lambda: ex.fetch_ticker(f"{token}/USDT:USDT"))
    fr = _retry(lambda: ex.fetch_funding_rate(f"{token}/USDT:USDT"))
    spot = spot_t.get("last") or spot_t.get("close")
    mark = perp_t.get("last") or perp_t.get("mark") or perp_t.get("close")
    interval_h = _interval_hours(fr, default=8)
    return _result(token, spot, mark, fr["fundingRate"], interval_h,
                   fr.get("nextFundingTimestamp"), "okx")


_VENUES = {"hyperliquid": _snap_hyperliquid, "okx": _snap_okx}


def snapshot(token: str, _quote: str = None) -> dict:
    """Normalized market snapshot for one token from config.CARRY_DATA_VENUE."""
    venue = getattr(config, "CARRY_DATA_VENUE", "hyperliquid").lower()
    fn = _VENUES.get(venue)
    if fn is None:
        raise ValueError(f"unknown CARRY_DATA_VENUE={venue!r} (use 'hyperliquid' or 'okx')")
    return fn(token)
