"""
Dynamic universe manager.

Every REFRESH_INTERVAL_HOURS it re-scans all active Coinbase USD pairs and
selects the top N coins based on a composite score of:
  - 24h volume (liquidity filter — must be above MIN_VOLUME_USD)
  - 24h price change % (trend momentum)
  - 7-day ROC computed from 6h OHLCV (medium-term uptrend confirmation)

Only coins passing ALL filters enter the tradeable universe.
The result is cached and exposed as get_watchlist().
"""

import time
import logging
import threading
import pandas as pd
import ccxt

import config
from market_data import get_exchange, fetch_ohlcv

logger = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────────

REFRESH_INTERVAL_HOURS = 1      # re-rank the universe every hour
TOP_N_COINS            = 15     # max symbols in the active watchlist
MIN_VOLUME_USD         = 5_000_000   # minimum 24h volume to be considered
MIN_PRICE_USD          = 0.001       # filter out dust/dead coins

# Coins to always keep regardless of ranking (anchors)
ANCHOR_SYMBOLS = {"BTC/USD", "ETH/USD", "SOL/USD"}

# Coins permanently blacklisted (stablecoins, wrapped tokens, LP tokens)
BLACKLIST = {
    "USDT/USD", "USDC/USD", "BUSD/USD", "DAI/USD", "TUSD/USD",
    "WBTC/USD", "CBETH/USD", "WETH/USD", "STETH/USD",
    "USDT/USD", "EURC/USD", "PYUSD/USD",
}

# ── Shared state ──────────────────────────────────────────────────────────────

_lock          = threading.Lock()
_watchlist:  list[str] = list(config.WATCHLIST)   # starts with static list
_last_refresh: float   = 0.0
_refresh_log:  list[dict] = []    # history of universe refreshes for dashboard


def get_watchlist() -> list[str]:
    """Return current dynamic watchlist (thread-safe)."""
    with _lock:
        return list(_watchlist)


def get_refresh_log() -> list[dict]:
    """Return last N refresh records for the dashboard."""
    with _lock:
        return list(_refresh_log[-20:])


# ── Discovery logic ───────────────────────────────────────────────────────────

def _all_usd_pairs() -> list[str]:
    """Return all active spot USD pairs on Coinbase."""
    try:
        ex = get_exchange()
        markets = ex.load_markets()
        pairs = [
            s for s, m in markets.items()
            if s.endswith("/USD")
            and m.get("spot", False)
            and m.get("active", False)
            and s not in BLACKLIST
        ]
        return pairs
    except Exception as e:
        logger.error("Failed to load Coinbase markets: %s", e)
        return list(config.WATCHLIST)


def _score_coin(sym: str, ticker: dict) -> float | None:
    """
    Composite score for ranking. Returns None if coin fails filters.
    Higher score = stronger uptrend candidate.
    """
    quote_vol  = ticker.get("quoteVolume") or 0
    pct_change = ticker.get("percentage")  or 0   # 24h %
    last_price = ticker.get("last")        or 0

    # Hard filters
    if quote_vol  < MIN_VOLUME_USD: return None
    if last_price < MIN_PRICE_USD:  return None

    # Volume score: log-scaled so BTC doesn't dominate
    import math
    vol_score = math.log10(max(quote_vol, 1))   # ~6-9 range

    # Momentum score: 24h change, clipped at ±20%
    mom_score = max(-20, min(20, pct_change))

    # Combined: 50% volume + 50% momentum (normalised roughly)
    return (vol_score / 9.0) * 50 + (mom_score / 20.0) * 50


def _compute_7d_roc(sym: str) -> float:
    """Fetch 6h candles and compute 7-day ROC. Returns 0 on failure."""
    try:
        df = fetch_ohlcv(sym, "6h", limit=30)
        if df.empty or len(df) < 28:
            return 0.0
        roc = (df["close"].iloc[-1] - df["close"].iloc[-28]) / df["close"].iloc[-28] * 100
        return float(roc)
    except Exception:
        return 0.0


def refresh_universe(force: bool = False) -> list[str]:
    """
    Re-score all Coinbase USD pairs and update the active watchlist.
    Called automatically by the bot loop; can also be forced from the UI.
    """
    global _last_refresh

    with _lock:
        if not force and (time.time() - _last_refresh) < REFRESH_INTERVAL_HOURS * 3600:
            return list(_watchlist)

    logger.info("Universe refresh starting — scanning all Coinbase USD pairs…")

    # Step 1: Fetch all pairs and their 24h tickers
    all_pairs = _all_usd_pairs()
    ex = get_exchange()

    try:
        tickers = ex.fetch_tickers(all_pairs)
    except Exception as e:
        logger.error("Batch ticker fetch failed: %s", e)
        return get_watchlist()

    # Step 2: Score every coin
    scored: list[tuple[float, str]] = []
    for sym, ticker in tickers.items():
        if sym in BLACKLIST:
            continue
        score = _score_coin(sym, ticker)
        if score is not None:
            scored.append((score, sym))

    # Sort by composite score descending
    scored.sort(reverse=True)

    # Step 3: Take top 40 candidates, then verify 7-day uptrend via OHLCV
    # (avoids coins that pumped today but are otherwise downtrending)
    candidates = [sym for _, sym in scored[:40]]
    logger.info("Top 40 candidates by 24h volume+momentum: %s", candidates[:10])

    roc_scores: list[tuple[float, str]] = []
    for sym in candidates:
        roc = _compute_7d_roc(sym)
        # Weight: 7d ROC adds to the ranking but doesn't hard-block
        base = next(s for s, sy in scored if sy == sym)
        final = base + (roc / 10.0) * 20   # 7d ROC contributes up to 20 pts
        roc_scores.append((final, sym))
        time.sleep(0.15)   # rate limit

    roc_scores.sort(reverse=True)

    # Step 4: Always include anchors; fill remaining slots from ranked list
    new_watchlist: list[str] = []
    for sym in ANCHOR_SYMBOLS:
        if sym in tickers:
            new_watchlist.append(sym)

    for _, sym in roc_scores:
        if sym not in new_watchlist:
            new_watchlist.append(sym)
        if len(new_watchlist) >= TOP_N_COINS:
            break

    # Fallback: if we got fewer than 5, keep previous watchlist
    if len(new_watchlist) < 5:
        logger.warning("Universe refresh returned too few coins — keeping previous list")
        return get_watchlist()

    with _lock:
        old = list(_watchlist)
        _watchlist.clear()
        _watchlist.extend(new_watchlist)
        _last_refresh = time.time()

        added   = [s for s in new_watchlist if s not in old]
        removed = [s for s in old           if s not in new_watchlist]

        record = {
            "time":     pd.Timestamp.now(tz="UTC").isoformat()[:19],
            "symbols":  list(new_watchlist),
            "added":    added,
            "removed":  removed,
            "total":    len(new_watchlist),
        }
        _refresh_log.append(record)

    logger.info("Universe refreshed → %d symbols | added=%s | removed=%s",
                len(new_watchlist), added, removed)
    return new_watchlist


def start_background_refresh() -> None:
    """Launch a daemon thread that refreshes the universe every hour."""
    def _loop():
        while True:
            try:
                refresh_universe()
            except Exception as e:
                logger.error("Universe refresh error: %s", e)
            time.sleep(REFRESH_INTERVAL_HOURS * 3600)

    t = threading.Thread(target=_loop, daemon=True, name="universe-refresh")
    t.start()
    logger.info("Dynamic universe manager started (refresh every %dh)", REFRESH_INTERVAL_HOURS)
