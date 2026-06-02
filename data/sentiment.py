"""
Sentiment module — 100% free, no API keys required.

Three sources combined:

1. Alternative.me Fear & Greed Index
   - Single market-wide score (0 = Extreme Fear, 100 = Extreme Greed)
   - Updates daily; cached for 1 hour
   - URL: https://api.alternative.me/fng/

2. CoinGecko Community Sentiment
   - Per-coin sentiment_votes_up_percentage (community bull/bear votes)
   - Also uses 24h price change as a momentum proxy
   - No key required on the free tier; cached 15 min per coin

3. RSS News Headlines (Bitcoinist + CryptoSlate + Decrypt + Bitcoin.com)
   - Scans latest headlines for coin-specific keywords
   - Positive/negative word scoring on titles
   - Cached 20 min per coin; fetched with a browser User-Agent

Final score: weighted average → normalised to [-1, +1]
  Fear & Greed : 30%
  CoinGecko    : 40%
  RSS headlines: 30%
"""

import time
import logging
import re
import requests
import feedparser

logger = logging.getLogger(__name__)

# ── Keyword lists ─────────────────────────────────────────────────────────────

_POSITIVE = {
    "surge", "rally", "bullish", "breakout", "adoption", "partnership",
    "upgrade", "milestone", "all-time high", "ath", "record", "launch",
    "approval", "etf", "institutional", "buy", "growth", "soar",
    "support", "listing", "integration", "gain", "rise", "positive",
    "recover", "accumulate", "moon", "pump", "outperform", "upgrade",
    "expand", "double", "triple", "highs", "boost", "strong",
}
_NEGATIVE = {
    "crash", "dump", "bearish", "hack", "exploit", "ban", "regulation",
    "lawsuit", "sec", "fraud", "scam", "rug", "delisting", "loss",
    "plunge", "drop", "fail", "attack", "liquidation", "fear", "concern",
    "warning", "reject", "sell", "collapse", "sanction", "weak",
    "decline", "fall", "slump", "trouble", "probe", "investigation",
    "halted", "suspend", "breach", "vulnerability", "stolen",
}

# ── CoinGecko coin ID mapping ─────────────────────────────────────────────────

_COINGECKO_ID = {
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "SOL":  "solana",
    "XRP":  "ripple",
    "ADA":  "cardano",
    "DOGE": "dogecoin",
    "LINK": "chainlink",
    "DOT":  "polkadot",
    "UNI":  "uniswap",
    "ATOM": "cosmos",
    "NEAR": "near",
    "AAVE": "aave",
    "CRV":  "curve-dao-token",
    "AVAX": "avalanche-2",
    "LTC":  "litecoin",
    "BNB":  "binancecoin",
}

# ── Coin keyword aliases (for headline matching) ──────────────────────────────

_COIN_KEYWORDS = {
    "BTC":  ["bitcoin", "btc"],
    "ETH":  ["ethereum", "eth", "ether"],
    "SOL":  ["solana", "sol"],
    "XRP":  ["xrp", "ripple"],
    "ADA":  ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "LINK": ["chainlink", "link"],
    "DOT":  ["polkadot", "dot"],
    "UNI":  ["uniswap", "uni"],
    "ATOM": ["cosmos", "atom"],
    "NEAR": ["near protocol", "near"],
    "AAVE": ["aave"],
    "CRV":  ["curve", "crv"],
    "AVAX": ["avalanche", "avax"],
    "LTC":  ["litecoin", "ltc"],
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoTradingBot/2.0)"}

# ── In-memory cache ───────────────────────────────────────────────────────────

_cache: dict = {}

def _cached(key: str, ttl_seconds: int):
    """Return cached value if still fresh, else None."""
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl_seconds:
        return entry["value"]
    return None

def _store(key: str, value):
    _cache[key] = {"ts": time.time(), "value": value}
    return value


# ── Source 1: Alternative.me Fear & Greed Index ───────────────────────────────

def _fear_greed_score() -> float:
    """
    Returns a normalised score in [-1, +1].
    0-29 (Fear/Extreme Fear) → negative
    30-60 (Neutral)          → near zero
    61-100 (Greed)           → positive
    Cached 1 hour (index updates daily).
    """
    cached = _cached("fear_greed", 3600)
    if cached is not None:
        return cached

    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1",
                         timeout=8, headers=_HEADERS)
        raw = int(r.json()["data"][0]["value"])   # 0–100
        # Normalise: 50 = 0, 0 = -1, 100 = +1
        score = (raw - 50) / 50.0
        logger.debug("Fear & Greed raw=%d  score=%.2f", raw, score)
        return _store("fear_greed", round(score, 3))
    except Exception as e:
        logger.debug("Fear & Greed fetch failed: %s", e)
        return _store("fear_greed", 0.0)


# ── Source 2: CoinGecko community sentiment ───────────────────────────────────

def _coingecko_score(symbol: str) -> float:
    """
    Combines:
    - sentiment_votes_up_percentage  (community bull/bear votes, 0-100)
    - price_change_percentage_24h    (momentum proxy)
    Cached 15 min per coin.
    """
    cache_key = f"cg_{symbol}"
    cached = _cached(cache_key, 900)
    if cached is not None:
        return cached

    coin_id = _COINGECKO_ID.get(symbol)
    if not coin_id:
        return _store(cache_key, 0.0)

    try:
        url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
               f"?localization=false&tickers=false"
               f"&market_data=true&community_data=true&sparkline=false")
        r = requests.get(url, timeout=10, headers=_HEADERS)
        if r.status_code == 429:
            logger.debug("CoinGecko rate-limited for %s", symbol)
            return _store(cache_key, 0.0)

        data = r.json()

        # Community vote score: 50% up = neutral (0), 100% up = +1, 0% up = -1
        votes_up = data.get("sentiment_votes_up_percentage") or 50.0
        vote_score = (votes_up - 50.0) / 50.0

        # 24h price change: clip at ±10% then scale to [-1, +1]
        price_chg = data.get("market_data", {}).get("price_change_percentage_24h") or 0.0
        price_score = max(-1.0, min(1.0, price_chg / 10.0))

        # Weighted average: votes 60%, price momentum 40%
        score = vote_score * 0.6 + price_score * 0.4
        logger.debug("CoinGecko %s votes_up=%.1f price_chg=%.2f score=%.2f",
                     symbol, votes_up, price_chg, score)
        return _store(cache_key, round(score, 3))

    except Exception as e:
        logger.debug("CoinGecko fetch failed for %s: %s", symbol, e)
        return _store(cache_key, 0.0)


# ── Source 3: RSS news headlines ──────────────────────────────────────────────

_RSS_FEEDS = [
    "https://bitcoinist.com/feed/",
    "https://cryptoslate.com/feed/",
    "https://decrypt.co/feed",
    "https://news.bitcoin.com/feed/",
]

def _fetch_headlines() -> list[str]:
    """Fetch all RSS headline titles. Cached 20 min (shared across coins)."""
    cached = _cached("rss_headlines", 1200)
    if cached is not None:
        return cached

    titles = []
    for url in _RSS_FEEDS:
        try:
            r = requests.get(url, timeout=8, headers=_HEADERS)
            if r.status_code == 200:
                feed = feedparser.parse(r.text)
                titles.extend(e.title for e in feed.entries if e.get("title"))
        except Exception as e:
            logger.debug("RSS fetch failed %s: %s", url, e)

    logger.debug("RSS: fetched %d headlines from %d feeds", len(titles), len(_RSS_FEEDS))
    return _store("rss_headlines", titles)


def _score_text(text: str) -> float:
    """Keyword sentiment score in [-1, +1]."""
    text = text.lower()
    pos = sum(1 for w in _POSITIVE if w in text)
    neg = sum(1 for w in _NEGATIVE if w in text)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def _rss_score(symbol: str) -> float:
    """
    Filter headlines for coin-specific mentions, score them.
    Returns average sentiment in [-1, +1]. Cached 20 min via headline cache.
    """
    keywords = _COIN_KEYWORDS.get(symbol, [symbol.lower()])
    titles   = _fetch_headlines()

    relevant = [t for t in titles
                if any(kw in t.lower() for kw in keywords)]

    if not relevant:
        return 0.0   # no news = neutral

    scores = [_score_text(t) for t in relevant]
    avg    = sum(scores) / len(scores)
    logger.debug("RSS %s: %d relevant headlines, avg_score=%.2f", symbol, len(relevant), avg)
    return round(avg, 3)


# ── Combined sentiment ────────────────────────────────────────────────────────

def get_sentiment(coin: str) -> dict:
    """
    Returns {"score": float[-1,1], "label": str, "source": str, "detail": dict}.

    Weights:
      Fear & Greed  30%  — macro market mood
      CoinGecko     40%  — coin-specific community + momentum
      RSS headlines 30%  — recent news tone
    """
    symbol = coin.split("/")[0].upper()

    fg    = _fear_greed_score()
    cg    = _coingecko_score(symbol)
    rss   = _rss_score(symbol)

    score = round(fg * 0.30 + cg * 0.40 + rss * 0.30, 3)
    score = max(-1.0, min(1.0, score))

    label  = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    source = "fear_greed+coingecko+rss"

    return {
        "score":  score,
        "label":  label,
        "source": source,
        "detail": {
            "fear_greed": round(fg, 3),
            "coingecko":  round(cg, 3),
            "rss":        round(rss, 3),
        },
    }
