"""Fetch OHLCV candles and order-book data from exchange via CCXT."""

import time
import logging
import pandas as pd
import ccxt
import config

logger = logging.getLogger(__name__)


def _wrap_coinbase_secret(secret: str) -> str:
    """
    Coinbase CDP API secrets are EC private keys.
    CCXT's jwt() calls from_pem(), which requires full PEM wrapping.
    If the user pasted only the base64 body (no headers), add them.
    """
    secret = secret.strip()
    if secret.startswith("-----"):
        return secret                    # already valid PEM
    # Normalise: the key body may have literal \n or real newlines
    body = secret.replace("\\n", "\n").strip()
    return f"-----BEGIN EC PRIVATE KEY-----\n{body}\n-----END EC PRIVATE KEY-----\n"


def _build_exchange(authenticated: bool = True) -> ccxt.Exchange:
    """
    Build a CCXT exchange instance.

    authenticated=True  → includes API keys, used for trading (orders, balance).
    authenticated=False → NO keys, used for public market data (candles, tickers).

    Why two clients? CCXT's Coinbase driver calls the *authenticated*
    transaction_summary endpoint during load_markets() to fetch fee tiers.
    If the API key has any issue this returns 401 and breaks even public
    candle fetches. A keyless client skips that call entirely, so public
    data always works regardless of key state.
    """
    params = {
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }

    if authenticated and config.API_KEY and config.API_SECRET:
        secret = config.API_SECRET
        if config.EXCHANGE == "coinbase":
            secret = _wrap_coinbase_secret(secret)
        params["apiKey"] = config.API_KEY
        params["secret"] = secret
        if config.EXCHANGE == "coinbase":
            params["options"]["advanced"] = True

    exchange_cls = getattr(ccxt, config.EXCHANGE)
    ex = exchange_cls(params)

    # Coinbase has no public sandbox — sandbox flag is ignored for coinbase
    if config.TESTNET and config.EXCHANGE != "coinbase":
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            pass
    return ex


_exchange:        ccxt.Exchange | None = None   # authenticated — trading
_public_exchange: ccxt.Exchange | None = None   # keyless — public data


def get_exchange() -> ccxt.Exchange:
    """Authenticated client — for placing orders and reading balance."""
    global _exchange
    if _exchange is None:
        _exchange = _build_exchange(authenticated=True)
    return _exchange


def get_public_exchange() -> ccxt.Exchange:
    """Keyless client — for public market data (candles, tickers, order book)."""
    global _public_exchange
    if _public_exchange is None:
        _public_exchange = _build_exchange(authenticated=False)
    return _public_exchange


_TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "1d": 86_400_000,
}
_PAGE_SIZE = 300   # Coinbase caps at 300 candles per request


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = config.CANDLE_LIMIT) -> pd.DataFrame:
    """
    Return OHLCV DataFrame. Paginates automatically for exchanges (like Coinbase)
    that cap each request at 300 candles.
    """
    ex = get_public_exchange()   # keyless — public candle data
    tf_ms = _TF_MS.get(timeframe, 3_600_000)

    all_raw: list = []
    # Work backwards: start from 'now - limit*tf' and page forward
    import time as _time
    since_ms = int(_time.time() * 1000) - limit * tf_ms

    remaining = limit
    retries = 3

    while remaining > 0:
        fetch_n = min(remaining, _PAGE_SIZE)
        for attempt in range(retries):
            try:
                raw = ex.fetch_ohlcv(symbol, timeframe=timeframe,
                                     since=since_ms, limit=fetch_n)
                break
            except ccxt.NetworkError as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Network error fetching %s %s: %s", symbol, timeframe, e)
                    raw = []
                    break
            except ccxt.BaseError as e:
                logger.error("CCXT error fetching %s %s: %s", symbol, timeframe, e)
                raw = []
                break

        if not raw:
            break

        all_raw.extend(raw)
        remaining -= len(raw)
        since_ms = raw[-1][0] + tf_ms   # advance past last candle
        if len(raw) < fetch_n:
            break   # exchange returned fewer than requested — we're at the end
        time.sleep(0.2)   # respect rate limits

    if not all_raw:
        return pd.DataFrame()

    df = pd.DataFrame(all_raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_ticker(symbol: str) -> dict:
    ex = get_public_exchange()   # keyless — public ticker
    try:
        return ex.fetch_ticker(symbol)
    except Exception as e:
        logger.warning("Ticker fetch failed for %s: %s", symbol, e)
        return {}


def fetch_order_book(symbol: str, depth: int = 10) -> dict:
    ex = get_public_exchange()   # keyless — public order book
    try:
        return ex.fetch_order_book(symbol, limit=depth)
    except Exception as e:
        logger.warning("Order book fetch failed for %s: %s", symbol, e)
        return {}


def get_balance() -> dict:
    """Return free balance dict keyed by currency (USD for Coinbase, USDT for others)."""
    ex = get_exchange()
    try:
        balance = ex.fetch_balance()
        return balance.get("free", {})
    except Exception as e:
        logger.error("Balance fetch failed: %s", e)
        return {}


def check_auth() -> tuple[bool, str]:
    """
    Test whether the authenticated API key actually works by calling a
    private endpoint (fetch_balance). Returns (ok, message).

    ok=True  → key is valid, trading will work
    ok=False → key is missing/invalid; live trading would fail (paper trading is fine)
    """
    if not config.API_KEY or not config.API_SECRET:
        return False, "No API key configured (.env API_KEY / API_SECRET empty)"
    try:
        ex = get_exchange()
        ex.fetch_balance()   # private call — fails with 401 if key is bad
        return True, "API key valid — authenticated trading available"
    except ccxt.AuthenticationError as e:
        return False, f"Authentication failed (401) — key invalid/expired/missing Trade permission: {e}"
    except ccxt.PermissionDenied as e:
        return False, f"Permission denied — key lacks Trade permission or IP not allowlisted: {e}"
    except Exception as e:
        return False, f"Auth check failed: {e}"


def get_quote_balance() -> float:
    """Return available quote-currency balance (USD on Coinbase, USDT elsewhere)."""
    quote = getattr(config, "QUOTE_CURRENCY", "USDT")
    return get_balance().get(quote, 0.0)


def place_order(symbol: str, side: str, amount: float,
                order_type: str = "market", price: float | None = None) -> dict | None:
    if config.DRY_RUN:
        logger.info("[DRY RUN] %s %s %s %.6f @ %s", order_type.upper(), side.upper(), symbol, amount, price or "market")
        return {"id": "DRY_RUN", "side": side, "amount": amount, "symbol": symbol}
    ex = get_exchange()
    try:
        if order_type == "limit" and price:
            return ex.create_order(symbol, "limit", side, amount, price)
        return ex.create_order(symbol, "market", side, amount)
    except Exception as e:
        logger.error("Order failed %s %s %s: %s", side, symbol, amount, e)
        return None