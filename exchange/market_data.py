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


def place_market_order_filled(symbol: str, side: str, amount: float,
                              poll_attempts: int = 5,
                              poll_delay: float = 0.6) -> dict:
    """
    Place a MARKET order and confirm the ACTUAL fill.

    Market orders can partially fill (thin book) or fill at a different average
    price than the last trade. Sizing protective stops or computing P&L on the
    *requested* amount instead of the *filled* amount is a real bug — so this
    returns what truly executed.

    Returns a normalised dict:
      {"id", "filled", "average", "status", "partial", "requested"}
        filled    : actual base quantity that executed
        average   : volume-weighted average fill price
        status    : "closed" (fully/partially done) | "rejected" | "none"
        partial   : True if filled < requested (within tolerance)
        requested : the amount we asked for

    DRY_RUN: simulates a full fill at the current ticker price.
    """
    if config.DRY_RUN:
        tkr   = fetch_ticker(symbol)
        price = (tkr.get("last") or tkr.get("close") or 0.0) if tkr else 0.0
        logger.info("[DRY RUN] MARKET %s %s %.6f @ ~%.6f (simulated full fill)",
                    side.upper(), symbol, amount, price)
        return {"id": "DRY_RUN", "filled": amount, "average": price,
                "status": "closed", "partial": False, "requested": amount}

    ex = get_exchange()
    try:
        order = ex.create_order(symbol, "market", side, amount)
    except Exception as e:
        logger.error("Market order REJECTED %s %s %.6f: %s", side, symbol, amount, e)
        return {"id": "", "filled": 0.0, "average": 0.0,
                "status": "rejected", "partial": False, "requested": amount}

    order_id = order.get("id", "")
    filled   = float(order.get("filled") or 0.0)
    average  = float(order.get("average") or order.get("price") or 0.0)
    status   = order.get("status", "")

    # Market orders usually fill instantly, but poll a few times to get the
    # final filled/average if the create response was incomplete.
    attempts = 0
    while order_id and status != "closed" and filled < amount and attempts < poll_attempts:
        time.sleep(poll_delay)
        attempts += 1
        fetched = get_order_status(order_id, symbol)
        if not fetched:
            continue
        filled  = float(fetched.get("filled")  or filled)
        average = float(fetched.get("average") or fetched.get("price") or average)
        status  = fetched.get("status", status)
        if status in ("closed", "canceled"):
            break

    tol     = amount * 0.995          # treat ≥99.5% as effectively full
    partial = 0.0 < filled < tol

    if filled <= 0:
        logger.error("Market order %s %s did not fill (status=%s)", side, symbol, status)
        return {"id": order_id, "filled": 0.0, "average": average,
                "status": "none", "partial": False, "requested": amount}

    if partial:
        logger.warning("PARTIAL FILL %s %s — filled %.6f / requested %.6f @ avg %.6f",
                       side.upper(), symbol, filled, amount, average)

    return {"id": order_id, "filled": filled, "average": average,
            "status": "closed", "partial": partial, "requested": amount}


# ── Protective (exchange-side) orders ─────────────────────────────────────────
# Professional-grade: the exchange enforces stop/target instantly, 24/7, even if
# the bot is slow, sleeping or crashed. Coinbase has no native OCO, so the bot
# implements manual OCO (cancel the sibling order when one fills).

def place_stop_limit_order(symbol: str, side: str, amount: float,
                           stop_price: float, limit_price: float) -> dict | None:
    """
    Place a STOP-LIMIT order on the exchange (the protective stop loss).

    side        : "sell" to protect a long, "buy" to protect a short
    stop_price  : trigger price — when market reaches this, the limit order activates
    limit_price : the worst price you'll accept once triggered (set slightly
                  beyond stop_price to improve fill odds in a fast move)
    """
    if config.DRY_RUN:
        logger.info("[DRY RUN] STOP-LIMIT %s %s %.6f trigger=%.6f limit=%.6f",
                    side.upper(), symbol, amount, stop_price, limit_price)
        return {"id": "DRY_RUN_STOP", "type": "stop", "symbol": symbol,
                "side": side, "amount": amount, "stopPrice": stop_price}
    ex = get_exchange()
    try:
        # CCXT Coinbase: stop is signalled via the stopPrice param on a limit order
        order = ex.create_order(
            symbol, "limit", side, amount, limit_price,
            {"stopPrice": stop_price, "stop_direction":
                "STOP_DIRECTION_STOP_DOWN" if side == "sell" else "STOP_DIRECTION_STOP_UP"},
        )
        logger.info("Exchange STOP-LIMIT placed: %s %s trigger=%.6f id=%s",
                    side.upper(), symbol, stop_price, order.get("id"))
        return order
    except Exception as e:
        logger.error("Stop-limit order failed %s %s: %s", side, symbol, e)
        return None


def place_take_profit_order(symbol: str, side: str, amount: float,
                            price: float) -> dict | None:
    """
    Place a LIMIT order on the exchange as the take-profit target.
    side : "sell" to take profit on a long, "buy" to take profit on a short.
    """
    if config.DRY_RUN:
        logger.info("[DRY RUN] TAKE-PROFIT LIMIT %s %s %.6f @ %.6f",
                    side.upper(), symbol, amount, price)
        return {"id": "DRY_RUN_TP", "type": "limit", "symbol": symbol,
                "side": side, "amount": amount, "price": price}
    ex = get_exchange()
    try:
        order = ex.create_order(symbol, "limit", side, amount, price)
        logger.info("Exchange TAKE-PROFIT placed: %s %s @ %.6f id=%s",
                    side.upper(), symbol, price, order.get("id"))
        return order
    except Exception as e:
        logger.error("Take-profit order failed %s %s: %s", side, symbol, e)
        return None


def cancel_order(order_id: str, symbol: str) -> bool:
    """Cancel an open order. Returns True on success."""
    if config.DRY_RUN or not order_id or order_id.startswith("DRY_RUN"):
        return True
    ex = get_exchange()
    try:
        ex.cancel_order(order_id, symbol)
        logger.info("Cancelled order %s on %s", order_id, symbol)
        return True
    except Exception as e:
        logger.warning("Cancel order %s on %s failed: %s", order_id, symbol, e)
        return False


def get_order_status(order_id: str, symbol: str) -> dict | None:
    """
    Fetch an order's current status. Returns the order dict, or None on error.
    Key field: order['status'] ∈ {'open','closed','canceled'}; 'closed' = filled.
    """
    if config.DRY_RUN or not order_id or order_id.startswith("DRY_RUN"):
        return None
    ex = get_exchange()
    try:
        return ex.fetch_order(order_id, symbol)
    except Exception as e:
        logger.warning("Fetch order %s on %s failed: %s", order_id, symbol, e)
        return None