"""
Webhook trade engine — turn a TradingView alert into a (paper or live) trade.

Pipeline for every inbound alert:
  1. Authenticate     — constant-time passphrase compare (reject 401 on mismatch)
  2. Parse + normalize — action, symbol (→ Coinbase BASE/USD), size
  3. Idempotency       — drop duplicate retries by alert id
  4. Risk gate         — symbol allowlist, pause, max positions, per-trade cap,
                         daily-loss kill switch, spot long-only
  5. Execute           — paper (simulated fill at live ticker) or live (real
                         Coinbase order via place_market_order_filled)
  6. Record + notify   — update the ledger, fire a non-blocking Telegram alert

Live orders are placed ONLY when the bot is fully armed (see _live_armed); in
every other case the trade is simulated so the dashboard/portfolio still work.
"""
from __future__ import annotations

import hmac
import logging
import threading

import config
from exchange.market_data import (fetch_ticker, place_market_order_filled,
                                  check_auth)
from notifications.notifier import send_telegram
from webhook import store

logger = logging.getLogger("webhook")

# Serialize the whole risk-check→execute→record critical section so two alerts
# arriving at once can't both pass the max-positions / cash checks (double-spend).
_trade_lock = threading.Lock()

# Resolved at init(): are we allowed to place REAL orders?
_live_armed = False
_auth_ok = False
_auth_msg = "not checked"


# ── Action vocabularies (case-insensitive) ────────────────────────────────────
_BUY_WORDS   = {"buy", "long", "enter_long", "enterlong", "buy_to_open", "bull"}
_SELL_WORDS  = {"sell", "exit", "close", "flat", "close_long", "closelong",
                "exit_long", "sell_to_close", "sell_to_open", "sellshort"}
_SHORT_WORDS = {"short", "sell_short", "enter_short", "entershort", "bear"}


# ── Init / mode ───────────────────────────────────────────────────────────────

def init() -> dict:
    """
    Load persisted state and decide whether live trading is armed. Called once at
    startup. Returns a status dict for logging / the dashboard health panel.
    """
    global _live_armed, _auth_ok, _auth_msg
    store.load()

    want_live = (not config.DRY_RUN) and config.WEBHOOK_LIVE_ENABLED
    has_pass  = bool(config.WEBHOOK_PASSPHRASE)

    if want_live:
        _auth_ok, _auth_msg = check_auth()
        _live_armed = _auth_ok and has_pass
    else:
        _auth_ok, _auth_msg = (False, "live disabled (paper mode)")
        _live_armed = False

    # Explain exactly why we are (not) live — no silent surprises.
    if want_live and not has_pass:
        logger.warning("WEBHOOK_LIVE_ENABLED set but WEBHOOK_PASSPHRASE is empty "
                       "→ refusing to arm live trading. Running PAPER.")
    if want_live and not _auth_ok:
        logger.warning("Live requested but API auth failed (%s) → running PAPER.", _auth_msg)

    return mode_status()


def mode_status() -> dict:
    return {
        "live":        _live_armed,
        "mode":        "LIVE" if _live_armed else "PAPER",
        "dry_run":     config.DRY_RUN,
        "live_enabled": config.WEBHOOK_LIVE_ENABLED,
        "auth_ok":     _auth_ok,
        "auth_msg":    _auth_msg,
        "passphrase_set": bool(config.WEBHOOK_PASSPHRASE),
    }


# ── Symbol normalization ──────────────────────────────────────────────────────

# Stablecoin / fiat quotes that map onto Coinbase's USD pairs.
_USD_QUOTES = {"USDT", "USDC", "USD", "USDU", "DAI"}
_KNOWN_QUOTES = ["USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH", "DAI"]


def normalize_symbol(raw: str) -> str | None:
    """
    Map a TradingView ticker to a Coinbase 'BASE/QUOTE' pair. Handles:
      'BTCUSD', 'BTCUSDT', 'BTC/USD', 'BTC-USD', 'COINBASE:BTCUSD', 'BINANCE:BTCUSDT'
    Stablecoin quotes (USDT/USDC/DAI) are folded to the configured QUOTE_CURRENCY
    (USD on Coinbase). Returns None if it can't be parsed.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    if ":" in s:                       # strip 'EXCHANGE:' prefix
        s = s.split(":", 1)[1]
    s = s.replace("-", "/").replace("_", "/")

    if "/" in s:
        base, quote = s.split("/", 1)
    else:
        quote = None
        for q in _KNOWN_QUOTES:        # longest-first match of a known quote suffix
            if s.endswith(q) and len(s) > len(q):
                base, quote = s[:-len(q)], q
                break
        if quote is None:
            base, quote = s, config.QUOTE_CURRENCY     # bare base → assume USD quote

    if quote in _USD_QUOTES:
        quote = config.QUOTE_CURRENCY                  # USDT/USDC/… → USD on Coinbase
    base, quote = base.strip(), quote.strip()
    if not base or not quote:
        return None
    return f"{base}/{quote}"


# ── Alert parsing ─────────────────────────────────────────────────────────────

def _f(v) -> float | None:
    try:
        return float(str(v).strip().replace("%", "").replace(",", ""))
    except Exception:
        return None


def parse_alert(payload: dict) -> dict:
    """
    Extract the trade intent from a (flexible) TradingView JSON payload.
    Returns {action, symbol, size, size_kind, price, strategy, alert_id, error}.
    """
    action_raw = str(payload.get("action") or payload.get("side") or
                     payload.get("order_action") or "").strip().lower()
    symbol_raw = (payload.get("symbol") or payload.get("ticker") or
                  payload.get("pair") or "")
    symbol = normalize_symbol(str(symbol_raw))

    if action_raw in _BUY_WORDS:
        action = "buy"
    elif action_raw in _SELL_WORDS:
        action = "sell"
    elif action_raw in _SHORT_WORDS:
        action = "short"
    else:
        action = ""

    # Size: percent of equity ('10%'), explicit base units (qty/contracts), or
    # USD notional (size/order_size/amount). Default handled at execution time.
    size, kind = None, "default"
    for k in ("qty", "quantity", "contracts", "base_qty"):
        if payload.get(k) is not None:
            size, kind = _f(payload[k]), "units"
            break
    if size is None:
        for k in ("order_size", "size", "usd", "amount", "notional"):
            v = payload.get(k)
            if v is not None:
                if isinstance(v, str) and v.strip().endswith("%"):
                    size, kind = _f(v), "percent"
                else:
                    size, kind = _f(v), "usd"
                break

    return {
        "action":   action,
        "action_raw": action_raw,
        "symbol":   symbol,
        "symbol_raw": str(symbol_raw),
        "size":     size,
        "size_kind": kind,
        "price":    _f(payload.get("price") or payload.get("close")),
        "strategy": str(payload.get("strategy") or payload.get("strategy_name")
                        or payload.get("comment") or "")[:60],
        "alert_id": str(payload.get("id") or payload.get("alert_id") or "")[:80],
    }


# ── Execution (paper or live) ─────────────────────────────────────────────────

def _live_price(symbol: str) -> float | None:
    tkr = fetch_ticker(symbol)
    px = (tkr.get("last") or tkr.get("close") or tkr.get("bid")) if tkr else None
    return float(px) if px else None


def _execute(symbol: str, side: str, qty: float) -> dict:
    """
    Place the order. Live → real Coinbase market order (fill-confirmed). Paper →
    simulated full fill at the live ticker. Returns {filled, average, order_id, live}.
    """
    if _live_armed:
        # DRY_RUN is guaranteed False here, so this places a REAL order.
        fill = place_market_order_filled(symbol, side, qty)
        return {
            "filled":   float(fill.get("filled") or 0.0),
            "average":  float(fill.get("average") or 0.0),
            "order_id": fill.get("id", ""),
            "partial":  fill.get("partial", False),
            "live":     True,
        }
    # Paper: simulate a full fill at the current market price.
    px = _live_price(symbol)
    if not px:
        return {"filled": 0.0, "average": 0.0, "order_id": "", "partial": False, "live": False}
    return {"filled": qty, "average": px, "order_id": "PAPER", "partial": False, "live": False}


def _fee(notional: float, live: bool) -> float:
    return abs(notional) * (config.FEE_RATE_PCT / 100.0)


# ── Main entry point ──────────────────────────────────────────────────────────

def process_alert(payload: dict, source_ip: str) -> tuple[int, dict]:
    """
    Validate, risk-check and execute a TradingView alert. Returns (http_status,
    response_dict). Every outcome is logged to the alert feed for the dashboard.
    """
    # 1. Authenticate (constant-time). Passphrase required if one is configured.
    expected = config.WEBHOOK_PASSPHRASE
    if expected:
        supplied = str(payload.get("passphrase") or payload.get("secret") or "")
        if not hmac.compare_digest(supplied, expected):
            store.log_alert(_redact(payload), source_ip, "rejected", "bad passphrase")
            logger.warning("Webhook rejected from %s: bad passphrase", source_ip)
            return 401, {"status": "rejected", "reason": "unauthorized"}

    intent = parse_alert(payload)
    sym, action = intent["symbol"], intent["action"]
    redacted = _redact(payload)

    def reject(msg: str, code: int = 400):
        store.log_alert(redacted, source_ip, "rejected", msg,
                        symbol=sym or intent["symbol_raw"], action=action or intent["action_raw"])
        logger.info("Alert rejected (%s): %s", sym or intent["symbol_raw"], msg)
        return code, {"status": "rejected", "reason": msg}

    # 2. Basic validity
    if not action:
        return reject(f"unknown action '{intent['action_raw']}'")
    if not sym:
        return reject(f"could not parse symbol '{intent['symbol_raw']}'")
    if action == "short" or (action == "sell" and not store.get_position(sym)
                             and not config.WEBHOOK_ALLOW_SHORT):
        # Coinbase spot is long-only. A short, or a sell with no long to close,
        # is a safe no-op rather than an error condition for the strategy.
        return reject("short/flat ignored — Coinbase spot is long-only", code=200)

    # 3. Idempotency
    if intent["alert_id"] and store.already_seen(intent["alert_id"]):
        store.log_alert(redacted, source_ip, "duplicate",
                        f"duplicate alert id {intent['alert_id']}", symbol=sym, action=action)
        return 200, {"status": "duplicate", "id": intent["alert_id"]}

    # 4. Pause toggle
    if store.is_paused() and action == "buy":
        return reject("bot paused — new entries blocked", code=200)

    # 5. Daily-loss kill switch (realized losses only; open positions keep running)
    cap = config.WEBHOOK_MAX_DAILY_LOSS_USD
    if cap > 0 and action == "buy" and store.daily_realized() <= -abs(cap):
        return reject(f"daily loss limit hit (${store.daily_realized():.2f}) — entries halted", code=200)

    # Everything past here mutates the ledger → take the trade lock.
    with _trade_lock:
        if action == "buy":
            return _do_buy(intent, source_ip, redacted)
        else:  # sell / close
            return _do_sell(intent, source_ip, redacted)


def _do_buy(intent: dict, source_ip: str, redacted: dict):
    sym = intent["symbol"]

    # Symbol allowlist (empty list = allow anything — discouraged when live).
    allow = config.WEBHOOK_SYMBOL_ALLOWLIST
    if allow and sym not in allow:
        store.log_alert(redacted, source_ip, "rejected", f"{sym} not in allowlist",
                        symbol=sym, action="buy")
        return 200, {"status": "rejected", "reason": f"{sym} not in symbol allowlist"}

    existing = store.get_position(sym)
    # Max open positions (adding to an existing one doesn't open a new slot).
    if not existing and store.open_position_count() >= config.WEBHOOK_MAX_OPEN_POSITIONS:
        store.log_alert(redacted, source_ip, "rejected", "max open positions reached",
                        symbol=sym, action="buy")
        return 200, {"status": "rejected", "reason": "max open positions reached"}

    price = _live_price(sym)
    if not price:
        store.log_alert(redacted, source_ip, "error", "no market price available",
                        symbol=sym, action="buy")
        return 503, {"status": "error", "reason": "no market price"}

    # Resolve target USD notional, then clamp to caps + available cash.
    notional = _resolve_buy_notional(intent, price)
    notional = min(notional, config.WEBHOOK_MAX_ORDER_USD, store.cash())
    if notional < config.WEBHOOK_MIN_ORDER_USD:
        store.log_alert(redacted, source_ip, "rejected",
                        f"order ${notional:.2f} below min/cash", symbol=sym, action="buy")
        return 200, {"status": "rejected", "reason": "insufficient cash or below min order size"}

    qty = notional / price
    fill = _execute(sym, "buy", qty)
    if fill["filled"] <= 0:
        store.log_alert(redacted, source_ip, "error", "order did not fill",
                        symbol=sym, action="buy")
        return 502, {"status": "error", "reason": "order did not fill"}

    avg = fill["average"]
    fee = _fee(fill["filled"] * avg, fill["live"])
    meta = {"strategy": intent["strategy"], "alert_id": intent["alert_id"],
            "order_id": fill["order_id"], "live": fill["live"]}
    pos = store.apply_buy(sym, fill["filled"], avg, fee, meta)
    if intent["alert_id"]:
        store.mark_seen(intent["alert_id"])

    msg = (f"{'LIVE' if fill['live'] else 'PAPER'} BUY {sym} "
           f"{fill['filled']:.6f} @ ${avg:,.4f} (${fill['filled']*avg:,.2f})")
    store.log_alert(redacted, source_ip, "executed", msg, symbol=sym, action="buy")
    logger.info(msg)
    _notify(f"📥 *{'LIVE' if fill['live'] else 'PAPER'} BUY* `{sym}`\n"
            f"Qty `{fill['filled']:.6f}` @ `${avg:,.4f}`\n"
            f"Notional `${fill['filled']*avg:,.2f}`"
            + (f"\nStrategy: `{intent['strategy']}`" if intent['strategy'] else ""))
    return 200, {"status": "executed", "action": "buy", "symbol": sym,
                 "qty": fill["filled"], "price": avg, "live": fill["live"],
                 "position": {"qty": pos["qty"], "entry": round(pos["entry"], 6)}}


def _do_sell(intent: dict, source_ip: str, redacted: dict):
    sym = intent["symbol"]
    pos = store.get_position(sym)
    if not pos:
        store.log_alert(redacted, source_ip, "rejected", "no open position to close",
                        symbol=sym, action="sell")
        return 200, {"status": "rejected", "reason": "no open position to close"}

    price = _live_price(sym)
    if not price:
        store.log_alert(redacted, source_ip, "error", "no market price available",
                        symbol=sym, action="sell")
        return 503, {"status": "error", "reason": "no market price"}

    # How much to sell: explicit size → that much; otherwise close the whole lot.
    qty = _resolve_sell_qty(intent, pos, price)
    qty = min(qty, pos["qty"])
    if qty <= 0:
        store.log_alert(redacted, source_ip, "rejected", "computed sell qty is zero",
                        symbol=sym, action="sell")
        return 200, {"status": "rejected", "reason": "computed sell qty is zero"}

    fill = _execute(sym, "sell", qty)
    if fill["filled"] <= 0:
        store.log_alert(redacted, source_ip, "error", "sell order did not fill",
                        symbol=sym, action="sell")
        return 502, {"status": "error", "reason": "sell order did not fill"}

    avg = fill["average"]
    fee = _fee(fill["filled"] * avg, fill["live"])
    reason = "signal_close" if qty >= pos["qty"] - 1e-12 else "signal_partial"
    meta = {"strategy": intent["strategy"], "alert_id": intent["alert_id"],
            "order_id": fill["order_id"], "live": fill["live"]}
    trade = store.apply_sell(sym, fill["filled"], avg, fee, reason, meta)
    if intent["alert_id"]:
        store.mark_seen(intent["alert_id"])

    pnl = trade.get("pnl", 0.0)
    emoji = "🟢" if pnl >= 0 else "🔴"
    msg = (f"{'LIVE' if fill['live'] else 'PAPER'} SELL {sym} "
           f"{fill['filled']:.6f} @ ${avg:,.4f} → P&L ${pnl:,.2f}")
    store.log_alert(redacted, source_ip, "executed", msg, symbol=sym, action="sell")
    logger.info(msg)
    _notify(f"📤 *{'LIVE' if fill['live'] else 'PAPER'} SELL* `{sym}`\n"
            f"Qty `{fill['filled']:.6f}` @ `${avg:,.4f}`\n"
            f"{emoji} P&L `${pnl:,.2f}` (`{trade.get('pnl_pct',0):+.2f}%`)")
    return 200, {"status": "executed", "action": "sell", "symbol": sym,
                 "qty": fill["filled"], "price": avg, "pnl": pnl, "live": fill["live"]}


# ── Sizing helpers ────────────────────────────────────────────────────────────

def _resolve_buy_notional(intent: dict, price: float) -> float:
    """Target USD to spend on a buy, before cap/cash clamping."""
    size, kind = intent["size"], intent["size_kind"]
    if size is None or size <= 0:
        return config.WEBHOOK_DEFAULT_ORDER_USD
    if kind == "usd":
        return size
    if kind == "units":
        return size * price
    if kind == "percent":
        # Percent of current equity (cash + holdings), via a no-price-arg mark.
        equity = store.mark_to_market({})["equity"]
        return equity * (size / 100.0)
    return config.WEBHOOK_DEFAULT_ORDER_USD


def _resolve_sell_qty(intent: dict, pos: dict, price: float) -> float:
    """Base units to sell. No explicit size → close the entire position."""
    size, kind = intent["size"], intent["size_kind"]
    if size is None or size <= 0 or kind == "default":
        return pos["qty"]
    if kind == "units":
        return size
    if kind == "usd":
        return size / price
    if kind == "percent":
        return pos["qty"] * (size / 100.0)
    return pos["qty"]


# ── Misc ──────────────────────────────────────────────────────────────────────

def _redact(payload: dict) -> dict:
    """Copy of the payload with the passphrase masked — safe to store/display."""
    out = dict(payload) if isinstance(payload, dict) else {"_raw": str(payload)}
    for k in ("passphrase", "secret"):
        if k in out:
            out[k] = "***"
    return out


def _notify(msg: str) -> None:
    try:
        send_telegram(msg)
    except Exception:
        pass


def flatten_all(reason: str = "manual_flatten") -> dict:
    """Close every open position at market (dashboard 'flatten all' button)."""
    closed = []
    with _trade_lock:
        for sym in list(store.snapshot()["positions"].keys()):
            pos = store.get_position(sym)
            if not pos:
                continue
            price = _live_price(sym)
            if not price:
                continue
            fill = _execute(sym, "sell", pos["qty"])
            if fill["filled"] <= 0:
                continue
            avg = fill["average"]
            fee = _fee(fill["filled"] * avg, fill["live"])
            meta = {"strategy": pos.get("strategy", ""), "order_id": fill["order_id"],
                    "live": fill["live"]}
            trade = store.apply_sell(sym, fill["filled"], avg, fee, reason, meta)
            closed.append(trade)
    if closed:
        _notify(f"✋ *Flatten all* — closed {len(closed)} position(s), "
                f"net P&L `${sum(t['pnl'] for t in closed):,.2f}`")
    return {"closed": len(closed), "trades": closed}
