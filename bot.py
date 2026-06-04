"""
Main trading bot loop.

Flow each cycle:
  1. For each symbol in WATCHLIST:
     a. Fetch 1h + 4h candles
     b. Enrich with indicators
     c. Score multi-strategy signal
     d. If strong enough signal and no open position → check risk → open trade
  2. For each open position:
     a. Fetch latest price
     b. Check stop/target/trailing → close if triggered
  3. Push state to web dashboard
  4. Sleep until next cycle

Web dashboard runs at http://localhost:8081
"""

import time
import threading
import logging
import os
from datetime import datetime, timezone
from dataclasses import asdict
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

import config
import ui.state as st
import exchange.universe as universe
from exchange.market_data import (fetch_ohlcv, fetch_ticker, place_order, check_auth,
                                   place_stop_limit_order, place_take_profit_order,
                                   cancel_order, get_order_status,
                                   place_market_order_filled)
from core.indicators import enrich
from core.strategies import analyse, SignalResult
from risk.risk_manager import RiskManager
from risk.circuit_breaker import get_breaker
from notifications.notifier import (notify_signal, notify_trade,
                       notify_trade_open, notify_trade_close,
                       notify_bot_started, notify_bot_stopped,
                       notify_portfolio_summary, notify_error)
from ui.dashboard import start_server, is_paused

colorama_init(autoreset=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log"),
    ],
)
logger = logging.getLogger("bot")

risk_mgr = RiskManager()
breaker  = get_breaker()             # circuit breaker / kill switch
_positions_lock = threading.Lock()   # guards position close/monitor against races


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color_score(score: float) -> str:
    if score >= config.STRONG_SIGNAL_SCORE:
        return Fore.GREEN + f"{score:.1f}" + Style.RESET_ALL
    elif score >= config.MIN_SIGNAL_SCORE:
        return Fore.YELLOW + f"{score:.1f}" + Style.RESET_ALL
    return Fore.WHITE + f"{score:.1f}" + Style.RESET_ALL


def _color_dir(direction: str) -> str:
    if direction == "long":
        return Fore.GREEN + "LONG" + Style.RESET_ALL
    elif direction == "short":
        return Fore.RED + "SHORT" + Style.RESET_ALL
    return Fore.WHITE + "—" + Style.RESET_ALL


def _signal_to_dict(sig: SignalResult) -> dict:
    return {
        "symbol":          sig.symbol,
        "direction":       sig.direction,
        "score":           sig.score,
        "entry_price":     sig.entry_price,
        "stop_loss":       sig.stop_loss,
        "take_profit":     sig.take_profit,
        "risk_reward":     sig.risk_reward,
        "sentiment":       sig.sentiment,
        "components":      sig.components,
        "atr":             sig.atr,
        "candle_patterns": sig.candle_patterns,
        "ml_prediction":   sig.ml_prediction,
        "ml_confidence":   sig.ml_confidence,
    }


# ── ML model training ─────────────────────────────────────────────────────────

def _train_ml_models(symbols: list[str]) -> None:
    """
    Train ML signal predictor + regime classifier for each symbol.
    Runs in background threads — fully non-blocking.
    Fetches a long history (1h candles) for training.
    """
    def _worker():
        try:
            from ml.signal_predictor import get_predictor
            from ml.regime_classifier import get_regime_classifier
            from ml.extrema_predictor import get_extrema_predictor
            predictor  = get_predictor()
            regime_clf = get_regime_classifier()
            extrema    = get_extrema_predictor()

            regime_trained = False
            for sym in symbols:
                try:
                    df = fetch_ohlcv(sym, config.TF_PRIMARY, limit=1000)
                    if df.empty or len(df) < 250:
                        continue
                    df = enrich(df)

                    # Train per-symbol signal predictor + extrema predictor
                    predictor.train(sym, df)
                    extrema.train(sym, df)

                    # Train the regime classifier once (on BTC — the market leader)
                    if not regime_trained and sym == "BTC/USD":
                        if regime_clf.train(df):
                            regime_clf.save()
                            regime_trained = True
                except Exception as e:
                    logger.warning("ML training error for %s: %s", sym, e)

            # If BTC wasn't in the list, train regime on the first available symbol
            if not regime_trained:
                for sym in symbols:
                    df = fetch_ohlcv(sym, config.TF_PRIMARY, limit=1000)
                    if not df.empty and len(df) >= 250:
                        df = enrich(df)
                        if regime_clf.train(df):
                            regime_clf.save()
                        break

            logger.info("ML model training pass complete (%d symbols)", len(symbols))
        except Exception as e:
            logger.error("ML training worker failed: %s", e)

    t = threading.Thread(target=_worker, daemon=True, name="ml-training")
    t.start()
    logger.info("ML training started in background for %d symbols", len(symbols))


# ── Dynamic symbol ranking ────────────────────────────────────────────────────

def _rank_symbols(watchlist: list[str]) -> list[str]:
    """
    Rank symbols by absolute 6h 14-bar ROC and return the top 60%.
    Coins with no data are kept so they can still be scanned.
    """
    scores: list[tuple[float, str]] = []
    for sym in watchlist:
        try:
            df = fetch_ohlcv(sym, config.TF_TREND, limit=20)
            if df.empty or len(df) < 15:
                scores.append((0.0, sym))
                continue
            roc = (df["close"].iloc[-1] - df["close"].iloc[-14]) / df["close"].iloc[-14] * 100
            scores.append((abs(float(roc)), sym))   # rank by momentum magnitude
        except Exception:
            scores.append((0.0, sym))
    scores.sort(reverse=True)
    keep = max(int(len(scores) * 0.6), 5)
    selected = [s for _, s in scores[:keep]]
    logger.info("Active symbols (%d/%d): %s", len(selected), len(watchlist), ", ".join(selected))
    return selected


# ── Signal scanning ───────────────────────────────────────────────────────────

def scan_symbol(symbol: str) -> SignalResult | None:
    try:
        df1h = fetch_ohlcv(symbol, config.TF_PRIMARY)
        df4h = fetch_ohlcv(symbol, config.TF_TREND)
        if df1h.empty:
            logger.warning("No 1h data for %s", symbol)
            return None
        df1h = enrich(df1h)
        df4h = enrich(df4h) if not df4h.empty else None
        return analyse(symbol, df1h, df4h, include_sentiment=True)
    except Exception as e:
        logger.error("Error scanning %s: %s", symbol, e)
        st.add_error(f"scan {symbol}: {e}")
        return None


# ── Trade execution ───────────────────────────────────────────────────────────

def _btc_crash_change() -> float | None:
    """
    BTC % change over the crash lookback window (1h candles).
    Used by the circuit breaker for correlation protection.
    Returns None if data unavailable.
    """
    try:
        bars = max(2, int(getattr(config, "CB_CRASH_LOOKBACK_HOURS", 4)) + 1)
        df = fetch_ohlcv("BTC/USD", "1h", limit=bars)
        if df.empty or len(df) < 2:
            return None
        first = df["close"].iloc[0]
        last  = df["close"].iloc[-1]
        return (last - first) / first * 100
    except Exception:
        return None


def _update_circuit_breaker(signals: list[SignalResult]) -> None:
    """Feed the breaker current equity, unrealised P&L and BTC crash signal."""
    try:
        summary    = risk_mgr.summary()
        equity     = summary["capital_usdt"]
        unrealised = sum(p.get("pnl", 0) for p in summary["open_details"])
        # Reuse BTC change from the scanned signals if present, else fetch
        btc_change = None
        for s in signals:
            if s.symbol == "BTC/USD":
                # ROC component already on the candle; fall back to dedicated fetch
                break
        btc_change = _btc_crash_change()
        breaker.update(equity, unrealised, btc_change)
        st.set_circuit_breaker(breaker.status())
    except Exception as e:
        logger.debug("Circuit breaker update skipped: %s", e)


def _daily_profit_locked() -> bool:
    """
    True if realised NET profit for the current UTC day has reached
    DAILY_PROFIT_TARGET_USD — in which case we stop opening new trades and
    lock in the day's gains. 0 = feature disabled.
    """
    target = getattr(config, "DAILY_PROFIT_TARGET_USD", 0.0)
    if target <= 0:
        return False
    today = datetime.now(timezone.utc).date().isoformat()
    day_net = 0.0
    for t in st.get()["trade_history"]:
        if t.get("action") == "close" and (t.get("time", "")[:10] == today):
            day_net += (t.get("pnl") or 0.0)
    return day_net >= target


def try_open_trade(signal: SignalResult) -> None:
    if signal.direction not in ("long", "short"):
        return
    if signal.score < config.MIN_SIGNAL_SCORE:
        return
    if signal.risk_reward < 1.5:
        logger.info("Skipping %s — RR %.2f < 1.5", signal.symbol, signal.risk_reward)
        return

    # Circuit breaker gate — no NEW entries while halted
    cb_ok, cb_reason = breaker.can_trade()
    if not cb_ok:
        logger.warning("Circuit breaker blocking entry on %s — %s", signal.symbol, cb_reason)
        return

    can_open, reason = risk_mgr.can_open(signal.symbol)
    if not can_open:
        logger.info("Cannot open %s: %s", signal.symbol, reason)
        return

    # Daily profit lock — once up enough for the day, stop opening new risk.
    if _daily_profit_locked():
        logger.info("Daily profit target reached — not opening %s (locking gains)",
                    signal.symbol)
        return

    entry = signal.entry_price
    stop  = signal.stop_loss
    tp    = signal.take_profit
    usdt  = risk_mgr.position_size_usdt(entry, stop, signal.score, signal.ml_confidence)

    if usdt < 5:
        logger.info("Position too small for %s (%.2f USDT)", signal.symbol, usdt)
        return

    qty  = usdt / entry
    side = "buy" if signal.direction == "long" else "sell"

    # ── Profit-floor gate ─────────────────────────────────────────────────────
    # Only take the trade if hitting its take-profit would net ≥ MIN_NET_PROFIT_USD
    # AFTER round-trip fees. A 'winner' that can't clear fees is a real loser.
    gross_at_tp = abs(tp - entry) * qty
    fees        = risk_mgr.round_trip_fees(entry, tp, qty)
    net_at_tp   = gross_at_tp - fees
    min_net     = getattr(config, "MIN_NET_PROFIT_USD", 1.0)
    if net_at_tp < min_net:
        logger.info("Skip %s — target nets only $%.2f after $%.2f fees (need ≥ $%.2f). "
                    "Move too small to be worth the fees.",
                    signal.symbol, net_at_tp, fees, min_net)
        return

    # Place the entry order and confirm the ACTUAL fill (handles partial fills
    # and price slippage). Sizing stops or P&L on requested-but-unfilled qty is a bug.
    fill = place_market_order_filled(signal.symbol, side, qty)
    if fill["filled"] <= 0:
        logger.warning("Entry not filled for %s (status=%s) — no position opened",
                       signal.symbol, fill["status"])
        return

    # Use what ACTUALLY executed: real filled qty + real average price.
    filled_qty  = fill["filled"]
    fill_price  = fill["average"] or entry
    entry       = fill_price                  # P&L now measured from the true fill

    # Recompute stop/target from the actual fill price so the R:R holds.
    atr = signal.atr
    if signal.direction == "long":
        stop = round(fill_price - config.ATR_STOP_MULTIPLIER   * atr, 6)
        tp   = round(fill_price + config.ATR_TARGET_MULTIPLIER * atr, 6)
    else:
        stop = round(fill_price + config.ATR_STOP_MULTIPLIER   * atr, 6)
        tp   = round(fill_price - config.ATR_TARGET_MULTIPLIER * atr, 6)

    if fill["partial"]:
        logger.warning("Partial entry on %s: filled %.6f / %.6f — sizing stops to filled qty",
                       signal.symbol, filled_qty, qty)
        try:
            notify_error(f"⚠️ Partial fill on {signal.symbol}: got "
                         f"{filled_qty:.4f}/{qty:.4f}. Position + stops sized to actual fill.")
        except Exception:
            pass

    qty = filled_qty   # everything downstream uses the real filled quantity

    risk_mgr.open_position(
        symbol=signal.symbol, side=signal.direction,
        entry=entry, qty=qty, stop=stop, take_profit=tp,
        atr=signal.atr, order_id=fill.get("id", ""),
    )

    # ── Place exchange-side protective orders (live mode) ─────────────────────
    # Professional-grade: the exchange enforces the stop/target instantly, even
    # if the bot is slow or down. In DRY_RUN these are simulated (no real order).
    # Sized to the ACTUAL filled quantity so the stop can't over/under-sell.
    _place_protective_orders(signal.symbol, signal.direction, qty, stop, tp)

    trade_record = {
        "time":   datetime.now(timezone.utc).isoformat(),
        "action": "open",
        "symbol": signal.symbol,
        "side":   signal.direction,
        "entry":  entry,
        "stop":   stop,
        "tp":     tp,
        "score":  signal.score,
        "pnl":    None,
        "reason": None,
    }
    st.add_trade(trade_record)
    st.set_capital(risk_mgr.summary()["capital_usdt"])

    risk_usd = abs(entry - stop) * qty
    notify_trade_open(
        symbol=signal.symbol,
        direction=signal.direction,
        entry=entry,
        qty=qty,
        stop=stop,
        take_profit=tp,
        risk_usd=risk_usd,
        score=signal.score,
        candle_patterns=signal.candle_patterns,
        sentiment_label=signal.sentiment.get("label", "neutral"),
    )
    logger.info("TRADE OPENED: %s %s score=%.1f  entry=%.6f  SL=%.6f  TP=%.6f  qty=%.6f",
                signal.direction.upper(), signal.symbol, signal.score, entry, stop, tp, qty)


def _place_protective_orders(symbol: str, direction: str, qty: float,
                             stop: float, take_profit: float) -> None:
    """
    Place exchange-side stop-limit (stop loss) + limit (take profit) orders.
    In DRY_RUN these are simulated. On live, if the stop is rejected we fall
    back to poll-based monitoring and alert — we never leave a position
    unprotected silently.
    """
    # Only place real exchange-side orders if enabled (live). DRY_RUN simulates.
    if not getattr(config, "USE_EXCHANGE_STOPS", True):
        return

    # Protective side is opposite the entry: long → sell to exit, short → buy
    exit_side = "sell" if direction == "long" else "buy"

    # Limit price for the stop sits slightly beyond the trigger to improve fill
    # odds in a fast move (0.3% buffer).
    if direction == "long":
        stop_limit_price = stop * 0.997
    else:
        stop_limit_price = stop * 1.003

    stop_order = place_stop_limit_order(symbol, exit_side, qty, stop, stop_limit_price)
    tp_order   = place_take_profit_order(symbol, exit_side, qty, take_profit)

    stop_id = stop_order.get("id", "") if stop_order else ""
    tp_id   = tp_order.get("id", "")   if tp_order   else ""

    risk_mgr.attach_protective_orders(symbol, stop_id, tp_id)

    if not config.DRY_RUN and not stop_id:
        # Live mode but the protective stop failed — this is a risk event.
        logger.error("⚠️ Exchange stop order FAILED for %s — falling back to "
                     "poll-based monitoring. Position is less protected.", symbol)
        try:
            notify_error(f"⚠️ Stop order failed on {symbol} — using poll-based "
                         f"monitoring. Watch this position.")
        except Exception:
            pass


def _close_position(symbol: str, pos, exit_price: float, reason: str,
                    already_filled_on_exchange: bool = False) -> None:
    """
    Single shared close routine: cancel sibling protective order (manual OCO),
    place the exit market order if needed, record the trade, notify, persist.

    already_filled_on_exchange=True means a protective order already executed on
    the exchange (so we must NOT place another exit order — just reconcile).
    """
    # Manual OCO: cancel whichever protective order did NOT fill
    if pos.protected and not config.DRY_RUN:
        if reason in ("stop_loss", "trailing"):
            cancel_order(pos.tp_order_id, symbol)      # stop hit → cancel TP
        elif reason == "take_profit":
            cancel_order(pos.stop_order_id, symbol)    # TP hit → cancel stop
        else:
            cancel_order(pos.stop_order_id, symbol)
            cancel_order(pos.tp_order_id, symbol)

    # Place the exit order only if the exchange hasn't already filled one
    if not already_filled_on_exchange:
        exit_side = "sell" if pos.side == "long" else "buy"
        fill = place_market_order_filled(symbol, exit_side, pos.qty)
        # Use the real exit fill price for accurate P&L
        if fill["average"]:
            exit_price = fill["average"]
        # Partial exit: retry the remaining quantity once so we fully flatten.
        remaining = pos.qty - fill["filled"]
        if not config.DRY_RUN and remaining > pos.qty * 0.005:
            logger.warning("Partial exit on %s — %.6f left, retrying", symbol, remaining)
            retry = place_market_order_filled(symbol, exit_side, remaining)
            if pos.qty - fill["filled"] - retry["filled"] > pos.qty * 0.01:
                try:
                    notify_error(f"⚠️ {symbol} did not fully close — "
                                 f"{remaining:.4f} may remain. Check the exchange.")
                except Exception:
                    pass

    pnl = risk_mgr.close_position(symbol, exit_price, reason)
    breaker.record_trade_result(pnl)   # feed consecutive-loss tracking

    trade_record = {
        "time":   datetime.now(timezone.utc).isoformat(),
        "action": "close",
        "symbol": symbol,
        "side":   pos.side,
        "entry":  pos.entry_price,
        "price":  exit_price,
        "pnl":    round(pnl, 4),
        "reason": reason,
        "score":  None,
    }
    st.add_trade(trade_record)
    st.set_capital(risk_mgr.summary()["capital_usdt"])

    hold_mins = None
    try:
        hold_mins = int((datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 60)
    except Exception:
        pass

    notify_trade_close(
        symbol=symbol, direction=pos.side, entry=pos.entry_price,
        exit_price=exit_price, qty=pos.qty, pnl=pnl, reason=reason,
        hold_duration_mins=hold_mins,
    )
    logger.info("TRADE CLOSED: %s @ %.6f  Reason=%s  PnL=%.4f",
                symbol, exit_price, reason, pnl)


def check_open_positions() -> None:
    """
    Monitor open positions. Runs frequently (fast loop).

    Live + protected: check whether the exchange-side stop/TP order filled.
                      If so, reconcile (manual OCO) — no extra exit order.
    Otherwise (dry-run, or live fallback): poll the ticker and enforce
                      stop/target/trailing in-code.

    Guarded by a lock so the entry loop and fast monitor never close the same
    position twice.
    """
    if not _positions_lock.acquire(blocking=False):
        return   # another thread is already checking — skip this tick
    try:
        _check_open_positions_impl()
    finally:
        _positions_lock.release()


def _check_open_positions_impl() -> None:
    for symbol, pos in list(risk_mgr.positions.items()):
        if pos.status != "open":
            continue

        # ── Live + protected: reconcile with exchange fills ───────────────────
        if pos.protected and not config.DRY_RUN:
            filled_reason = None
            filled_price  = None
            stop_o = get_order_status(pos.stop_order_id, symbol)
            if stop_o and stop_o.get("status") == "closed":
                filled_reason = "stop_loss"
                filled_price  = stop_o.get("average") or stop_o.get("price") or pos.stop_loss
            else:
                tp_o = get_order_status(pos.tp_order_id, symbol)
                if tp_o and tp_o.get("status") == "closed":
                    filled_reason = "take_profit"
                    filled_price  = tp_o.get("average") or tp_o.get("price") or pos.take_profit

            if filled_reason:
                _close_position(symbol, pos, float(filled_price), filled_reason,
                                already_filled_on_exchange=True)
                continue

            ticker = fetch_ticker(symbol)
            cur = ticker.get("last") or ticker.get("close", pos.entry_price) if ticker else None

            # ── Gap-through safety net ────────────────────────────────────────
            # In a violent move, price can blow past BOTH the stop trigger and the
            # stop-limit's limit price without filling (no buyers at the limit).
            # The stop order sits open while the position bleeds. Detect that —
            # price gapped beyond the stop by a buffer but the stop hasn't filled —
            # and force a MARKET exit (always fills; accepts slippage to guarantee out).
            if cur is not None:
                gap = getattr(config, "STOP_GAP_BUFFER_PCT", 0.5) / 100.0
                gapped = (
                    (pos.side == "long"  and cur <= pos.stop_loss * (1 - gap)) or
                    (pos.side == "short" and cur >= pos.stop_loss * (1 + gap))
                )
                if gapped:
                    logger.error("⚠️ GAP-THROUGH on %s — price %.6f blew past stop %.6f "
                                 "but stop-limit unfilled. Forcing MARKET exit.",
                                 symbol, cur, pos.stop_loss)
                    try:
                        notify_error(f"⚠️ {symbol} gapped through its stop — forcing "
                                     f"market exit to guarantee the position closes.")
                    except Exception:
                        pass
                    # _close_position cancels both protective orders, then market-exits.
                    _close_position(symbol, pos, cur, "stop_gap")
                    continue

            # Trailing stop: if price moved enough to raise the stop, cancel the
            # old exchange stop and re-place it at the new (tighter) level.
            if ticker:
                old_stop = pos.stop_loss
                risk_mgr.update_position(symbol, cur)   # updates trailing internally
                if abs(pos.stop_loss - old_stop) > 1e-9:
                    exit_side = "sell" if pos.side == "long" else "buy"
                    cancel_order(pos.stop_order_id, symbol)
                    lim = pos.stop_loss * (0.997 if pos.side == "long" else 1.003)
                    new_stop_o = place_stop_limit_order(symbol, exit_side, pos.qty,
                                                        pos.stop_loss, lim)
                    if new_stop_o:
                        risk_mgr.attach_protective_orders(
                            symbol, new_stop_o.get("id", ""), pos.tp_order_id)
                        logger.info("Trailing stop re-placed for %s → %.6f",
                                    symbol, pos.stop_loss)
            continue

        # ── Dry-run or live fallback: in-code stop/target/trailing ────────────
        ticker = fetch_ticker(symbol)
        if not ticker:
            continue
        current_price = ticker.get("last") or ticker.get("close", pos.entry_price)
        action_info   = risk_mgr.update_position(symbol, current_price)

        if action_info["action"] == "close":
            _close_position(symbol, pos, current_price, action_info["reason"])

    # Push updated open positions to state
    summary = risk_mgr.summary()
    st.set_open_positions(summary["open_details"])


def _position_monitor_loop() -> None:
    """
    Dedicated fast loop for position safety — runs independently of the slower
    entry-scan loop so stops/targets are checked every POSITION_CHECK_SECONDS
    instead of once per full scan. This is the professional-grade exit guard.
    """
    interval = getattr(config, "POSITION_CHECK_SECONDS", 45)
    while True:
        try:
            if not is_paused() and any(p.status == "open" for p in risk_mgr.positions.values()):
                check_open_positions()
        except Exception as e:
            logger.error("Position monitor error: %s", e)
        time.sleep(interval)


# ── Terminal dashboard (secondary output) ─────────────────────────────────────

def print_terminal(signals: list[SignalResult]) -> None:
    os.system("clear" if os.name != "nt" else "cls")
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = risk_mgr.summary()
    print(f"\n{'='*70}")
    print(f"  CRYPTO TRADING BOT  |  {now}  |  {'DRY RUN' if config.DRY_RUN else 'LIVE'}")
    print(f"  Dashboard → http://localhost:8081")
    print(f"{'='*70}")
    print(f"  Capital: ${summary['capital_usdt']:.2f}  |  "
          f"Open: {summary['open_positions']}  |  "
          f"Closed: {summary['closed_positions']}")
    print(f"{'='*70}\n")

    rows = [[sig.symbol, _color_dir(sig.direction), _color_score(sig.score),
             f"{sig.entry_price:.4f}",
             f"{sig.stop_loss:.4f}"  if sig.stop_loss  else "—",
             f"{sig.take_profit:.4f}" if sig.take_profit else "—",
             f"{sig.risk_reward}"    if sig.risk_reward  else "—",
             sig.sentiment.get("label", "—")]
            for sig in sorted(signals, key=lambda s: s.score, reverse=True)]
    print(tabulate(rows, headers=["Symbol","Dir","Score","Entry","Stop","Target","RR","Sentiment"],
                   tablefmt="rounded_outline"))
    print(f"\n  Next scan in {config.SCAN_INTERVAL_SECONDS}s  |  Ctrl+C to stop\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info("Starting trading bot  |  exchange=%s  testnet=%s  dry_run=%s",
                config.EXCHANGE, config.TESTNET, config.DRY_RUN)

    # ── API key health check ──────────────────────────────────────────────────
    auth_ok, auth_msg = check_auth()
    try:
        st.set_auth_status(auth_ok, auth_msg)
    except Exception:
        pass
    if auth_ok:
        logger.info("✅ API key check: %s", auth_msg)
    else:
        banner = "═" * 68
        if config.DRY_RUN:
            # Paper trading — auth not required, just inform
            logger.warning("%s", banner)
            logger.warning("⚠️  API KEY NOT WORKING — but you are in DRY_RUN (paper) mode.")
            logger.warning("    Reason: %s", auth_msg)
            logger.warning("    Paper trading works fine. Fix the key BEFORE going live.")
            logger.warning("%s", banner)
            try:
                notify_error(f"⚠️ API key not working ({auth_msg[:80]}). "
                             f"OK for paper trading — fix before going live.")
            except Exception:
                pass
        else:
            # LIVE mode with a broken key — refuse to start, this is dangerous
            logger.error("%s", banner)
            logger.error("🛑 LIVE MODE but API KEY IS NOT WORKING — refusing to start.")
            logger.error("    Reason: %s", auth_msg)
            logger.error("    Real orders would fail. Fix the key or set DRY_RUN=True.")
            logger.error("    Get a new key: coinbase.com → Settings → API (View + Trade).")
            logger.error("%s", banner)
            try:
                notify_error(f"🛑 LIVE mode aborted — API key not working: {auth_msg[:80]}")
            except Exception:
                pass
            return   # do not start live trading with a broken key

    # Boot web dashboard in background thread
    start_server(port=8081)

    # Start dynamic universe (auto-discovers top trending Coinbase coins every hour)
    universe.start_background_refresh()

    # Start the fast exit-guard loop (checks stops/targets every POSITION_CHECK_SECONDS,
    # independently of the slower entry scan — professional-grade exit protection)
    threading.Thread(target=_position_monitor_loop, daemon=True,
                     name="position-monitor").start()
    logger.info("Position monitor active — exit guard every %ds",
                getattr(config, "POSITION_CHECK_SECONDS", 45))

    # Initialise state
    st.set_capital(config.TOTAL_CAPITAL_USDT, initial=True)
    st.set_bot_status("running")

    # Arm the circuit breaker with starting equity
    breaker.initialise(config.TOTAL_CAPITAL_USDT)

    logger.info("Dashboard live at http://localhost:8081")
    logger.info("Dynamic universe active — refreshes every 1h, currently %d symbols",
                len(universe.get_watchlist()))
    notify_bot_started(config.TOTAL_CAPITAL_USDT, universe.get_watchlist())

    # ── Train ML models on startup (background, non-blocking) ─────────────────
    if getattr(config, "ML_ENABLED", False) and getattr(config, "ML_TRAIN_ON_START", False):
        _train_ml_models(universe.get_watchlist())

    last_summary_hour = -1   # track hourly portfolio summary
    last_ml_train_hour = -1  # track ML retraining cadence

    while True:
        if is_paused():
            time.sleep(5)
            continue

        st.set_bot_status("scanning")
        signals: list[SignalResult] = []

        # Get current dynamic universe (refreshed every hour automatically)
        # then rank by 6h ROC and keep top 60% for this scan cycle
        current_universe = universe.get_watchlist()
        active_symbols   = _rank_symbols(current_universe)
        st.set_universe(current_universe, universe.get_refresh_log())

        for symbol in active_symbols:
            logger.debug("Scanning %s …", symbol)
            sig = scan_symbol(symbol)
            if sig is not None:
                signals.append(sig)
                if sig.direction in ("long", "short") and sig.score >= config.MIN_SIGNAL_SCORE:
                    logger.info("SIGNAL %s dir=%s score=%.1f rr=%.2f",
                                symbol, sig.direction, sig.score, sig.risk_reward)

        # Push signals to web state
        st.set_signals([_signal_to_dict(s) for s in signals])

        # ── Update circuit breaker BEFORE opening trades ──────────────────────
        _update_circuit_breaker(signals)

        for sig in signals:
            try_open_trade(sig)

        # NOTE: position monitoring is handled by the dedicated fast monitor
        # thread (_position_monitor_loop), not here — avoids double-close races
        # and gives much tighter stop/target enforcement.

        st.set_bot_status("running")
        print_terminal(signals)

        # Send portfolio summary once per hour
        current_hour = datetime.now(timezone.utc).hour
        if current_hour != last_summary_hour:
            last_summary_hour = current_hour
            summary = risk_mgr.summary()
            all_trades = st.get()["trade_history"]
            closed = [t for t in all_trades if t.get("action") == "close"]
            notify_portfolio_summary(
                capital=summary["capital_usdt"],
                start_capital=config.TOTAL_CAPITAL_USDT,
                open_positions=summary["open_details"],
                closed_trades=closed,
            )

        # Retrain ML models every ML_RETRAIN_HOURS
        if getattr(config, "ML_ENABLED", False):
            retrain_interval = getattr(config, "ML_RETRAIN_HOURS", 12)
            if (current_hour % retrain_interval == 0) and (current_hour != last_ml_train_hour):
                last_ml_train_hour = current_hour
                _train_ml_models(current_universe)

        time.sleep(config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        st.set_bot_status("stopped")
        print("\n\nBot stopped.")
        summary = risk_mgr.summary()
        all_trades = st.get()["trade_history"]
        notify_bot_stopped(
            capital=summary["capital_usdt"],
            start_capital=config.TOTAL_CAPITAL_USDT,
            total_trades=len([t for t in all_trades if t.get("action") == "close"]),
        )
        print(f"Final capital: ${summary['capital_usdt']:.2f} USDT")