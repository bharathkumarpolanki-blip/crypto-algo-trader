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
from exchange.market_data import fetch_ohlcv, fetch_ticker, place_order
from core.indicators import enrich
from core.strategies import analyse, SignalResult
from risk.risk_manager import RiskManager
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

def try_open_trade(signal: SignalResult) -> None:
    if signal.direction not in ("long", "short"):
        return
    if signal.score < config.MIN_SIGNAL_SCORE:
        return
    if signal.risk_reward < 1.5:
        logger.info("Skipping %s — RR %.2f < 1.5", signal.symbol, signal.risk_reward)
        return

    can_open, reason = risk_mgr.can_open(signal.symbol)
    if not can_open:
        logger.info("Cannot open %s: %s", signal.symbol, reason)
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
    order = place_order(signal.symbol, side, qty)
    if order is None:
        return

    risk_mgr.open_position(
        symbol=signal.symbol, side=signal.direction,
        entry=entry, qty=qty, stop=stop, take_profit=tp,
        atr=signal.atr, order_id=order.get("id", ""),
    )

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


def check_open_positions() -> None:
    for symbol, pos in list(risk_mgr.positions.items()):
        if pos.status != "open":
            continue
        ticker = fetch_ticker(symbol)
        if not ticker:
            continue
        current_price = ticker.get("last") or ticker.get("close", pos.entry_price)
        action_info   = risk_mgr.update_position(symbol, current_price)

        if action_info["action"] == "close":
            pnl      = risk_mgr.close_position(symbol, current_price, action_info["reason"])
            side_str = "sell" if pos.side == "long" else "buy"
            place_order(symbol, side_str, pos.qty)

            trade_record = {
                "time":   datetime.now(timezone.utc).isoformat(),
                "action": "close",
                "symbol": symbol,
                "side":   pos.side,
                "entry":  pos.entry_price,
                "price":  current_price,
                "pnl":    round(pnl, 4),
                "reason": action_info["reason"],
                "score":  None,
            }
            st.add_trade(trade_record)
            st.set_capital(risk_mgr.summary()["capital_usdt"])

            # Calculate how long the trade was held
            hold_mins = None
            try:
                opened_at = pos.opened_at
                hold_mins = int((datetime.now(timezone.utc) - opened_at).total_seconds() / 60)
            except Exception:
                pass

            notify_trade_close(
                symbol=symbol,
                direction=pos.side,
                entry=pos.entry_price,
                exit_price=current_price,
                qty=pos.qty,
                pnl=pnl,
                reason=action_info["reason"],
                hold_duration_mins=hold_mins,
            )
            logger.info("TRADE CLOSED: %s @ %.6f  Reason=%s  PnL=%.4f",
                        symbol, current_price, action_info["reason"], pnl)

    # Push updated open positions to state
    summary = risk_mgr.summary()
    st.set_open_positions(summary["open_details"])


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

    # Boot web dashboard in background thread
    start_server(port=8081)

    # Start dynamic universe (auto-discovers top trending Coinbase coins every hour)
    universe.start_background_refresh()

    # Initialise state
    st.set_capital(config.TOTAL_CAPITAL_USDT, initial=True)
    st.set_bot_status("running")

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

        for sig in signals:
            try_open_trade(sig)

        check_open_positions()

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