"""
Flask web dashboard — serves the trading bot UI and JSON API.

Endpoints:
  GET  /              → HTML dashboard
  GET  /api/state     → full state JSON (polled every 5s by the UI)
  GET  /api/trades    → paginated trade history
  GET  /api/positions → open positions
  GET  /api/signals   → latest signals
  GET  /api/equity    → equity curve data
  POST /api/control   → {"action": "pause" | "resume"}
"""

import threading
import logging
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

import ui.state as st

logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

_bot_paused = threading.Event()
_bot_paused.clear()   # not paused by default


def is_paused() -> bool:
    return _bot_paused.is_set()


# ── API ────────────────────────────────────────────────────────────────────────

@app.route("/api/state")
def api_state():
    return jsonify(st.get())


@app.route("/api/trades")
def api_trades():
    page  = int(request.args.get("page", 1))
    size  = int(request.args.get("size", 50))
    data  = st.get()["trade_history"]
    total = len(data)
    data  = list(reversed(data))   # newest first
    start = (page - 1) * size
    return jsonify({"total": total, "page": page, "trades": data[start: start + size]})


@app.route("/api/positions")
def api_positions():
    return jsonify(st.get()["open_positions"])


@app.route("/api/signals")
def api_signals():
    return jsonify(st.get()["signals"])


@app.route("/api/equity")
def api_equity():
    return jsonify(st.get()["equity_curve"])


@app.route("/api/control", methods=["POST"])
def api_control():
    action = (request.json or {}).get("action", "")
    if action == "pause":
        _bot_paused.set()
        st.set_bot_status("paused")
        return jsonify({"status": "paused"})
    elif action == "resume":
        _bot_paused.clear()
        st.set_bot_status("running")
        return jsonify({"status": "running"})
    return jsonify({"error": "unknown action"}), 400


# ── Symbol detail API ─────────────────────────────────────────────────────────

_candle_cache: dict = {}
_CANDLE_TTL = 60   # seconds

@app.route("/api/candles/<path:symbol>")
def api_candles(symbol):
    """Return OHLCV + indicators for TradingView Lightweight Charts."""
    import time as _time
    symbol = symbol.replace("-", "/")

    cached = _candle_cache.get(symbol)
    if cached and (_time.time() - cached["ts"]) < _CANDLE_TTL:
        return jsonify(cached["data"])

    try:
        from exchange.market_data import fetch_ohlcv
        from core.indicators import (add_emas, add_rsi, add_macd, add_bollinger,
                                 add_atr, add_volume_indicators, add_adx)
        import numpy as np

        df = fetch_ohlcv(symbol, "1h", limit=200)
        if df.empty:
            return jsonify({"error": "no data"}), 404

        df = add_emas(df); df = add_rsi(df); df = add_macd(df)
        df = add_bollinger(df); df = add_atr(df)
        df = add_volume_indicators(df); df = add_adx(df)

        def s(v):
            if v is None: return None
            try:
                f = float(v)
                return None if np.isnan(f) else round(f, 6)
            except Exception: return None

        candles, volumes, ema9, ema21, ema50, ema200 = [], [], [], [], [], []
        rsi_data, macd_data, macd_sig, macd_hist = [], [], [], []
        bb_up_data, bb_mid_data, bb_low_data = [], [], []
        vol_ratio_data = []

        for ts, row in df.iterrows():
            t = int(ts.timestamp())
            candles.append({"time": t, "open": s(row["open"]), "high": s(row["high"]),
                            "low": s(row["low"]), "close": s(row["close"])})
            volumes.append({"time": t, "value": s(row["volume"]),
                            "color": "rgba(63,185,80,.5)" if row["close"] >= row["open"]
                                     else "rgba(248,81,73,.5)"})
            ema9.append({"time": t, "value": s(row.get("ema9"))})
            ema21.append({"time": t, "value": s(row.get("ema21"))})
            ema50.append({"time": t, "value": s(row.get("ema50"))})
            ema200.append({"time": t, "value": s(row.get("ema200"))})
            rsi_data.append({"time": t, "value": s(row.get("rsi"))})
            macd_data.append({"time": t, "value": s(row.get("macd"))})
            macd_sig.append({"time": t, "value": s(row.get("macd_signal"))})
            macd_hist.append({"time": t, "value": s(row.get("macd_hist")),
                              "color": "rgba(63,185,80,.7)" if (row.get("macd_hist") or 0) >= 0
                                       else "rgba(248,81,73,.7)"})
            bb_up_data.append({"time": t, "value": s(row.get("bb_up"))})
            bb_mid_data.append({"time": t, "value": s(row.get("bb_mid"))})
            bb_low_data.append({"time": t, "value": s(row.get("bb_low"))})
            vol_ratio_data.append(s(row.get("vol_ratio")))

        # Filter out None values from indicator series
        def clean(series):
            return [p for p in series if p.get("value") is not None]

        result = {
            "symbol": symbol,
            "candles": candles,
            "volume": volumes,
            "ema9": clean(ema9), "ema21": clean(ema21),
            "ema50": clean(ema50), "ema200": clean(ema200),
            "rsi": clean(rsi_data),
            "macd": clean(macd_data), "macd_signal": clean(macd_sig),
            "macd_hist": [p for p in macd_hist if p.get("value") is not None],
            "bb_up": clean(bb_up_data), "bb_mid": clean(bb_mid_data),
            "bb_low": clean(bb_low_data),
            "last_price": s(df["close"].iloc[-1]),
            "price_change_pct": round(
                (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100, 2
            ) if len(df) > 1 else 0,
        }
        _candle_cache[symbol] = {"ts": _time.time(), "data": result}
        return jsonify(result)
    except Exception as e:
        logger.error("Candle fetch failed for %s: %s", symbol, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/symbol/<path:symbol>")
def api_symbol_detail(symbol):
    """Return signal, open position, and recent trades for one symbol."""
    symbol = symbol.replace("-", "/")
    state  = st.get()

    signal   = next((s for s in state["signals"] if s["symbol"] == symbol), None)
    position = next((p for p in state["open_positions"] if p["symbol"] == symbol), None)
    trades   = [t for t in reversed(state["trade_history"]) if t.get("symbol") == symbol][:20]

    return jsonify({"symbol": symbol, "signal": signal,
                    "position": position, "trades": trades})


# ── Universe API ──────────────────────────────────────────────────────────────

@app.route("/api/universe", methods=["GET"])
def api_universe():
    return jsonify(st.get_universe())


@app.route("/api/universe/refresh", methods=["POST"])
def api_universe_refresh():
    def _run():
        import exchange.universe as u
        u.refresh_universe(force=True)
        import ui.state as _st
        _st.set_universe(u.get_watchlist(), u.get_refresh_log())
    threading.Thread(target=_run, daemon=True, name="universe-force-refresh").start()
    return jsonify({"status": "refreshing"})


# ── ML API ─────────────────────────────────────────────────────────────────────

@app.route("/api/ml/status", methods=["GET"])
def api_ml_status():
    """Return per-symbol ML model status: AUC, MAE, features, top features."""
    try:
        import config as _cfg
        from ml.signal_predictor import get_predictor
        from ml.extrema_predictor import get_extrema_predictor
        import exchange.universe as u

        predictor = get_predictor()
        extrema   = get_extrema_predictor()
        symbols   = u.get_watchlist()

        rows = []
        for sym in symbols:
            sig_stats = predictor.model_stats(sym)
            ex_stats  = extrema.model_stats(sym)
            if not sig_stats and not ex_stats:
                continue
            test_auc = sig_stats.get("test_auc", 0.0)
            rows.append({
                "symbol":       sym,
                "test_auc":     test_auc,
                "train_auc":    sig_stats.get("train_auc", 0.0),
                "n_features":   sig_stats.get("n_features", 0),
                "auc_usable":   test_auc >= 0.53,
                "extrema_mae":  ex_stats.get("mae", None),
                "extrema_usable": (ex_stats.get("mae", 1.0) <= 0.45) if ex_stats else False,
                "trained_at":   sig_stats.get("trained_at", ex_stats.get("trained_at", "")),
                "top_features": predictor.feature_importance(sym, 5),
            })

        return jsonify({
            "enabled":   getattr(_cfg, "ML_ENABLED", False),
            "weight":    getattr(_cfg, "ML_WEIGHT", 1.0),
            "models":    rows,
            "n_models":  len(rows),
        })
    except Exception as e:
        logger.error("ML status failed: %s", e)
        return jsonify({"error": str(e), "models": []}), 500


@app.route("/api/ml/train", methods=["POST"])
def api_ml_train():
    """Trigger ML retraining for all current universe symbols."""
    def _run():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import config as _cfg
        import exchange.universe as u
        from exchange.market_data import fetch_ohlcv
        from core.indicators import enrich
        from ml.signal_predictor import get_predictor
        from ml.extrema_predictor import get_extrema_predictor
        from ml.regime_classifier import get_regime_classifier

        predictor = get_predictor()
        extrema   = get_extrema_predictor()
        regime    = get_regime_classifier()
        st.set_ml_training(True, "Starting…")

        symbols = u.get_watchlist()
        regime_done = False
        for i, sym in enumerate(symbols):
            st.set_ml_training(True, f"[{i+1}/{len(symbols)}] Training {sym}…")
            try:
                df = fetch_ohlcv(sym, _cfg.TF_PRIMARY, limit=1000)
                if df.empty or len(df) < 250:
                    continue
                df = enrich(df)
                predictor.train(sym, df)
                extrema.train(sym, df)
                if not regime_done and sym == "BTC/USD":
                    if regime.train(df):
                        regime.save()
                        regime_done = True
            except Exception as e:
                logger.warning("ML train error %s: %s", sym, e)
        st.set_ml_training(False, "Done")

    if st.get_ml_training()["active"]:
        return jsonify({"error": "training already running"}), 409
    threading.Thread(target=_run, daemon=True, name="ml-train-manual").start()
    return jsonify({"status": "started"})


@app.route("/api/ml/training_status", methods=["GET"])
def api_ml_training_status():
    return jsonify(st.get_ml_training())


# ── Circuit Breaker API ─────────────────────────────────────────────────────────

@app.route("/api/breaker", methods=["GET"])
def api_breaker_status():
    return jsonify(st.get_circuit_breaker())


@app.route("/api/breaker/reset", methods=["POST"])
def api_breaker_reset():
    try:
        from risk.circuit_breaker import get_breaker
        get_breaker().manual_reset()
        st.set_circuit_breaker(get_breaker().status())
        return jsonify({"status": "reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/breaker/trip", methods=["POST"])
def api_breaker_trip():
    """Manual emergency halt — operator kill switch."""
    try:
        from risk.circuit_breaker import get_breaker
        get_breaker().manual_trip("manual halt from dashboard")
        st.set_circuit_breaker(get_breaker().status())
        return jsonify({"status": "tripped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── SMA Trend strategy API ──────────────────────────────────────────────────────

@app.route("/api/sma/status", methods=["GET"])
def api_sma_status():
    """Live SMA200 (+weekly) status per coin."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import config as _cfg
        import sma_bot
        signals = sma_bot.evaluate()
        rows = []
        for sym, v in signals.items():
            rows.append({
                "symbol":     sym,
                "close":      round(v["close"], 4),
                "sma":        round(v["sma"], 4),
                "pct":        round(v["pct"], 2),
                "daily_up":   v.get("daily_up", v["uptrend"]),
                "weekly_up":  v.get("weekly_up", True),
                "weekly_pct": round(v.get("weekly_pct", 0), 2),
                "hold":       v["uptrend"],
            })
        mode = "alert-only" if getattr(_cfg, "SMA_ALERT_ONLY", True) else \
               ("paper" if _cfg.DRY_RUN else "live")
        return jsonify({
            "period":      getattr(_cfg, "SMA_PERIOD", 200),
            "weekly_gate": getattr(_cfg, "SMA_USE_WEEKLY_GATE", False),
            "weekly_period": getattr(_cfg, "SMA_WEEKLY_PERIOD", 30),
            "mode":        mode,
            "rows":        rows,
        })
    except Exception as e:
        logger.error("SMA status failed: %s", e)
        return jsonify({"error": str(e), "rows": []}), 500


@app.route("/api/sma/paper", methods=["GET"])
def api_sma_paper():
    """
    Paper/live activity from a separately-running `sma_bot.py` loop. Read from
    sma_state.json (the cross-process channel — the SMA bot runs in its own
    process, so we can't use the in-memory dashboard state).
    """
    try:
        import json, os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sma_state.json")
        if not os.path.exists(path):
            return jsonify({"running": False, "msg": "sma_bot has not run yet"})
        state = json.load(open(path))
        paper = state.get("paper")
        if not paper:
            return jsonify({"running": False, "msg": "no paper activity recorded yet"})
        return jsonify({"running": True, "last_check": state.get("last_check"), **paper})
    except Exception as e:
        logger.error("SMA paper read failed: %s", e)
        return jsonify({"running": False, "error": str(e)}), 500


@app.route("/api/sma/backtest", methods=["POST"])
def api_sma_backtest():
    """Run the SMA portfolio backtest in the background."""
    body  = request.json or {}
    years = float(body.get("years", 3))
    cap   = float(body.get("capital", 1000))

    if st.get_sma_backtest().get("status") == "running":
        return jsonify({"error": "already running"}), 409

    def _run():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import sma_bot
        st.set_sma_backtest({"status": "running", "result": None})
        try:
            res = sma_bot.backtest_sma(years=years, capital=cap)
            st.set_sma_backtest({"status": "done", "result": res})
        except Exception as e:
            logger.error("SMA backtest failed: %s", e)
            st.set_sma_backtest({"status": "error", "result": {"error": str(e)}})

    threading.Thread(target=_run, daemon=True, name="sma-backtest").start()
    return jsonify({"status": "started"})


@app.route("/api/sma/backtest_status", methods=["GET"])
def api_sma_backtest_status():
    return jsonify(st.get_sma_backtest())


# ── Auto-Tuner API ─────────────────────────────────────────────────────────────

@app.route("/api/tuner/run", methods=["POST"])
def api_tuner_run():
    """Run Optuna parameter tuning on a symbol."""
    body    = request.json or {}
    symbol  = body.get("symbol", "BTC/USD")
    days    = int(body.get("days", 90))
    trials  = int(body.get("trials", 50))

    if st.get_tuner()["active"]:
        return jsonify({"error": "tuner already running"}), 409

    def _run():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from exchange.market_data import fetch_ohlcv
        from core.indicators import enrich
        from core.strategies import analyse
        from ml.auto_tuner import tune_parameters

        st.set_tuner(True, f"Fetching {symbol} data…", None)
        try:
            limit = min(days * 24 + 300, 1000)
            df = fetch_ohlcv(symbol, "1h", limit=limit)
            if df.empty or len(df) < 200:
                st.set_tuner(False, "Not enough data", None)
                return
            df = enrich(df)

            def progress(n, total, best):
                st.set_tuner(True, f"Trial {n}/{total} · best Sharpe {best:.2f}", None)

            result = tune_parameters(df, analyse, symbol, n_trials=trials,
                                     progress_callback=progress)
            st.set_tuner(False, "Complete", {
                "symbol":       symbol,
                "best_params":  result.best_params,
                "best_sharpe":  result.best_sharpe,
                "best_return":  result.best_return,
                "n_trials":     result.n_trials,
                "improvement":  result.improvement,
            })
        except Exception as e:
            logger.error("Tuner failed: %s", e)
            st.set_tuner(False, f"Error: {e}", None)

    threading.Thread(target=_run, daemon=True, name="tuner").start()
    return jsonify({"status": "started"})


@app.route("/api/tuner/status", methods=["GET"])
def api_tuner_status():
    return jsonify(st.get_tuner())


@app.route("/api/tuner/apply", methods=["POST"])
def api_tuner_apply():
    """Apply the last tuning result to the running config."""
    result = st.get_tuner().get("result")
    if not result:
        return jsonify({"error": "no tuning result to apply"}), 400
    import config as _cfg
    for k, v in result.get("best_params", {}).items():
        if hasattr(_cfg, k):
            setattr(_cfg, k, v)
    return jsonify({"status": "applied", "params": result["best_params"]})


# ── Backtest API ───────────────────────────────────────────────────────────────

@app.route("/api/backtest", methods=["GET"])
def api_backtest_get():
    return jsonify(st.get_backtest())


@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    body    = request.json or {}
    symbols = body.get("symbols", [])   # [] means all watchlist
    days    = int(body.get("days", 90))
    capital = float(body.get("capital", 1000))

    if st.get_backtest()["status"] == "running":
        return jsonify({"error": "backtest already running"}), 409

    def _run():
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # ensure root on path
        import config as _cfg
        from backtest import run_backtest

        # Use dynamic universe when no specific symbols requested
        if symbols:
            target = symbols
        else:
            try:
                import exchange.universe as _u
                target = _u.get_watchlist()
            except Exception:
                target = _cfg.WATCHLIST
        all_results = []
        for i, sym in enumerate(target):
            st.set_backtest_status(
                "running",
                f"[{i+1}/{len(target)}] Scanning {sym}…"
            )
            try:
                r = run_backtest(sym, "1h", days, capital,
                                 _cfg.RISK_PER_TRADE_PCT, verbose=False)
                if r:
                    # attach per-trade equity curve for the chart
                    all_results.append(r)
            except Exception as e:
                logger.warning("Backtest failed for %s: %s", sym, e)

        all_results.sort(key=lambda x: x.get("return_pct", 0), reverse=True)
        st.set_backtest_results(all_results)
        st.set_backtest_status("done", f"Completed {len(all_results)} symbols")

    t = threading.Thread(target=_run, daemon=True, name="backtest-runner")
    t.start()
    return jsonify({"status": "started"})


# ── HTML dashboard ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Crypto Trading Bot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root {
  --bg:      #0d1117; --bg2: #161b22; --bg3: #21262d;
  --border:  #30363d; --text: #e6edf3; --muted: #8b949e;
  --green:   #3fb950; --red: #f85149; --yellow: #d29922;
  --blue:    #58a6ff; --purple: #bc8cff; --radius: 8px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace; font-size: 14px; }

/* ── Layout ── */
.header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 18px; font-weight: 700; }
.header h1 span { color: var(--blue); }
.status-pill { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600; padding: 5px 12px; border-radius: 20px; background: var(--bg3); border: 1px solid var(--border); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.dot.running  { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
.dot.scanning { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); animation: pulse .8s infinite; }
.dot.paused   { background: var(--yellow); }
.dot.stopped  { background: var(--red); }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.4} }

/* ── Tabs ── */
.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); background: var(--bg2); padding: 0 24px; }
.tab { padding: 12px 20px; cursor: pointer; font-size: 13px; font-weight: 600; color: var(--muted); border-bottom: 2px solid transparent; transition: all .15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--blue); border-bottom-color: var(--blue); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

.ticker-bar { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 6px 24px; font-size: 12px; color: var(--muted); display: flex; gap: 24px; overflow-x: auto; white-space: nowrap; }
.main { padding: 20px 24px; max-width: 1600px; margin: 0 auto; }

/* ── Stats bar ── */
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 20px; }
.stat-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; }
.stat-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px; }
.stat-card .value { font-size: 22px; font-weight: 700; }
.stat-card .sub   { font-size: 12px; color: var(--muted); margin-top: 3px; }
.pos { color: var(--green); } .neg { color: var(--red); } .neu { color: var(--text); }

/* ── Grid ── */
.row       { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.row.three { grid-template-columns: 2fr 1fr; }
@media (max-width: 900px) { .row, .row.three { grid-template-columns: 1fr; } }

/* ── Cards ── */
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; margin-bottom: 16px; }
.card-header { padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
.card-header h2 { font-size: 13px; font-weight: 600; }
.chart-wrap { padding: 12px 16px 8px; height: 220px; }

/* ── Tables ── */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: .5px; padding: 8px 16px; text-align: left; border-bottom: 1px solid var(--border); }
td { padding: 9px 16px; border-bottom: 1px solid #1c2129; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #1c2129; }
.mono { font-family: monospace; }

/* ── Score bar ── */
.score-wrap { display: flex; align-items: center; gap: 8px; }
.score-bar  { flex: 1; height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; max-width: 80px; }
.score-fill { height: 100%; border-radius: 3px; transition: width .4s; }

/* ── PF bar ── */
.pf-bar-wrap { display: flex; align-items: center; gap: 8px; }
.pf-bar      { width: 80px; height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
.pf-fill     { height: 100%; border-radius: 3px; }

/* ── Badges ── */
.badge         { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-long    { background: rgba(63,185,80,.15); color: var(--green); }
.badge-short   { background: rgba(248,81,73,.15); color: var(--red); }
.badge-neutral { background: var(--bg3); color: var(--muted); }
.badge-tp      { background: rgba(63,185,80,.15); color: var(--green); }
.badge-sl      { background: rgba(248,81,73,.15); color: var(--red); }

/* ── Buttons ── */
.btn          { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg3); color: var(--text); cursor: pointer; font-size: 12px; font-weight: 600; transition: all .15s; }
.btn:hover    { background: var(--border); }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-danger   { border-color: var(--red);   color: var(--red); }
.btn-success  { border-color: var(--green); color: var(--green); }
.btn-primary  { border-color: var(--blue);  color: var(--blue); }
.btn-danger:hover  { background: rgba(248,81,73,.15); }
.btn-success:hover { background: rgba(63,185,80,.15); }
.btn-primary:hover { background: rgba(88,166,255,.12); }

/* ── Pagination ── */
.pagination { display: flex; align-items: center; gap: 8px; padding: 10px 16px; border-top: 1px solid var(--border); justify-content: flex-end; }

/* ── Empty ── */
.empty { text-align: center; padding: 32px; color: var(--muted); font-size: 13px; }

/* ── Backtest form ── */
.bt-form { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; padding: 16px; border-bottom: 1px solid var(--border); }
.bt-form label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; display: flex; flex-direction: column; gap: 5px; }
.bt-form input, .bt-form select {
  background: var(--bg3); border: 1px solid var(--border); color: var(--text);
  padding: 6px 10px; border-radius: 6px; font-size: 13px; width: 140px;
}
.bt-form select[multiple] { height: 100px; }
.bt-progress { padding: 12px 16px; font-size: 13px; color: var(--muted); border-bottom: 1px solid var(--border); display: none; }
.bt-progress.visible { display: block; }
.spinner { display: inline-block; width: 10px; height: 10px; border: 2px solid var(--muted); border-top-color: var(--blue); border-radius: 50%; animation: spin .7s linear infinite; margin-right: 8px; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Summary cards ── */
.bt-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; padding: 16px; }
.bt-stat { background: var(--bg3); border-radius: var(--radius); padding: 12px; }
.bt-stat .l { font-size: 11px; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
.bt-stat .v { font-size: 18px; font-weight: 700; }

.last-update { font-size: 11px; color: var(--muted); }

/* ── Scrollable table containers ── */
.table-scroll { overflow-x: auto; overflow-y: auto; max-height: 460px; }
.table-scroll thead th { position: sticky; top: 0; z-index: 2; background: var(--bg2); }

/* ── Custom visible scrollbars (dark theme) ── */
* { scrollbar-width: thin; scrollbar-color: #484f58 var(--bg2); }
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: var(--bg2); border-radius: 6px; }
::-webkit-scrollbar-thumb {
  background: #484f58; border-radius: 6px; border: 2px solid var(--bg2);
}
::-webkit-scrollbar-thumb:hover { background: #5b626b; }
::-webkit-scrollbar-corner { background: var(--bg2); }

/* ── Symbol detail drawer ── */
.drawer-overlay { position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:200;display:none;backdrop-filter:blur(2px); }
.drawer-overlay.open { display:block; }
.drawer { position:fixed;top:0;right:0;width:min(960px,96vw);height:100vh;background:var(--bg2);
  border-left:1px solid var(--border);z-index:201;display:flex;flex-direction:column;
  transform:translateX(100%);transition:transform .25s cubic-bezier(.4,0,.2,1); }
.drawer.open { transform:translateX(0); }
.drawer-header { padding:14px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-shrink:0; }
.drawer-header h2 { font-size:18px;font-weight:700;flex:1; }
.drawer-price { font-size:22px;font-weight:700;font-family:monospace; }
.drawer-change { font-size:13px;font-weight:600;padding:3px 10px;border-radius:12px; }
.drawer-body { flex:1;overflow-y:auto;padding:16px; }

/* Chart containers */
.chart-panel { background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:12px;overflow:hidden; }
.chart-panel-header { padding:8px 14px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border);font-size:12px;font-weight:600; }
.chart-panel-body { position:relative; }

/* Indicator legend */
.legend { display:flex;gap:14px;font-size:11px; }
.legend span { display:flex;align-items:center;gap:4px; }
.legend i { width:20px;height:2px;display:inline-block;border-radius:1px; }

/* Position / signal cards in drawer */
.info-grid { display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px; }
.info-card { background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);padding:12px; }
.info-card .label { font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px; }
.info-card .value { font-size:15px;font-weight:700; }

/* Clickable rows */
tr.clickable { cursor:pointer; }
tr.clickable:hover td { background:#1f2937 !important; }
tr.clickable td:first-child::after { content:' ↗';font-size:10px;color:var(--muted); }
</style>
</head>
<body>

<header class="header">
  <h1>🤖 <span>Crypto</span> Trading Bot</h1>
  <div style="display:flex;gap:12px;align-items:center">
    <span class="last-update" id="lastUpdate">–</span>
    <div class="status-pill"><div class="dot" id="statusDot"></div><span id="statusText">Connecting…</span></div>
    <button class="btn btn-danger" id="pauseBtn" onclick="toggleBot()">Pause</button>
  </div>
</header>

<div id="authBanner" style="display:none;padding:10px 24px;font-size:13px;font-weight:600;text-align:center"></div>
<div id="cbBanner" style="display:none;padding:10px 24px;font-size:13px;font-weight:700;text-align:center;background:rgba(248,81,73,.18);color:var(--red);border-bottom:1px solid var(--red)">
  🛑 <span id="cbBannerText">CIRCUIT BREAKER TRIPPED</span>
  <button class="btn btn-success" style="margin-left:12px;padding:3px 10px" onclick="resetBreaker()">Reset &amp; Resume</button>
</div>
<div class="ticker-bar" id="tickerBar">Loading…</div>

<!-- ═══════════════════ SYMBOL DETAIL DRAWER ════════════════════════════════ -->
<div class="drawer-overlay" id="drawerOverlay" onclick="closeDrawer()"></div>
<div class="drawer" id="symbolDrawer">
  <div class="drawer-header">
    <div>
      <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px" id="drawerExchange">Coinbase · 1h</div>
      <h2 id="drawerSymbol">—</h2>
    </div>
    <div class="drawer-price" id="drawerPrice">—</div>
    <div class="drawer-change" id="drawerChange">—</div>
    <button class="btn" onclick="closeDrawer()" style="margin-left:auto">✕ Close</button>
  </div>

  <div class="drawer-body">

    <!-- Info cards row -->
    <div class="info-grid" id="drawerCards"></div>

    <!-- Candlestick + EMA chart -->
    <div class="chart-panel">
      <div class="chart-panel-header">
        Candlestick · 1h
        <div class="legend">
          <span><i style="background:#f97316"></i>EMA9</span>
          <span><i style="background:#facc15"></i>EMA21</span>
          <span><i style="background:#60a5fa"></i>EMA50</span>
          <span><i style="background:#a78bfa"></i>EMA200</span>
          <span><i style="background:rgba(88,166,255,.4);height:1px;border:1px dashed #58a6ff"></i>BB</span>
        </div>
        <div style="margin-left:auto;display:flex;gap:6px">
          <button class="btn" onclick="setTf('1h')" id="tf1h" style="font-size:11px;padding:3px 8px">1h</button>
          <button class="btn" onclick="setTf('6h')" id="tf6h" style="font-size:11px;padding:3px 8px">6h</button>
          <button class="btn" onclick="setTf('1d')" id="tf1d" style="font-size:11px;padding:3px 8px">1d</button>
        </div>
      </div>
      <div class="chart-panel-body"><div id="candleChart" style="height:320px"></div></div>
    </div>

    <!-- Volume chart -->
    <div class="chart-panel">
      <div class="chart-panel-header">Volume <span id="volRatioLabel" style="color:var(--muted);font-weight:400"></span></div>
      <div class="chart-panel-body"><div id="volumeChart" style="height:100px"></div></div>
    </div>

    <!-- RSI chart -->
    <div class="chart-panel">
      <div class="chart-panel-header">
        RSI (14)
        <div class="legend" style="margin-left:8px">
          <span style="color:var(--red)">Overbought 65</span>
          <span style="color:var(--green)">Oversold 35</span>
        </div>
      </div>
      <div class="chart-panel-body"><div id="rsiChart" style="height:100px"></div></div>
    </div>

    <!-- MACD chart -->
    <div class="chart-panel">
      <div class="chart-panel-header">MACD (12/26/9)</div>
      <div class="chart-panel-body"><div id="macdChart" style="height:110px"></div></div>
    </div>

    <!-- Trade history for this symbol -->
    <div class="card" style="margin-top:4px">
      <div class="card-header"><h2 id="drawerTradeTitle">Trade History</h2></div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Time</th><th>Action</th><th>Side</th><th>Entry</th><th>Exit / TP</th><th>P&L</th><th>Reason</th></tr></thead>
          <tbody id="drawerTradeBody"><tr><td colspan="7" class="empty">No trades yet</td></tr></tbody>
        </table>
      </div>
    </div>

  </div>
</div>

<!-- Tab bar -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('live')">📊 Live Trading</div>
  <div class="tab" onclick="switchTab('backtest')">🧪 Backtest</div>
  <div class="tab" onclick="switchTab('universe')">🌍 Universe</div>
  <div class="tab" onclick="switchTab('ml')">🧠 Machine Learning</div>
  <div class="tab" onclick="switchTab('sma')">📈 SMA Trend</div>
</div>

<!-- ═══════════════════════════════ LIVE TAB ════════════════════════════════ -->
<div id="tab-live" class="tab-panel active">
<main class="main">

  <div class="stats-grid">
    <div class="stat-card"><div class="label">Portfolio Value</div><div class="value" id="capital">—</div><div class="sub" id="capitalChange">—</div></div>
    <div class="stat-card"><div class="label">Total P&amp;L</div><div class="value" id="totalPnl">—</div><div class="sub" id="totalPnlPct">—</div></div>
    <div class="stat-card"><div class="label">Open Positions</div><div class="value" id="openCount">—</div><div class="sub" id="openUnreal">unrealised</div></div>
    <div class="stat-card"><div class="label">Total Trades</div><div class="value" id="tradeCount">—</div><div class="sub" id="winRate">—</div></div>
    <div class="stat-card"><div class="label">Scan Count</div><div class="value" id="scanCount">—</div><div class="sub" id="lastScan">—</div></div>
  </div>

  <div class="row three">
    <div class="card">
      <div class="card-header"><h2>Equity Curve</h2><span class="last-update">live</span></div>
      <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-header"><h2>Open Positions</h2></div>
      <div class="table-scroll">
        <table><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>SL</th><th>TP</th><th>Unreal P&L</th></tr></thead>
        <tbody id="posBody"><tr><td colspan="6" class="empty">No open positions</td></tr></tbody></table>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Signal Scanner</h2><span class="last-update" id="sigNote">–</span></div>
    <div class="table-scroll">
      <table><thead><tr><th>Symbol</th><th>Direction</th><th>Score</th><th>Entry</th><th>Stop</th><th>Target</th><th>R:R</th><th>Sentiment</th><th>Key Signals</th><th>Patterns</th></tr></thead>
      <tbody id="sigBody"><tr><td colspan="10" class="empty">Waiting for first scan…</td></tr></tbody></table>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Trade History</h2></div>
    <div class="table-scroll">
      <table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Action</th><th>Entry</th><th>Exit</th><th>Score</th><th>P&amp;L</th><th>Reason</th></tr></thead>
      <tbody id="tradeBody"><tr><td colspan="9" class="empty">No trades yet</td></tr></tbody></table>
    </div>
    <div class="pagination">
      <span class="last-update" id="tradePagination"></span>
      <button class="btn" id="prevBtn" onclick="changePage(-1)" disabled>‹ Prev</button>
      <button class="btn" id="nextBtn" onclick="changePage(1)">Next ›</button>
    </div>
  </div>

</main>
</div><!-- /tab-live -->

<!-- ════════════════════════════ BACKTEST TAB ══════════════════════════════ -->
<div id="tab-backtest" class="tab-panel">
<main class="main">

  <div class="card">
    <div class="card-header">
      <h2>Run Backtest</h2>
      <span class="last-update" id="btLastRun">–</span>
    </div>

    <div class="bt-form">
      <label>Symbols
        <select id="btSymbols" multiple style="height:110px;width:200px">
          <option value="__all__" selected>— All Watchlist —</option>
        </select>
      </label>
      <label>Days
        <input id="btDays" type="number" value="90" min="7" max="365"/>
      </label>
      <label>Capital (USD)
        <input id="btCapital" type="number" value="1000" min="100"/>
      </label>
      <button class="btn btn-primary" id="btRunBtn" onclick="runBacktest()" style="margin-bottom:1px">▶ Run Backtest</button>
    </div>

    <div class="bt-progress" id="btProgress">
      <span class="spinner"></span><span id="btProgressText">Running…</span>
    </div>
  </div>

  <!-- Aggregate stats (shown when ≥1 result) -->
  <div id="btSummarySection" style="display:none">
    <div class="bt-summary" id="btSummary"></div>
  </div>

  <!-- Per-symbol results table + equity chart side by side -->
  <div class="row" id="btResultsRow" style="display:none">
    <div class="card">
      <div class="card-header"><h2>Per-Symbol Results</h2></div>
      <div class="table-scroll">
        <table id="btTable">
          <thead><tr>
            <th>Symbol</th><th>Return</th><th>P&L</th>
            <th>Trades</th><th>Win %</th><th>Profit Factor</th>
            <th>Avg Win</th><th>Avg Loss</th><th>Max DD</th><th>Sharpe</th>
          </tr></thead>
          <tbody id="btBody"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><h2>Return Distribution</h2></div>
      <div class="chart-wrap" style="height:260px"><canvas id="btBarChart"></canvas></div>
    </div>
  </div>

  <!-- Detailed trade log for selected symbol -->
  <div class="card" id="btTradeLogCard" style="display:none">
    <div class="card-header">
      <h2>Trade Log — <span id="btTradeLogSymbol">–</span></h2>
      <button class="btn" onclick="closeBtTradeLog()">✕ Close</button>
    </div>
    <div class="row" style="padding:12px 16px;gap:12px;margin-bottom:0">
      <div>
        <canvas id="btEquityChart" style="height:180px"></canvas>
      </div>
      <div class="table-scroll" style="max-height:280px;overflow-y:auto">
        <table>
          <thead><tr><th>Time</th><th>Side</th><th>Entry</th><th>Exit</th><th>Reason</th><th>P&L</th></tr></thead>
          <tbody id="btTradeLog"></tbody>
        </table>
      </div>
    </div>
  </div>

</main>
</div><!-- /tab-backtest -->

<!-- ═══════════════════════════ UNIVERSE TAB ═══════════════════════════════ -->
<div id="tab-universe" class="tab-panel">
<main class="main">

  <div class="card">
    <div class="card-header">
      <h2>🌍 Dynamic Universe — Active Trading Symbols</h2>
      <div style="display:flex;gap:8px;align-items:center">
        <span class="last-update" id="univNextRefresh">–</span>
        <button class="btn btn-primary" onclick="forceRefreshUniverse()">⟳ Refresh Now</button>
      </div>
    </div>
    <div style="padding:12px 16px;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted)">
      Re-scans all 390+ Coinbase USD pairs every hour. Ranks by 24h volume, 24h momentum, and 7-day ROC.
      BTC, ETH, SOL are always included as anchors.
    </div>
    <div id="univCurrentSymbols" style="padding:16px;display:flex;flex-wrap:wrap;gap:8px">
      <span style="color:var(--muted)">Loading…</span>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Refresh History</h2></div>
    <div class="table-scroll">
      <table>
        <thead><tr><th>Time (UTC)</th><th>Total</th><th>Added</th><th>Removed</th><th>Symbols</th></tr></thead>
        <tbody id="univLogBody"><tr><td colspan="5" class="empty">No refresh history yet</td></tr></tbody>
      </table>
    </div>
  </div>

</main>
</div><!-- /tab-universe -->

<!-- ═══════════════════════════════ ML TAB ══════════════════════════════════ -->
<div id="tab-ml" class="tab-panel">
<main class="main">

  <!-- ML models overview -->
  <div class="card">
    <div class="card-header">
      <h2>🧠 ML Models — Signal Predictor &amp; Extrema</h2>
      <div style="display:flex;gap:10px;align-items:center">
        <span class="last-update" id="mlTrainStatus">–</span>
        <button class="btn btn-primary" id="mlTrainBtn" onclick="trainML()">⟳ Retrain All</button>
      </div>
    </div>
    <div style="padding:12px 16px;font-size:12px;color:var(--muted);border-bottom:1px solid var(--border)">
      Models predict the probability of a profitable trade (AUC) and proximity to local tops/bottoms (extrema).
      Only models that beat random (AUC&nbsp;≥&nbsp;0.53) are used in live scoring.
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr>
          <th>Symbol</th><th>Test AUC</th><th>Train AUC</th><th>Status</th>
          <th>Extrema MAE</th><th>Features</th><th>Top Features</th><th>Trained</th>
        </tr></thead>
        <tbody id="mlBody"><tr><td colspan="8" class="empty">No models trained yet — click “Retrain All”.</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Auto-tuner -->
  <div class="card">
    <div class="card-header"><h2>⚙️ Auto-Tuner — Optuna Parameter Optimisation</h2></div>
    <div style="padding:12px 16px;font-size:12px;color:var(--muted);border-bottom:1px solid var(--border)">
      Bayesian search over MIN_SIGNAL_SCORE, ATR multipliers, ADX &amp; volume thresholds — maximising backtested Sharpe ratio.
    </div>
    <div class="bt-form">
      <label>Symbol
        <select id="tunerSymbol" style="width:160px"></select>
      </label>
      <label>Days
        <input id="tunerDays" type="number" value="90" min="14" max="365"/>
      </label>
      <label>Trials
        <input id="tunerTrials" type="number" value="50" min="10" max="200"/>
      </label>
      <button class="btn btn-primary" id="tunerRunBtn" onclick="runTuner()" style="margin-bottom:1px">▶ Run Tuner</button>
    </div>
    <div class="bt-progress" id="tunerProgress">
      <span class="spinner"></span><span id="tunerProgressText">Running…</span>
    </div>
    <div id="tunerResult" style="display:none;padding:16px">
      <div class="bt-summary" id="tunerSummary"></div>
      <div style="margin-top:12px">
        <h3 style="font-size:13px;margin-bottom:8px">Suggested parameter changes</h3>
        <pre id="tunerImprovement" style="background:var(--bg3);padding:12px;border-radius:6px;font-size:12px;white-space:pre-wrap;color:var(--text)"></pre>
        <button class="btn btn-success" onclick="applyTuner()" style="margin-top:10px">✓ Apply These Parameters</button>
        <span class="last-update" id="tunerApplied" style="margin-left:10px"></span>
      </div>
    </div>
  </div>

</main>
</div><!-- /tab-ml -->

<!-- ═══════════════════════════════ SMA TREND TAB ════════════════════════════ -->
<div id="tab-sma" class="tab-panel">
<main class="main">
  <div class="card">
    <div class="card-header">
      <h2>📈 SMA Trend Strategy — Live Status</h2>
      <span class="last-update" id="smaMode">—</span>
    </div>
    <div style="padding:10px 16px;font-size:12px;color:var(--muted)">
      Holds a coin only when its daily close is above its 200-day SMA
      <b>and</b> the weekly trend (30-week SMA) is up. Otherwise → cash.
      Honest note: this reduces drawdown, it does not beat buy-and-hold on return.
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr><th>Symbol</th><th>Price</th><th>200d SMA</th><th>vs Daily</th><th>Weekly</th><th>Signal</th></tr></thead>
        <tbody id="smaBody"><tr><td colspan="6" class="empty">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-header">
      <h2>💵 SMA Paper / Live Activity</h2>
      <span class="last-update" id="smaPaperUpd">—</span>
    </div>
    <div style="padding:8px 16px;font-size:12px;color:var(--muted)">
      Live equity &amp; P&amp;L from a running <code>sma_bot.py</code> loop
      (set <code>SMA_ALERT_ONLY=False</code>). Reads <code>sma_state.json</code>.
    </div>
    <div class="bt-summary" id="smaPaperCards">
      <div class="empty" style="padding:10px 16px">sma_bot.py not running in trade mode yet.</div>
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr><th>Holding</th><th>Units</th><th>Entry</th><th>Price</th><th>Value</th><th>P&amp;L</th></tr></thead>
        <tbody id="smaPaperBody"><tr><td colspan="6" class="empty">—</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Backtest (daily, fees included)</h2></div>
    <div class="bt-form">
      <label>Years <input id="smaYears" type="number" value="3" min="1" max="6" step="0.5"/></label>
      <label>Capital (USD) <input id="smaCap" type="number" value="1000" min="100"/></label>
      <button class="btn btn-primary" id="smaBtRun" onclick="runSmaBacktest()" style="margin-bottom:1px">▶ Run Backtest</button>
      <span class="last-update" id="smaBtProg"></span>
    </div>
    <div id="smaBtResult" style="display:none">
      <div class="bt-summary" id="smaBtCards"></div>
      <div class="chart-wrap" style="height:240px"><canvas id="smaEquityChart"></canvas></div>
    </div>
  </div>
</main>
</div><!-- /tab-sma -->

<script>
// ── Tab switching ──────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['live','backtest','universe','ml','sma'][i]===name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if (name === 'backtest') loadBacktestState();
  if (name === 'universe') loadUniverseState();
  if (name === 'ml')       loadMLState();
  if (name === 'sma')    { loadSmaStatus(); loadSmaPaper(); }
}

// ── SMA Trend tab ───────────────────────────────────────────────────────────
let smaEquityChart = null, smaBtPoll = null;
async function loadSmaStatus() {
  try {
    const r = await fetch('/api/sma/status'); const d = await r.json();
    document.getElementById('smaMode').textContent =
      `mode: ${d.mode} · daily ${d.period}d` + (d.weekly_gate ? ` + weekly ${d.weekly_period}wk gate` : '');
    const tb = document.getElementById('smaBody');
    if (!d.rows || !d.rows.length) { tb.innerHTML = '<tr><td colspan="6" class="empty">No data</td></tr>'; return; }
    tb.innerHTML = d.rows.map(x => {
      const sig = x.hold ? '<span class="badge badge-long">🟢 HOLD</span>'
                         : '<span class="badge badge-short">🔴 CASH</span>';
      const wk  = x.weekly_up ? `<span class="pos">↑ ${fmt(x.weekly_pct,1)}%</span>`
                              : `<span class="neg">↓ ${fmt(x.weekly_pct,1)}%</span>`;
      return `<tr><td><strong>${x.symbol}</strong></td>
        <td class="mono">$${fmt(x.close, x.close>100?2:4)}</td>
        <td class="mono">$${fmt(x.sma, x.sma>100?2:4)}</td>
        <td class="mono ${x.daily_up?'pos':'neg'}">${x.pct>=0?'+':''}${fmt(x.pct,1)}%</td>
        <td class="mono">${wk}</td><td>${sig}</td></tr>`;
    }).join('');
  } catch(e) { document.getElementById('smaBody').innerHTML='<tr><td colspan="6" class="empty">Error loading</td></tr>'; }
}

async function loadSmaPaper() {
  try {
    const r = await fetch('/api/sma/paper'); const d = await r.json();
    const cards = document.getElementById('smaPaperCards');
    const body  = document.getElementById('smaPaperBody');
    const upd   = document.getElementById('smaPaperUpd');
    if (!d.running) {
      upd.textContent = '';
      cards.innerHTML = `<div class="empty" style="padding:10px 16px">${d.msg || 'sma_bot.py not running in trade mode'}</div>`;
      body.innerHTML  = '<tr><td colspan="6" class="empty">—</td></tr>';
      return;
    }
    upd.textContent = 'mode: ' + d.mode + (d.last_check ? ' · ' + new Date(d.last_check).toLocaleString() : '');
    const pcls = d.pnl >= 0 ? 'pos' : 'neg', sgn = d.pnl >= 0 ? '+' : '';
    cards.innerHTML =
      `<div class="bt-card"><div class="bt-card-label">Equity</div><div class="bt-card-val">$${fmt(d.equity,2)}</div></div>
       <div class="bt-card"><div class="bt-card-label">Cash</div><div class="bt-card-val">$${fmt(d.cash,2)}</div></div>
       <div class="bt-card"><div class="bt-card-label">P&L</div><div class="bt-card-val ${pcls}">${sgn}$${fmt(d.pnl,2)} (${sgn}${fmt(d.pnl_pct,1)}%)</div></div>
       <div class="bt-card"><div class="bt-card-label">Start</div><div class="bt-card-val">$${fmt(d.start_capital,0)}</div></div>`;
    if (!d.positions || !d.positions.length) {
      body.innerHTML = '<tr><td colspan="6" class="empty">All cash — no holdings</td></tr>';
    } else {
      body.innerHTML = d.positions.map(p => {
        const cls = (p.pnl||0) >= 0 ? 'pos' : 'neg', s = (p.pnl||0) >= 0 ? '+' : '';
        return `<tr><td><strong>${p.symbol}</strong></td>
          <td class="mono">${p.units!=null?fmt(p.units,4):'—'}</td>
          <td class="mono">${p.entry!=null?'$'+fmt(p.entry,2):'—'}</td>
          <td class="mono">${p.price!=null?'$'+fmt(p.price,2):'—'}</td>
          <td class="mono">${p.value!=null?'$'+fmt(p.value,2):'—'}</td>
          <td class="mono ${cls}">${p.pnl!=null?s+'$'+fmt(p.pnl,2):'—'}</td></tr>`;
      }).join('');
    }
  } catch(e) { /* leave placeholder */ }
}

async function runSmaBacktest() {
  const years = parseFloat(document.getElementById('smaYears').value)||3;
  const capital = parseFloat(document.getElementById('smaCap').value)||1000;
  document.getElementById('smaBtRun').disabled = true;
  document.getElementById('smaBtProg').textContent = 'Running…';
  await fetch('/api/sma/backtest',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({years,capital})});
  if (smaBtPoll) clearInterval(smaBtPoll);
  smaBtPoll = setInterval(pollSmaBacktest, 2000);
}
async function pollSmaBacktest() {
  const r = await fetch('/api/sma/backtest_status'); const d = await r.json();
  if (d.status === 'running') { document.getElementById('smaBtProg').textContent = 'Running…'; return; }
  clearInterval(smaBtPoll); smaBtPoll = null;
  document.getElementById('smaBtRun').disabled = false;
  document.getElementById('smaBtProg').textContent = '';
  if (d.status === 'done' && d.result && !d.result.error) renderSmaBacktest(d.result);
  else document.getElementById('smaBtProg').textContent = 'Error: ' + (d.result?.error||'failed');
}
function renderSmaBacktest(r) {
  document.getElementById('smaBtResult').style.display = 'block';
  const beat = r.total_return >= r.bh_return;
  const ddBetter = r.max_drawdown > r.bh_drawdown;  // less negative = better
  document.getElementById('smaBtCards').innerHTML = `
    <div class="bt-stat"><div class="l">Window</div><div class="v" style="font-size:14px">${r.years}y (${r.symbols.length} coins)</div></div>
    <div class="bt-stat"><div class="l">Total Return</div><div class="v ${r.total_return>=0?'pos':'neg'}">${r.total_return>=0?'+':''}${r.total_return}%</div></div>
    <div class="bt-stat"><div class="l">CAGR</div><div class="v">${r.cagr>=0?'+':''}${r.cagr}%</div></div>
    <div class="bt-stat"><div class="l">Max Drawdown</div><div class="v ${ddBetter?'pos':'neg'}">${r.max_drawdown}%</div></div>
    <div class="bt-stat"><div class="l">Calmar</div><div class="v">${r.calmar}</div></div>
    <div class="bt-stat"><div class="l">Sharpe</div><div class="v">${r.sharpe}</div></div>
    <div class="bt-stat"><div class="l">Trades</div><div class="v">${r.trades}</div></div>
    <div class="bt-stat"><div class="l">vs Hold BTC</div><div class="v" style="font-size:13px">${r.bh_return}% / DD ${r.bh_drawdown}%</div></div>`;
  if (smaEquityChart) smaEquityChart.destroy();
  const ctx = document.getElementById('smaEquityChart').getContext('2d');
  smaEquityChart = new Chart(ctx, {
    type:'line',
    data:{labels:r.curve.map(p=>p.t),datasets:[{label:'Equity',data:r.curve.map(p=>p.v),
      borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,.08)',borderWidth:2,pointRadius:0,fill:true,tension:.2}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{display:false}},
              y:{ticks:{color:'#8b949e',callback:v=>'$'+v},grid:{color:'#21262d'}}}}
  });
}

// ── Shared helpers ─────────────────────────────────────────────────────────────
let tradePage = 1, totalTrades = 0;
const PAGE_SIZE = 50;
let equityChart = null, btBarChart = null, btEquityChart = null;
let btResults = [], btPolling = null;
let botPaused = false;

function fmt(n, d=2)  { return Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}); }
function fmtPnl(n)    { const v=Number(n),s=(v>=0?'+':'')+fmt(v); return `<span class="${v>0?'pos':v<0?'neg':'neu'}">${s}</span>`; }
function fmtTime(iso) { return iso?(iso.replace('T',' ').slice(0,19)):'—'; }

function scoreFill(score) {
  const pct=(score/10*100).toFixed(1), c=score>=7?'#3fb950':score>=4?'#d29922':'#8b949e';
  return `<div class="score-wrap"><div class="score-bar"><div class="score-fill" style="width:${pct}%;background:${c}"></div></div><span>${fmt(score,1)}</span></div>`;
}
function dirBadge(dir) {
  if(dir==='long')  return '<span class="badge badge-long">LONG</span>';
  if(dir==='short') return '<span class="badge badge-short">SHORT</span>';
  return '<span class="badge badge-neutral">—</span>';
}
function sentBadge(label) {
  if(!label||label==='neutral') return `<span style="color:var(--muted)">neutral</span>`;
  return `<span style="color:${label==='bullish'?'var(--green)':'var(--red)'}">${label}</span>`;
}

// ── Live chart ─────────────────────────────────────────────────────────────────
function initEquityChart() {
  const ctx = document.getElementById('equityChart').getContext('2d');
  equityChart = new Chart(ctx, {
    type:'line',
    data:{labels:[],datasets:[{label:'Portfolio USDT',data:[],borderColor:'#58a6ff',
      backgroundColor:'rgba(88,166,255,.08)',borderWidth:2,pointRadius:0,fill:true,tension:.3}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'$'+c.parsed.y.toFixed(2)}}},
      scales:{x:{display:false},y:{grid:{color:'#21262d'},ticks:{color:'#8b949e',callback:v=>'$'+v}}}}
  });
}

// ── Live updates ───────────────────────────────────────────────────────────────
function updateStatus(data) {
  const dot=document.getElementById('statusDot'), text=document.getElementById('statusText');
  const s=data.bot_status||'stopped';
  dot.className='dot '+s; text.textContent=s.charAt(0).toUpperCase()+s.slice(1);
  botPaused=(s==='paused');
  const btn=document.getElementById('pauseBtn');
  btn.textContent=botPaused?'Resume':'Pause';
  btn.className=botPaused?'btn btn-success':'btn btn-danger';
  document.getElementById('lastUpdate').textContent='Updated '+new Date().toLocaleTimeString();

  // API-key auth banner
  const banner = document.getElementById('authBanner');
  if (data.auth_ok === false) {
    banner.style.display = 'block';
    banner.style.background = 'rgba(248,81,73,.15)';
    banner.style.color = 'var(--red)';
    banner.style.borderBottom = '1px solid var(--red)';
    banner.innerHTML = '⚠️ API key not working — paper trading is fine, but fix the key before going live. '
      + '<span style="font-weight:400;color:var(--muted)">' + (data.auth_msg||'') + '</span>';
  } else if (data.auth_ok === true) {
    banner.style.display = 'none';
  }
}
function updateStats(data) {
  const cap=data.capital_usdt||0, start=data.capital_start||cap;
  const pnl=cap-start, pct=start>0?(pnl/start*100):0;
  document.getElementById('capital').textContent='$'+fmt(cap);
  document.getElementById('capitalChange').innerHTML=fmtPnl(pnl)+' from start';
  document.getElementById('totalPnl').innerHTML=fmtPnl(pnl);
  document.getElementById('totalPnlPct').innerHTML=`<span class="${pnl>=0?'pos':'neg'}">${pnl>=0?'+':''}${fmt(pct,2)}%</span>`;
  document.getElementById('scanCount').textContent=data.scan_count||0;
  document.getElementById('lastScan').textContent=data.last_scan?fmtTime(data.last_scan)+' UTC':'—';
  const trades=data.trade_history||[], wins=trades.filter(t=>t.action==='close'&&(t.pnl||0)>0);
  const closes=trades.filter(t=>t.action==='close');
  const wr=closes.length>0?(wins.length/closes.length*100).toFixed(1):'—';
  document.getElementById('tradeCount').textContent=closes.length;
  document.getElementById('winRate').textContent=wr!=='—'?`${wr}% win rate`:'no closed trades';
  const pos=data.open_positions||[], unreal=pos.reduce((s,p)=>s+(p.pnl||0),0);
  document.getElementById('openCount').textContent=pos.length;
  document.getElementById('openUnreal').innerHTML=fmtPnl(unreal)+' unrealised';
}
function updateEquityChart(data) {
  const curve=data.equity_curve||[];
  if(!equityChart||!curve.length) return;
  equityChart.data.labels=curve.map(p=>fmtTime(p.time).slice(11));
  equityChart.data.datasets[0].data=curve.map(p=>p.value);
  const first=curve[0]?.value||0, last=curve[curve.length-1]?.value||0;
  equityChart.data.datasets[0].borderColor=last>=first?'#3fb950':'#f85149';
  equityChart.data.datasets[0].backgroundColor=last>=first?'rgba(63,185,80,.07)':'rgba(248,81,73,.07)';
  equityChart.update('none');
}
function updatePositions(data) {
  const positions=data.open_positions||[], tbody=document.getElementById('posBody');
  if(!positions.length){tbody.innerHTML='<tr><td colspan="6" class="empty">No open positions</td></tr>';return;}
  tbody.innerHTML=positions.map(p=>`<tr class="clickable" onclick="openSymbolDrawer('${p.symbol}')">
    <td><strong>${p.symbol}</strong></td><td>${dirBadge(p.side)}</td>
    <td class="mono">$${fmt(p.entry,4)}</td><td class="mono neg">$${fmt(p.sl,4)}</td>
    <td class="mono pos">$${fmt(p.tp,4)}</td><td class="mono">${fmtPnl(p.pnl)}</td>
  </tr>`).join('');
}
function updateSignals(data) {
  const signals=data.signals||[], tbody=document.getElementById('sigBody'), note=document.getElementById('sigNote');
  if(!signals.length){tbody.innerHTML='<tr><td colspan="9" class="empty">Waiting for first scan…</td></tr>';return;}
  note.textContent=signals.length+' symbols';
  const ticker=document.getElementById('tickerBar');
  ticker.innerHTML=signals.filter(s=>s.direction!=='neutral').slice(0,12).map(s=>{
    const c=s.direction==='long'?'var(--green)':'var(--red)';
    return `<span><strong style="color:${c}">${s.symbol}</strong> <span>${s.direction.toUpperCase()} ${fmt(s.score,1)}/10</span></span>`;
  }).join(' · ')||'<span style="color:var(--muted)">No actionable signals</span>';
  tbody.innerHTML=signals.map(s=>{
    const topKeys=Object.entries(s.components||{}).filter(([,v])=>Math.abs(v)>1).map(([k])=>k.replace(/_/g,' ')).slice(0,3).join(', ')||'—';
    const patterns=Object.keys(s.candle_patterns||{}).map(k=>k.replace(/_/g,' ')).join(', ')||'—';
    const hasPat = patterns !== '—';
    return `<tr class="clickable" onclick="openSymbolDrawer('${s.symbol}')">
      <td><strong>${s.symbol}</strong></td><td>${dirBadge(s.direction)}</td>
      <td>${scoreFill(s.score)}</td><td class="mono">$${fmt(s.entry_price,4)}</td>
      <td class="mono ${s.stop_loss?'neg':''}">$${s.stop_loss?fmt(s.stop_loss,4):'—'}</td>
      <td class="mono ${s.take_profit?'pos':''}">$${s.take_profit?fmt(s.take_profit,4):'—'}</td>
      <td>${s.risk_reward||'—'}</td><td>${sentBadge(s.sentiment?.label)}</td>
      <td style="color:var(--muted);font-size:12px">${topKeys}</td>
      <td style="font-size:11px;color:${hasPat?'var(--yellow)':'var(--muted)'}">${patterns}</td></tr>`;
  }).join('');
}
function updateTrades(trades, total) {
  totalTrades=total;
  const tbody=document.getElementById('tradeBody'), pag=document.getElementById('tradePagination');
  const pages=Math.ceil(total/PAGE_SIZE);
  pag.textContent=total>0?`${total} trades · page ${tradePage}/${pages}`:'';
  document.getElementById('prevBtn').disabled=tradePage<=1;
  document.getElementById('nextBtn').disabled=tradePage>=pages;
  if(!trades.length){tbody.innerHTML='<tr><td colspan="9" class="empty">No trades yet</td></tr>';return;}
  tbody.innerHTML=trades.map(t=>{
    const isClose=t.action==='close';
    const reasonBadge=t.reason==='take_profit'?'<span class="badge badge-tp">TP ✓</span>':
      t.reason==='stop_loss'?'<span class="badge badge-sl">SL ✗</span>':
      `<span class="badge badge-neutral">${t.reason||'—'}</span>`;
    return `<tr class="clickable" onclick="openSymbolDrawer('${t.symbol}')">
      <td style="color:var(--muted)">${fmtTime(t.time)}</td>
      <td><strong>${t.symbol}</strong></td><td>${dirBadge(t.side||(t.action==='open'?'—':'—'))}</td>
      <td><span class="badge ${isClose?'badge-short':'badge-long'}">${(t.action||'').toUpperCase()}</span></td>
      <td class="mono">$${t.entry?fmt(t.entry,4):'—'}</td>
      <td class="mono">$${t.price?fmt(t.price,4):(t.tp?fmt(t.tp,4):'—')}</td>
      <td>${t.score?fmt(t.score,1):'—'}</td>
      <td class="mono">${t.pnl!==undefined?fmtPnl(t.pnl):'—'}</td>
      <td>${reasonBadge}</td></tr>`;
  }).join('');
}
async function fetchState() {
  try {
    const [sr,tr]=await Promise.all([fetch('/api/state'),fetch(`/api/trades?page=${tradePage}&size=${PAGE_SIZE}`)]);
    const data=await sr.json(), tdata=await tr.json();
    updateStatus(data); updateStats(data); updateEquityChart(data);
    updatePositions(data); updateSignals(data);
    updateTrades(tdata.trades||[],tdata.total||0);
    updateBreaker(data.circuit_breaker);
  } catch(e){
    document.getElementById('statusText').textContent='Error';
    document.getElementById('statusDot').className='dot stopped';
  }
}
function updateBreaker(cb){
  const banner=document.getElementById('cbBanner');
  if(!banner) return;
  if(cb && cb.status==='tripped'){
    banner.style.display='block';
    let extra='';
    if(cb.cooldown_hours) extra=` · auto-resumes in ≤${cb.cooldown_hours}h`;
    document.getElementById('cbBannerText').textContent=
      'CIRCUIT BREAKER TRIPPED — new trades halted. Reason: '+(cb.reason||'—')+extra;
  } else {
    banner.style.display='none';
  }
}
async function resetBreaker(){
  await fetch('/api/breaker/reset',{method:'POST'});
  fetchState();
}
function changePage(d){
  const pages=Math.ceil(totalTrades/PAGE_SIZE);
  tradePage=Math.max(1,Math.min(pages,tradePage+d));
  fetchState();
}
async function toggleBot(){
  const action=botPaused?'resume':'pause';
  await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
  fetchState();
}

// ── Backtest ───────────────────────────────────────────────────────────────────
async function populateBtSymbols() {
  try {
    // Populate from dynamic universe first, fall back to signals
    const [ur, sr] = await Promise.all([fetch('/api/universe'), fetch('/api/state')]);
    const ud = await ur.json();
    const sd = await sr.json();
    const sel = document.getElementById('btSymbols');

    // Combine universe symbols + any signal symbols not already there
    const allSymbols = new Set([
      ...(ud.symbols || []),
      ...(sd.signals || []).map(s => s.symbol),
    ]);

    allSymbols.forEach(sym => {
      const exists = [...sel.options].some(o => o.value === sym);
      if (!exists) {
        const o = document.createElement('option');
        o.value = sym; o.textContent = sym;
        sel.appendChild(o);
      }
    });
  } catch(_){}
}

async function runBacktest() {
  const sel=document.getElementById('btSymbols');
  const chosen=[...sel.selectedOptions].map(o=>o.value).filter(v=>v!=='__all__');
  const days=parseInt(document.getElementById('btDays').value)||90;
  const capital=parseFloat(document.getElementById('btCapital').value)||1000;
  document.getElementById('btRunBtn').disabled=true;
  document.getElementById('btProgressText').textContent='Starting…';
  document.getElementById('btProgress').classList.add('visible');
  document.getElementById('btSummarySection').style.display='none';
  document.getElementById('btResultsRow').style.display='none';
  document.getElementById('btTradeLogCard').style.display='none';

  await fetch('/api/backtest/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({symbols:chosen,days,capital})});

  // Poll for completion
  if(btPolling) clearInterval(btPolling);
  btPolling=setInterval(pollBacktest,2000);
}

async function pollBacktest() {
  try {
    const r=await fetch('/api/backtest');
    const d=await r.json();
    document.getElementById('btProgressText').textContent=d.progress||'Running…';
    if(d.last_run) document.getElementById('btLastRun').textContent='Last run: '+fmtTime(d.last_run);

    if(d.status==='done'||d.status==='error'){
      clearInterval(btPolling); btPolling=null;
      document.getElementById('btRunBtn').disabled=false;
      document.getElementById('btProgress').classList.remove('visible');
      if(d.status==='done') renderBtResults(d.results||[]);
    }
  } catch(_){}
}

async function loadBacktestState() {
  await populateBtSymbols();
  const r=await fetch('/api/backtest');
  const d=await r.json();
  if(d.last_run) document.getElementById('btLastRun').textContent='Last run: '+fmtTime(d.last_run);
  if(d.status==='running'){
    document.getElementById('btRunBtn').disabled=true;
    document.getElementById('btProgress').classList.add('visible');
    if(btPolling) clearInterval(btPolling);
    btPolling=setInterval(pollBacktest,2000);
  } else if((d.status==='done')&&d.results?.length){
    renderBtResults(d.results);
  }
}

function renderBtResults(results) {
  btResults=results;
  if(!results.length) return;

  // ── Aggregate summary ──────────────────────────────────────────────────
  const total_pnl    = results.reduce((s,r)=>s+(r.total_pnl||0),0);
  const avg_return   = results.reduce((s,r)=>s+(r.return_pct||0),0)/results.length;
  const profitable   = results.filter(r=>(r.return_pct||0)>0).length;
  const avg_pf       = results.reduce((s,r)=>s+(r.profit_factor||0),0)/results.length;
  const avg_wr       = results.reduce((s,r)=>s+(r.win_rate||0),0)/results.length;
  const best         = results[0];
  const worst        = results[results.length-1];

  document.getElementById('btSummarySection').style.display='block';
  document.getElementById('btSummary').innerHTML = `
    <div class="bt-stat"><div class="l">Total P&L</div><div class="v ${total_pnl>=0?'pos':'neg'}">${total_pnl>=0?'+':''}$${fmt(Math.abs(total_pnl))}</div></div>
    <div class="bt-stat"><div class="l">Avg Return</div><div class="v ${avg_return>=0?'pos':'neg'}">${avg_return>=0?'+':''}${fmt(avg_return)}%</div></div>
    <div class="bt-stat"><div class="l">Profitable</div><div class="v">${profitable} / ${results.length}</div></div>
    <div class="bt-stat"><div class="l">Avg Win Rate</div><div class="v">${fmt(avg_wr,1)}%</div></div>
    <div class="bt-stat"><div class="l">Avg Profit Factor</div><div class="v ${avg_pf>=1?'pos':'neg'}">${fmt(avg_pf,2)}</div></div>
    <div class="bt-stat"><div class="l">Best Symbol</div><div class="v pos">${best?.symbol||'—'}</div></div>
    <div class="bt-stat"><div class="l">Worst Symbol</div><div class="v neg">${worst?.symbol||'—'}</div></div>
  `;

  // ── Per-symbol table ───────────────────────────────────────────────────
  document.getElementById('btResultsRow').style.display='grid';
  const tbody=document.getElementById('btBody');
  tbody.innerHTML=results.map(r=>{
    const ret=r.return_pct||0, pf=r.profit_factor||0;
    const pfPct=Math.min(pf/3*100,100).toFixed(1), pfColor=pf>=1?'#3fb950':'#f85149';
    return `<tr style="cursor:pointer" onclick="showBtTradeLog('${r.symbol}')">
      <td><strong>${r.symbol}</strong></td>
      <td class="mono ${ret>=0?'pos':'neg'}">${ret>=0?'+':''}${fmt(ret,2)}%</td>
      <td class="mono ${(r.total_pnl||0)>=0?'pos':'neg'}">${(r.total_pnl||0)>=0?'+':''}$${fmt(Math.abs(r.total_pnl||0))}</td>
      <td>${r.trades||0}</td>
      <td>${fmt(r.win_rate||0,1)}%</td>
      <td>
        <div class="pf-bar-wrap">
          <div class="pf-bar"><div class="pf-fill" style="width:${pfPct}%;background:${pfColor}"></div></div>
          <span class="${pf>=1?'pos':'neg'}">${fmt(pf,2)}</span>
        </div>
      </td>
      <td class="pos mono">+$${fmt(r.avg_win||0)}</td>
      <td class="neg mono">-$${fmt(Math.abs(r.avg_loss||0))}</td>
      <td class="neg mono">${fmt(r.max_dd||0,1)}%</td>
      <td class="${(r.sharpe||0)>=0?'pos':'neg'} mono">${fmt(r.sharpe||0,2)}</td>
    </tr>`;
  }).join('');

  // ── Bar chart ──────────────────────────────────────────────────────────
  if(btBarChart) { btBarChart.destroy(); btBarChart=null; }
  const ctx=document.getElementById('btBarChart').getContext('2d');
  const sorted=[...results].sort((a,b)=>b.return_pct-a.return_pct);
  btBarChart=new Chart(ctx,{
    type:'bar',
    data:{
      labels:sorted.map(r=>r.symbol.replace('/USD','')),
      datasets:[{
        label:'Return %',
        data:sorted.map(r=>r.return_pct||0),
        backgroundColor:sorted.map(r=>(r.return_pct||0)>=0?'rgba(63,185,80,.7)':'rgba(248,81,73,.7)'),
        borderColor:sorted.map(r=>(r.return_pct||0)>=0?'#3fb950':'#f85149'),
        borderWidth:1,borderRadius:4,
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.parsed.y.toFixed(2)+'%'}}},
      scales:{
        x:{ticks:{color:'#8b949e'},grid:{display:false}},
        y:{ticks:{color:'#8b949e',callback:v=>v+'%'},grid:{color:'#21262d'},
           title:{display:true,text:'Return %',color:'#8b949e'}}
      }
    }
  });
}

function showBtTradeLog(symbol) {
  const r=btResults.find(x=>x.symbol===symbol);
  if(!r) return;
  document.getElementById('btTradeLogSymbol').textContent=symbol;
  document.getElementById('btTradeLogCard').style.display='block';
  document.getElementById('btTradeLogCard').scrollIntoView({behavior:'smooth',block:'start'});

  // Equity curve for this symbol
  const trades=r.trade_list||[];
  const equityCurve=trades.filter(t=>t.capital!=null).map(t=>({
    t: fmtTime(String(t.exit_time)).slice(5,16), v: t.capital
  }));

  if(btEquityChart){btEquityChart.destroy();btEquityChart=null;}
  const ctx2=document.getElementById('btEquityChart').getContext('2d');
  const first=equityCurve[0]?.v||0, last=equityCurve[equityCurve.length-1]?.v||0;
  btEquityChart=new Chart(ctx2,{
    type:'line',
    data:{labels:equityCurve.map(p=>p.t),datasets:[{
      label:'Capital',data:equityCurve.map(p=>p.v),
      borderColor:last>=first?'#3fb950':'#f85149',
      backgroundColor:last>=first?'rgba(63,185,80,.07)':'rgba(248,81,73,.07)',
      borderWidth:2,pointRadius:2,fill:true,tension:.3
    }]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{display:false}},
              y:{ticks:{color:'#8b949e',callback:v=>'$'+v},grid:{color:'#21262d'}}}}
  });

  // Trade log table
  const tbody=document.getElementById('btTradeLog');
  tbody.innerHTML=trades.map(t=>{
    const pnl=t.pnl||0;
    const reasonBadge=t.reason==='TP'?'<span class="badge badge-tp">TP ✓</span>':
      t.reason==='SL'?'<span class="badge badge-sl">SL ✗</span>':
      `<span class="badge badge-neutral">${t.reason}</span>`;
    return `<tr>
      <td style="color:var(--muted)">${fmtTime(String(t.exit_time)).slice(0,16)}</td>
      <td>${dirBadge(t.side)}</td>
      <td class="mono">$${fmt(t.entry,2)}</td>
      <td class="mono">$${fmt(t.exit,2)}</td>
      <td>${reasonBadge}</td>
      <td class="mono ${pnl>0?'pos':'neg'}">${pnl>=0?'+':''}$${fmt(Math.abs(pnl),2)}</td>
    </tr>`;
  }).join('');
}
function closeBtTradeLog(){
  document.getElementById('btTradeLogCard').style.display='none';
  if(btEquityChart){btEquityChart.destroy();btEquityChart=null;}
}

// ── Universe tab ──────────────────────────────────────────────────────────────
async function loadUniverseState() {
  try {
    const r = await fetch('/api/universe');
    const d = await r.json();
    renderUniverse(d);
  } catch(e) { console.error('Universe fetch failed', e); }
}

function renderUniverse(d) {
  const symbols = d.symbols || [];
  const log     = (d.log || []).slice().reverse();   // newest first

  // Current symbols chips
  const container = document.getElementById('univCurrentSymbols');
  if (!symbols.length) {
    container.innerHTML = '<span style="color:var(--muted)">No symbols yet — bot not started</span>';
  } else {
    const anchors = new Set(['BTC/USD','ETH/USD','SOL/USD']);
    container.innerHTML = symbols.map(s => {
      const isAnchor = anchors.has(s);
      const color = isAnchor ? 'var(--blue)' : 'var(--green)';
      const border = isAnchor ? 'var(--blue)' : 'var(--border)';
      return `<span style="padding:5px 12px;border-radius:20px;border:1px solid ${border};
        color:${color};font-size:12px;font-weight:600;background:var(--bg3)">
        ${s.replace('/USD','')}${isAnchor ? ' ⚓' : ''}
      </span>`;
    }).join('');
  }

  // Refresh log table
  const tbody = document.getElementById('univLogBody');
  if (!log.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No refresh history yet</td></tr>';
    return;
  }
  tbody.innerHTML = log.map(r => {
    const added   = (r.added   || []).map(s => `<span style="color:var(--green)">${s.replace('/USD','')}</span>`).join(' ') || '—';
    const removed = (r.removed || []).map(s => `<span style="color:var(--red)">${s.replace('/USD','')}</span>`).join(' ') || '—';
    const syms    = (r.symbols || []).map(s => s.replace('/USD','')).join(', ');
    return `<tr>
      <td style="color:var(--muted)">${r.time || '—'}</td>
      <td>${r.total || 0}</td>
      <td>${added}</td>
      <td>${removed}</td>
      <td style="font-size:11px;color:var(--muted)">${syms}</td>
    </tr>`;
  }).join('');
}

async function forceRefreshUniverse() {
  const btn = event.target;
  btn.disabled = true; btn.textContent = '⟳ Refreshing…';
  await fetch('/api/universe/refresh', {method:'POST'});
  setTimeout(async () => {
    await loadUniverseState();
    btn.disabled = false; btn.textContent = '⟳ Refresh Now';
  }, 8000);
}

// ── Symbol Detail Drawer ──────────────────────────────────────────────────────
let _drawerSymbol = null, _drawerTf = '1h';
let _lwCandle = null, _lwVolume = null, _lwRsi = null, _lwMacd = null;
const _chartBg = '#0d1117', _gridColor = '#21262d', _textColor = '#8b949e';

const _lwOpts = {
  layout: { background: { color: _chartBg }, textColor: _textColor },
  grid: { vertLines: { color: _gridColor }, horzLines: { color: _gridColor } },
  timeScale: { borderColor: _gridColor, timeVisible: true, secondsVisible: false },
  crosshair: { mode: 1 },
  rightPriceScale: { borderColor: _gridColor },
};

function _destroyCharts() {
  [_lwCandle, _lwVolume, _lwRsi, _lwMacd].forEach(c => { try { c && c.remove(); } catch(_){} });
  _lwCandle = _lwVolume = _lwRsi = _lwMacd = null;
  ['candleChart','volumeChart','rsiChart','macdChart'].forEach(id => {
    const el = document.getElementById(id); if (el) el.innerHTML = '';
  });
}

async function openSymbolDrawer(symbol) {
  _drawerSymbol = symbol;
  document.getElementById('drawerSymbol').textContent = symbol;
  document.getElementById('drawerPrice').textContent = '…';
  document.getElementById('drawerChange').textContent = '';
  document.getElementById('drawerCards').innerHTML = '';
  document.getElementById('drawerTradeBody').innerHTML = '<tr><td colspan="7" class="empty">Loading…</td></tr>';

  document.getElementById('symbolDrawer').classList.add('open');
  document.getElementById('drawerOverlay').classList.add('open');
  document.body.style.overflow = 'hidden';

  await _loadDrawerData(symbol, _drawerTf);
}

async function _loadDrawerData(symbol, tf) {
  _destroyCharts();

  // Fetch candles + symbol detail in parallel
  const slug = symbol.replace('/', '-');
  const [cRes, dRes] = await Promise.all([
    fetch(`/api/candles/${slug}?tf=${tf}`),
    fetch(`/api/symbol/${slug}`),
  ]);
  const cData = await cRes.json();
  const dData = await dRes.json();

  if (cData.error) {
    document.getElementById('candleChart').innerHTML =
      `<div class="empty">${cData.error}</div>`;
    return;
  }

  // ── Price header ──────────────────────────────────────────────────────────
  const price  = cData.last_price || 0;
  const change = cData.price_change_pct || 0;
  document.getElementById('drawerPrice').textContent = '$' + fmt(price, price > 100 ? 2 : 4);
  const chEl = document.getElementById('drawerChange');
  chEl.textContent = (change >= 0 ? '+' : '') + fmt(change, 2) + '%';
  chEl.style.background = change >= 0 ? 'rgba(63,185,80,.15)' : 'rgba(248,81,73,.15)';
  chEl.style.color = change >= 0 ? 'var(--green)' : 'var(--red)';

  // ── Info cards ────────────────────────────────────────────────────────────
  const sig = dData.signal, pos = dData.position;
  let cards = '';
  if (sig) {
    cards += `
      <div class="info-card">
        <div class="label">Current Signal</div>
        <div class="value">${dirBadge(sig.direction)} <span style="font-size:13px">${fmt(sig.score,1)}/10</span></div>
      </div>
      <div class="info-card">
        <div class="label">R:R Ratio</div>
        <div class="value">${sig.risk_reward || '—'}</div>
      </div>
      <div class="info-card">
        <div class="label">Stop Loss</div>
        <div class="value neg">$${sig.stop_loss ? fmt(sig.stop_loss, price>100?2:4) : '—'}</div>
      </div>
      <div class="info-card">
        <div class="label">Take Profit</div>
        <div class="value pos">$${sig.take_profit ? fmt(sig.take_profit, price>100?2:4) : '—'}</div>
      </div>`;
  }
  if (pos) {
    cards += `
      <div class="info-card" style="border-color:${pos.pnl>=0?'var(--green)':'var(--red)'}">
        <div class="label">Open Position</div>
        <div class="value">${dirBadge(pos.side)} @ $${fmt(pos.entry, price>100?2:4)}</div>
      </div>
      <div class="info-card">
        <div class="label">Unrealised P&L</div>
        <div class="value ${pos.pnl>=0?'pos':'neg'}">${fmtPnl(pos.pnl)}</div>
      </div>`;
  }
  // Candle patterns card
  const patterns = Object.keys(sig?.candle_patterns||{});
  if (patterns.length) {
    cards += `<div class="info-card" style="grid-column:1/-1">
      <div class="label">Active Candlestick Patterns</div>
      <div class="value" style="font-size:13px;color:var(--yellow)">${patterns.map(p=>p.replace(/_/g,' ')).join(' · ')}</div>
    </div>`;
  }

  document.getElementById('drawerCards').innerHTML = cards ||
    '<div style="color:var(--muted);font-size:12px;padding:4px 0">No active signal or open position</div>';

  // ── Candlestick + EMA + BB chart ──────────────────────────────────────────
  _lwCandle = LightweightCharts.createChart(
    document.getElementById('candleChart'), { ..._lwOpts, height: 320 }
  );
  const cs = _lwCandle.addCandlestickSeries({
    upColor:'#3fb950', downColor:'#f85149',
    borderUpColor:'#3fb950', borderDownColor:'#f85149',
    wickUpColor:'#3fb950', wickDownColor:'#f85149',
  });
  cs.setData(cData.candles);

  // Entry/exit markers from trade history
  const markers = (dData.trades || [])
    .filter(t => t.entry && t.time)
    .map(t => ({
      time: Math.floor(new Date(t.time).getTime() / 1000),
      position: t.action === 'open' ? 'belowBar' : 'aboveBar',
      color: t.action === 'open'
        ? (t.side === 'long' ? '#3fb950' : '#f85149')
        : (t.pnl >= 0 ? '#3fb950' : '#f85149'),
      shape: t.action === 'open'
        ? (t.side === 'long' ? 'arrowUp' : 'arrowDown')
        : 'circle',
      text: t.action === 'open'
        ? `${(t.side||'').toUpperCase()} @${fmt(t.entry||0,2)}`
        : `${t.reason==='take_profit'?'TP':'SL'} ${t.pnl>=0?'+':''}$${fmt(Math.abs(t.pnl||0),2)}`,
      size: 1,
    }))
    .filter(m => m.time > 0)
    .sort((a,b) => a.time - b.time);
  if (markers.length) cs.setMarkers(markers);

  // EMA lines
  const emaOpts = [
    [cData.ema9,   '#f97316', 1],
    [cData.ema21,  '#facc15', 1],
    [cData.ema50,  '#60a5fa', 1.5],
    [cData.ema200, '#a78bfa', 2],
  ];
  emaOpts.forEach(([data, color, width]) => {
    if (!data?.length) return;
    const s = _lwCandle.addLineSeries({ color, lineWidth: width, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    s.setData(data);
  });

  // Bollinger Bands
  if (cData.bb_up?.length) {
    const bbStyle = { lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
    const bbu = _lwCandle.addLineSeries({ ...bbStyle, color: 'rgba(88,166,255,.4)', lineStyle: 2 });
    const bbm = _lwCandle.addLineSeries({ ...bbStyle, color: 'rgba(88,166,255,.25)', lineStyle: 1 });
    const bbl = _lwCandle.addLineSeries({ ...bbStyle, color: 'rgba(88,166,255,.4)', lineStyle: 2 });
    bbu.setData(cData.bb_up); bbm.setData(cData.bb_mid); bbl.setData(cData.bb_low);
  }

  // ── Volume chart ──────────────────────────────────────────────────────────
  _lwVolume = LightweightCharts.createChart(
    document.getElementById('volumeChart'), { ..._lwOpts, height: 100 }
  );
  const vs = _lwVolume.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' });
  vs.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0 } });
  vs.setData(cData.volume);

  // ── RSI chart ─────────────────────────────────────────────────────────────
  _lwRsi = LightweightCharts.createChart(
    document.getElementById('rsiChart'), { ..._lwOpts, height: 100 }
  );
  const rs = _lwRsi.addLineSeries({ color: '#c084fc', lineWidth: 1.5, priceLineVisible: false });
  rs.setData(cData.rsi || []);
  // Overbought/oversold reference lines
  if (cData.rsi?.length) {
    const mkLine = (v, color) => {
      const s = _lwRsi.addLineSeries({ color, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      s.setData(cData.rsi.map(p => ({ time: p.time, value: v })));
    };
    mkLine(65, 'rgba(248,81,73,.5)');
    mkLine(35, 'rgba(63,185,80,.5)');
    mkLine(50, 'rgba(139,148,158,.3)');
  }

  // ── MACD chart ────────────────────────────────────────────────────────────
  _lwMacd = LightweightCharts.createChart(
    document.getElementById('macdChart'), { ..._lwOpts, height: 110 }
  );
  if (cData.macd_hist?.length) {
    const mh = _lwMacd.addHistogramSeries({ priceLineVisible: false });
    mh.setData(cData.macd_hist);
  }
  if (cData.macd?.length) {
    const ml = _lwMacd.addLineSeries({ color: '#60a5fa', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
    ml.setData(cData.macd);
  }
  if (cData.macd_signal?.length) {
    const msl = _lwMacd.addLineSeries({ color: '#f97316', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
    msl.setData(cData.macd_signal);
  }

  // Sync crosshair across all charts
  [_lwCandle, _lwVolume, _lwRsi, _lwMacd].forEach((chart, idx) => {
    chart.subscribeCrosshairMove(p => {
      if (!p.time) return;
      [_lwCandle, _lwVolume, _lwRsi, _lwMacd].forEach((c, i) => {
        if (i !== idx) try { c.setCrosshairPosition(p.seriesData.values().next().value?.value, p.time, c.addLineSeries()); } catch(_){}
      });
    });
  });

  // Scroll all to the same position (most recent)
  _lwCandle.timeScale().scrollToRealTime();
  _lwVolume.timeScale().scrollToRealTime();
  _lwRsi.timeScale().scrollToRealTime();
  _lwMacd.timeScale().scrollToRealTime();

  // ── Trade log ─────────────────────────────────────────────────────────────
  const trades = dData.trades || [];
  const tbody = document.getElementById('drawerTradeBody');
  document.getElementById('drawerTradeTitle').textContent =
    `Trade History — ${symbol} (${trades.length} trades)`;
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No trades for this symbol yet</td></tr>';
  } else {
    tbody.innerHTML = trades.map(t => {
      const isClose = t.action === 'close';
      const reasonBadge = t.reason === 'take_profit' ? '<span class="badge badge-tp">TP ✓</span>' :
        t.reason === 'stop_loss' ? '<span class="badge badge-sl">SL ✗</span>' :
        `<span class="badge badge-neutral">${t.reason||'—'}</span>`;
      return `<tr>
        <td style="color:var(--muted)">${fmtTime(t.time)}</td>
        <td><span class="badge ${isClose?'badge-short':'badge-long'}">${(t.action||'').toUpperCase()}</span></td>
        <td>${dirBadge(t.side)}</td>
        <td class="mono">$${t.entry ? fmt(t.entry, price>100?2:4) : '—'}</td>
        <td class="mono">$${t.price ? fmt(t.price, price>100?2:4) : (t.tp ? fmt(t.tp, price>100?2:4) : '—')}</td>
        <td class="mono">${t.pnl !== undefined ? fmtPnl(t.pnl) : '—'}</td>
        <td>${reasonBadge}</td>
      </tr>`;
    }).join('');
  }
}

async function setTf(tf) {
  _drawerTf = tf;
  ['1h','6h','1d'].forEach(t => {
    const btn = document.getElementById('tf'+t);
    if (btn) btn.style.borderColor = t === tf ? 'var(--blue)' : 'var(--border)';
    if (btn) btn.style.color = t === tf ? 'var(--blue)' : 'var(--text)';
  });
  if (_drawerSymbol) await _loadDrawerData(_drawerSymbol, tf);
}

function closeDrawer() {
  document.getElementById('symbolDrawer').classList.remove('open');
  document.getElementById('drawerOverlay').classList.remove('open');
  document.body.style.overflow = '';
  setTimeout(_destroyCharts, 300);
}

// Make signal table rows clickable
function makeClickable(tbodyId, symbolFn) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.querySelectorAll('tr').forEach(tr => {
    const sym = symbolFn(tr);
    if (sym) { tr.classList.add('clickable'); tr.onclick = () => openSymbolDrawer(sym); }
  });
}

// ── Machine Learning tab ────────────────────────────────────────────────────────
let mlTrainPoll = null, tunerPoll = null;

async function loadMLState() {
  await populateTunerSymbols();
  await refreshMLModels();
  // resume polling if a job is already running
  const ts = await (await fetch('/api/ml/training_status')).json();
  if (ts.active) startMLTrainPoll();
  const tu = await (await fetch('/api/tuner/status')).json();
  if (tu.active) startTunerPoll();
  if (tu.result) renderTunerResult(tu.result);
}

async function populateTunerSymbols() {
  try {
    const d = await (await fetch('/api/universe')).json();
    const sel = document.getElementById('tunerSymbol');
    const cur = sel.value;
    sel.innerHTML = (d.symbols || ['BTC/USD']).map(s => `<option value="${s}">${s}</option>`).join('');
    if (cur) sel.value = cur;
  } catch(_){}
}

async function refreshMLModels() {
  try {
    const d = await (await fetch('/api/ml/status')).json();
    const tbody = document.getElementById('mlBody');
    const models = d.models || [];
    if (!models.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty">No models trained yet — click “Retrain All”.</td></tr>';
      return;
    }
    tbody.innerHTML = models.map(m => {
      const aucColor = m.auc_usable ? 'var(--green)' : 'var(--red)';
      const statusBadge = m.auc_usable
        ? '<span class="badge badge-long">USED</span>'
        : '<span class="badge badge-short">IGNORED &lt;0.53</span>';
      const topFeat = Object.keys(m.top_features || {}).slice(0,3).map(f=>f.replace(/_/g,' ')).join(', ') || '—';
      const mae = m.extrema_mae != null ? fmt(m.extrema_mae,3) : '—';
      const maeColor = m.extrema_usable ? 'var(--green)' : 'var(--muted)';
      return `<tr>
        <td><strong>${m.symbol}</strong></td>
        <td class="mono" style="color:${aucColor};font-weight:700">${fmt(m.test_auc,3)}</td>
        <td class="mono" style="color:var(--muted)">${fmt(m.train_auc,3)}</td>
        <td>${statusBadge}</td>
        <td class="mono" style="color:${maeColor}">${mae}</td>
        <td>${m.n_features}</td>
        <td style="font-size:11px;color:var(--muted)">${topFeat}</td>
        <td style="font-size:11px;color:var(--muted)">${m.trained_at ? fmtTime(m.trained_at).slice(0,16) : '—'}</td>
      </tr>`;
    }).join('');
  } catch(e){ console.error('ML status error', e); }
}

async function trainML() {
  document.getElementById('mlTrainBtn').disabled = true;
  document.getElementById('mlTrainStatus').textContent = 'Starting…';
  await fetch('/api/ml/train', {method:'POST'});
  startMLTrainPoll();
}

function startMLTrainPoll() {
  if (mlTrainPoll) clearInterval(mlTrainPoll);
  document.getElementById('mlTrainBtn').disabled = true;
  mlTrainPoll = setInterval(async () => {
    const s = await (await fetch('/api/ml/training_status')).json();
    document.getElementById('mlTrainStatus').textContent = s.progress || '';
    if (!s.active) {
      clearInterval(mlTrainPoll); mlTrainPoll = null;
      document.getElementById('mlTrainBtn').disabled = false;
      document.getElementById('mlTrainStatus').textContent = 'Done';
      refreshMLModels();
    }
  }, 2000);
}

// ── Auto-tuner ──────────────────────────────────────────────────────────────────
async function runTuner() {
  const symbol = document.getElementById('tunerSymbol').value;
  const days   = parseInt(document.getElementById('tunerDays').value)||90;
  const trials = parseInt(document.getElementById('tunerTrials').value)||50;
  document.getElementById('tunerRunBtn').disabled = true;
  document.getElementById('tunerProgress').classList.add('visible');
  document.getElementById('tunerResult').style.display = 'none';
  document.getElementById('tunerProgressText').textContent = 'Starting…';
  await fetch('/api/tuner/run', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({symbol, days, trials})});
  startTunerPoll();
}

function startTunerPoll() {
  if (tunerPoll) clearInterval(tunerPoll);
  document.getElementById('tunerRunBtn').disabled = true;
  document.getElementById('tunerProgress').classList.add('visible');
  tunerPoll = setInterval(async () => {
    const s = await (await fetch('/api/tuner/status')).json();
    document.getElementById('tunerProgressText').textContent = s.progress || 'Running…';
    if (!s.active) {
      clearInterval(tunerPoll); tunerPoll = null;
      document.getElementById('tunerRunBtn').disabled = false;
      document.getElementById('tunerProgress').classList.remove('visible');
      if (s.result) renderTunerResult(s.result);
    }
  }, 1500);
}

function renderTunerResult(r) {
  document.getElementById('tunerResult').style.display = 'block';
  document.getElementById('tunerSummary').innerHTML = `
    <div class="bt-stat"><div class="l">Symbol</div><div class="v">${r.symbol}</div></div>
    <div class="bt-stat"><div class="l">Best Sharpe</div><div class="v ${r.best_sharpe>=0?'pos':'neg'}">${fmt(r.best_sharpe,2)}</div></div>
    <div class="bt-stat"><div class="l">Best Return</div><div class="v ${r.best_return>=0?'pos':'neg'}">${r.best_return>=0?'+':''}${fmt(r.best_return,1)}%</div></div>
    <div class="bt-stat"><div class="l">Trials</div><div class="v">${r.n_trials}</div></div>
  `;
  document.getElementById('tunerImprovement').textContent = r.improvement || 'No changes';
}

async function applyTuner() {
  const res = await (await fetch('/api/tuner/apply', {method:'POST'})).json();
  document.getElementById('tunerApplied').textContent =
    res.status === 'applied' ? '✓ Applied to running config' : (res.error||'failed');
}

// ── Init ───────────────────────────────────────────────────────────────────────
initEquityChart();
fetchState();
setInterval(fetchState, 5000);
// After each state update, make rows clickable
const _origFetchState = fetchState;
window.addEventListener('load', () => setTf('1h'));
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


def start_server(host: str = "0.0.0.0", port: int = 8081) -> None:
    """Run Flask in a background daemon thread."""
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
        name="dashboard-server",
    )
    t.start()
    logger.info("Dashboard running at http://localhost:%d", port)