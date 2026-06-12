"""
Flask app for the TradingView webhook bot: the webhook receiver, the JSON API,
and the dashboard — all in one process on one port.

  POST  {WEBHOOK_PATH}     ← TradingView alerts land here
  GET   /                  → dashboard (HTML)
  GET   /api/health        → mode, webhook URL, counts, live auth/balance
  GET   /api/portfolio     → marked-to-market portfolio summary
  GET   /api/positions     → open positions
  GET   /api/trades        → closed trade history (newest first)
  GET   /api/alerts        → raw inbound alert log (newest first)
  GET   /api/equity        → equity curve points
  GET   /api/markets       → allowlist ticker prices (mini market widget)
  GET   /api/candles/<sym> → OHLCV + EMAs for the price chart
  POST  /api/control       → {"action": pause|resume|flatten|reset|reset_keep}
"""
from __future__ import annotations

import json
import time
import logging
import threading

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import config
from webhook import store, engine
from exchange.market_data import fetch_ticker, get_quote_balance

logger = logging.getLogger("webhook")

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024   # cap webhook body size (anti-abuse)

# Caches refreshed by the background mark-to-market loop so request handlers never
# block on network I/O.
_last_summary: dict = {}
_market_tickers: dict = {}
_candle_cache: dict = {}
_CANDLE_TTL = 60


# ── Webhook receiver ──────────────────────────────────────────────────────────

def _parse_body() -> dict | None:
    """
    TradingView posts the alert message as the request body. It may arrive as
    application/json or as text/plain that happens to be JSON. Parse both.
    """
    data = request.get_json(force=True, silent=True)
    if isinstance(data, dict):
        return data
    raw = request.get_data(as_text=True) or ""
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


@app.route(config.WEBHOOK_PATH, methods=["POST"])
def webhook():
    source_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    source_ip = source_ip.split(",")[0].strip()

    # Optional IP allowlist (off by default — useless behind a tunnel/proxy).
    if config.WEBHOOK_IP_ALLOWLIST_ENABLED and source_ip not in config.WEBHOOK_ALLOWED_IPS:
        logger.warning("Webhook from non-allowlisted IP %s rejected", source_ip)
        return jsonify({"status": "rejected", "reason": "ip not allowed"}), 403

    payload = _parse_body()
    if payload is None:
        store.log_alert({"_raw": request.get_data(as_text=True)[:500]}, source_ip,
                        "rejected", "body is not valid JSON")
        return jsonify({"status": "rejected", "reason": "invalid json body"}), 400

    try:
        code, resp = engine.process_alert(payload, source_ip)
    except Exception as e:
        logger.exception("Webhook processing error")
        store.log_alert(engine._redact(payload), source_ip, "error", str(e))
        return jsonify({"status": "error", "reason": str(e)}), 500
    return jsonify(resp), code


# ── JSON API ──────────────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    ms = engine.mode_status()
    snap = store.snapshot()
    alerts = snap.get("alerts", [])
    health = {
        **ms,
        "webhook_path": config.WEBHOOK_PATH,
        "port":         config.WEBHOOK_PORT,
        "passphrase_set": bool(config.WEBHOOK_PASSPHRASE),
        "ip_allowlist": config.WEBHOOK_IP_ALLOWLIST_ENABLED,
        "alerts_total": len(alerts),
        "last_alert":   alerts[-1]["time"] if alerts else None,
        "paused":       snap.get("paused", False),
        "symbol_allowlist": config.WEBHOOK_SYMBOL_ALLOWLIST,
        "caps": {
            "default_order_usd": config.WEBHOOK_DEFAULT_ORDER_USD,
            "max_order_usd":     config.WEBHOOK_MAX_ORDER_USD,
            "max_open_positions": config.WEBHOOK_MAX_OPEN_POSITIONS,
            "max_daily_loss_usd": config.WEBHOOK_MAX_DAILY_LOSS_USD,
        },
    }
    # Real exchange USD balance (live only — for reconciliation against the ledger)
    if ms["live"]:
        try:
            health["exchange_balance"] = round(get_quote_balance(), 2)
        except Exception:
            health["exchange_balance"] = None
    return jsonify(health)


@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(_last_summary or store.mark_to_market(_prices_for_positions()))


@app.route("/api/positions")
def api_positions():
    summ = _last_summary or store.mark_to_market(_prices_for_positions())
    return jsonify(summ.get("open_positions", []))


@app.route("/api/trades")
def api_trades():
    page = int(request.args.get("page", 1))
    size = min(int(request.args.get("size", 100)), 500)
    trades = list(reversed(store.snapshot()["trades"]))   # newest first
    start = (page - 1) * size
    return jsonify({"total": len(trades), "page": page,
                    "trades": trades[start:start + size]})


@app.route("/api/alerts")
def api_alerts():
    size = min(int(request.args.get("size", 100)), 300)
    alerts = list(reversed(store.snapshot()["alerts"]))   # newest first
    return jsonify(alerts[:size])


@app.route("/api/equity")
def api_equity():
    return jsonify(store.snapshot()["equity_curve"])


@app.route("/api/markets")
def api_markets():
    return jsonify(_market_tickers)


@app.route("/api/candles/<path:symbol>")
def api_candles(symbol):
    symbol = symbol.replace("-", "/")
    cached = _candle_cache.get(symbol)
    if cached and (time.time() - cached["ts"]) < _CANDLE_TTL:
        return jsonify(cached["data"])
    tf = request.args.get("tf", "1h")
    try:
        from exchange.market_data import fetch_ohlcv
        from core.indicators import add_emas
        import numpy as np
        df = fetch_ohlcv(symbol, tf, limit=200)
        if df.empty:
            return jsonify({"error": "no data"}), 404
        df = add_emas(df)

        def s(v):
            try:
                f = float(v)
                return None if np.isnan(f) else round(f, 6)
            except Exception:
                return None

        candles, volume, ema21, ema50 = [], [], [], []
        for ts, row in df.iterrows():
            t = int(ts.timestamp())
            candles.append({"time": t, "open": s(row["open"]), "high": s(row["high"]),
                            "low": s(row["low"]), "close": s(row["close"])})
            volume.append({"time": t, "value": s(row["volume"]),
                           "color": "rgba(63,185,80,.5)" if row["close"] >= row["open"]
                                    else "rgba(248,81,73,.5)"})
            ema21.append({"time": t, "value": s(row.get("ema21"))})
            ema50.append({"time": t, "value": s(row.get("ema50"))})
        data = {
            "symbol": symbol, "tf": tf, "candles": candles, "volume": volume,
            "ema21": [p for p in ema21 if p["value"] is not None],
            "ema50": [p for p in ema50 if p["value"] is not None],
            "last_price": s(df["close"].iloc[-1]),
            "change_pct": round((df["close"].iloc[-1] - df["close"].iloc[-2])
                                / df["close"].iloc[-2] * 100, 2) if len(df) > 1 else 0,
        }
        _candle_cache[symbol] = {"ts": time.time(), "data": data}
        return jsonify(data)
    except Exception as e:
        logger.error("Candle fetch failed for %s: %s", symbol, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/control", methods=["POST"])
def api_control():
    action = (request.get_json(silent=True) or {}).get("action", "")
    if action == "pause":
        store.set_paused(True)
        return jsonify({"status": "paused"})
    if action == "resume":
        store.set_paused(False)
        return jsonify({"status": "running"})
    if action == "flatten":
        res = engine.flatten_all()
        _refresh_summary()
        return jsonify({"status": "flattened", **res})
    if action == "reset":
        store.reset(keep_history=False)
        _refresh_summary()
        return jsonify({"status": "reset"})
    if action == "reset_keep":
        store.reset(keep_history=True)
        _refresh_summary()
        return jsonify({"status": "reset (history kept)"})
    return jsonify({"error": "unknown action"}), 400


@app.route("/")
def dashboard():
    # Served raw (not via Jinja) so the TradingView '{{...}}' samples in the Setup
    # panel aren't interpreted as template syntax.
    return Response(DASHBOARD_HTML, mimetype="text/html")


# ── Background mark-to-market loop ────────────────────────────────────────────

def _prices_for_positions() -> dict:
    prices = {}
    for sym in store.snapshot()["positions"]:
        px = _market_tickers.get(sym, {}).get("price")
        if not px:
            tkr = fetch_ticker(sym)
            px = (tkr.get("last") or tkr.get("close")) if tkr else None
        if px:
            prices[sym] = float(px)
    return prices


def _refresh_summary() -> None:
    global _last_summary
    _last_summary = store.mark_to_market(_prices_for_positions())


def _mark_loop() -> None:
    """Periodically refresh tickers + mark the portfolio to market (off-path)."""
    global _market_tickers, _last_summary
    while True:
        try:
            # Refresh allowlist + held-symbol tickers for the markets widget + marking.
            symbols = set(config.WEBHOOK_SYMBOL_ALLOWLIST) | set(store.snapshot()["positions"].keys())
            tickers, prices = {}, {}
            for sym in symbols:
                tkr = fetch_ticker(sym)
                if not tkr:
                    continue
                px = tkr.get("last") or tkr.get("close")
                if px:
                    prices[sym] = float(px)
                    tickers[sym] = {
                        "price": round(float(px), 6),
                        "change_pct": round(float(tkr.get("percentage") or 0.0), 2),
                    }
            if tickers:
                _market_tickers = tickers
            _last_summary = store.mark_to_market(prices)
        except Exception as e:
            logger.debug("mark loop error: %s", e)
        time.sleep(max(5, config.WEBHOOK_MARK_INTERVAL_SEC))


def start(host: str | None = None, port: int | None = None) -> None:
    """Start the background mark loop and run Flask (blocking)."""
    host = host or config.WEBHOOK_HOST
    port = port or config.WEBHOOK_PORT
    threading.Thread(target=_mark_loop, daemon=True, name="webhook-mark").start()
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


# ── Dashboard HTML (served raw) ───────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TradingView Webhook Bot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2230;--border:#30363d;--fg:#e6edf3;
        --muted:#8b949e;--green:#3fb950;--red:#f85149;--blue:#58a6ff;--yellow:#d29922;}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  a{color:var(--blue);text-decoration:none}
  .wrap{max-width:1280px;margin:0 auto;padding:18px}
  header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px}
  header h1{font-size:18px;font-weight:650;display:flex;align-items:center;gap:8px}
  .badge{padding:3px 10px;border-radius:20px;font-size:12px;font-weight:650;border:1px solid var(--border)}
  .badge.live{background:rgba(248,81,73,.15);color:var(--red);border-color:var(--red)}
  .badge.paper{background:rgba(88,166,255,.12);color:var(--blue);border-color:var(--blue)}
  .badge.paused{background:rgba(210,153,34,.15);color:var(--yellow);border-color:var(--yellow)}
  .spacer{flex:1}
  button{background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:7px;
         padding:7px 13px;font-size:13px;cursor:pointer;font-weight:550}
  button:hover{border-color:var(--blue)}
  button.danger:hover{border-color:var(--red);color:var(--red)}
  button.active{border-color:var(--blue);color:var(--blue)}
  .tabs{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
  .tab{padding:8px 16px;border-radius:8px;cursor:pointer;color:var(--muted);font-weight:600;border:1px solid transparent}
  .tab.active{background:var(--panel);color:var(--fg);border-color:var(--border)}
  .grid{display:grid;gap:14px}
  .cards{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
  .card .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
  .card .v{font-size:24px;font-weight:680;margin-top:4px}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:14px}
  .panel h2{font-size:14px;margin-bottom:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;position:sticky;top:0;background:var(--panel)}
  tr:hover td{background:var(--panel2)}
  .pos{color:var(--green)} .neg{color:var(--red)} .mut{color:var(--muted)}
  .pill{padding:2px 8px;border-radius:12px;font-size:11px;font-weight:650}
  .pill.executed{background:rgba(63,185,80,.15);color:var(--green)}
  .pill.rejected{background:rgba(210,153,34,.15);color:var(--yellow)}
  .pill.duplicate{background:rgba(139,148,158,.18);color:var(--muted)}
  .pill.error{background:rgba(248,81,73,.15);color:var(--red)}
  .pill.buy{background:rgba(63,185,80,.15);color:var(--green)}
  .pill.sell{background:rgba(248,81,73,.15);color:var(--red)}
  .empty{text-align:center;color:var(--muted);padding:22px}
  .scroll{max-height:420px;overflow:auto}
  code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
  pre{background:#0a0e14;border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;color:#c9d1d9}
  .kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px;font-size:13px}
  .kv .k{color:var(--muted)}
  .hide{display:none}
  .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
  select{background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:7px;padding:6px 10px}
  .warn{background:rgba(210,153,34,.1);border:1px solid var(--yellow);border-radius:8px;padding:10px 12px;color:#e3b341;font-size:13px;margin-bottom:12px}
  .mkt{display:flex;gap:10px;flex-wrap:wrap}
  .mkt .m{background:var(--panel2);border:1px solid var(--border);border-radius:9px;padding:8px 12px;min-width:120px}
  .mkt .m .s{font-weight:650} .mkt .m .p{font-size:13px;margin-top:2px}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
  .copy{font-size:11px;padding:3px 8px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>📡 TradingView Webhook Bot</h1>
    <span id="modeBadge" class="badge paper">PAPER</span>
    <span id="pausedBadge" class="badge paused hide">PAUSED</span>
    <div class="spacer"></div>
    <span class="mut" id="lastUpdate"></span>
    <button id="pauseBtn" onclick="togglePause()">Pause</button>
    <button class="danger" onclick="flatten()">Flatten All</button>
    <button class="danger" onclick="resetPortfolio()">Reset</button>
  </header>

  <div id="modeWarn" class="warn hide"></div>

  <div class="tabs">
    <div class="tab active" data-tab="overview" onclick="switchTab('overview')">Overview</div>
    <div class="tab" data-tab="trades" onclick="switchTab('trades')">Trades</div>
    <div class="tab" data-tab="chart" onclick="switchTab('chart')">Chart</div>
    <div class="tab" data-tab="alerts" onclick="switchTab('alerts')">Webhook Log</div>
    <div class="tab" data-tab="setup" onclick="switchTab('setup')">Setup</div>
  </div>

  <!-- OVERVIEW -->
  <section id="tab-overview">
    <div class="grid cards" style="margin-bottom:14px">
      <div class="card"><div class="l">Equity</div><div class="v" id="cEquity">—</div></div>
      <div class="card"><div class="l">Total P&amp;L</div><div class="v" id="cPnl">—</div></div>
      <div class="card"><div class="l">Cash</div><div class="v" id="cCash">—</div></div>
      <div class="card"><div class="l">Unrealized</div><div class="v" id="cUnreal">—</div></div>
      <div class="card"><div class="l">Win Rate</div><div class="v" id="cWin">—</div></div>
      <div class="card"><div class="l">Open / Closed</div><div class="v" id="cCounts">—</div></div>
    </div>

    <div class="panel">
      <h2>Equity Curve</h2>
      <canvas id="equityChart" height="90"></canvas>
    </div>

    <div class="panel">
      <h2>Markets</h2>
      <div class="mkt" id="markets"><div class="empty">Loading…</div></div>
    </div>

    <div class="panel">
      <h2>Open Positions</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Price</th>
            <th>Value</th><th>P&amp;L</th><th>%</th><th>Strategy</th></tr></thead>
          <tbody id="posBody"><tr><td colspan="9" class="empty">No open positions</td></tr></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- TRADES -->
  <section id="tab-trades" class="hide">
    <div class="panel">
      <h2>Closed Trades (net of fees)</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Closed</th><th>Symbol</th><th>Qty</th><th>Entry</th><th>Exit</th>
            <th>P&amp;L</th><th>%</th><th>Reason</th><th>Strategy</th><th>Mode</th></tr></thead>
          <tbody id="tradesBody"><tr><td colspan="10" class="empty">No trades yet</td></tr></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- CHART -->
  <section id="tab-chart" class="hide">
    <div class="panel">
      <div class="row" style="margin-bottom:12px">
        <select id="chartSym" onchange="loadChart()"></select>
        <select id="chartTf" onchange="loadChart()">
          <option value="15m">15m</option><option value="1h" selected>1h</option>
          <option value="6h">6h</option><option value="1d">1d</option>
        </select>
        <span id="chartPrice" class="mut"></span>
      </div>
      <div id="priceChart" style="height:420px"></div>
    </div>
  </section>

  <!-- ALERTS -->
  <section id="tab-alerts" class="hide">
    <div class="panel">
      <h2>Inbound Webhook Alerts</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Time</th><th>Status</th><th>Symbol</th><th>Action</th>
            <th>Message</th><th>IP</th></tr></thead>
          <tbody id="alertsBody"><tr><td colspan="6" class="empty">No alerts received yet</td></tr></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- SETUP -->
  <section id="tab-setup" class="hide">
    <div class="panel">
      <h2>Webhook URL</h2>
      <div class="row">
        <pre id="hookUrl" style="flex:1">—</pre>
        <button class="copy" onclick="copyHook()">Copy</button>
      </div>
      <p class="mut" style="margin-top:8px">Paste this into a TradingView alert's
        <b>“Webhook URL”</b> field. TradingView only posts to ports <b>80/443</b> on a
        <b>public</b> URL — run this behind a tunnel (ngrok / cloudflared) or reverse proxy.</p>
    </div>
    <div class="panel">
      <h2>Alert Message (JSON) — paste into the alert's “Message” box</h2>
      <pre id="samplePayload">—</pre>
      <p class="mut" style="margin-top:8px">Fields: <code>action</code> (buy/sell/close),
        <code>symbol</code>, optional <code>order_size</code> (USD, e.g. <code>150</code>; or
        <code>"10%"</code> of equity; or <code>qty</code> in units), optional <code>id</code>
        (de-dupes retries), <code>strategy</code>. The <code>passphrase</code> must match the
        server's <code>WEBHOOK_PASSPHRASE</code>.</p>
    </div>
    <div class="panel">
      <h2>Configuration</h2>
      <div class="kv" id="cfgKv"></div>
    </div>
  </section>

  <footer class="mut" style="text-align:center;padding:14px;font-size:12px">
    Paper P&amp;L is net of estimated round-trip fees. Live trading is armed only when
    DRY_RUN=False, WEBHOOK_LIVE_ENABLED=True, a passphrase is set, and the API key validates.
  </footer>
</div>

<script>
const $ = id => document.getElementById(id);
const fmt = (n,d=2) => (n==null||isNaN(n)) ? '—' : Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const usd = (n,d=2) => (n==null||isNaN(n)) ? '—' : '$'+fmt(n,d);
const cls = n => n>0?'pos':(n<0?'neg':'mut');
const sign = n => (n>=0?'+':'');
let HEALTH = {};

function switchTab(t){
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===t));
  ['overview','trades','chart','alerts','setup'].forEach(s=>$('tab-'+s).classList.toggle('hide',s!==t));
  if(t==='trades') loadTrades();
  if(t==='alerts') loadAlerts();
  if(t==='chart') { buildSymbolOptions(); loadChart(); }
}

// ── Equity chart ──────────────────────────────────────────────────────────────
let equityChart;
function initEquity(){
  equityChart = new Chart($('equityChart'), {
    type:'line',
    data:{labels:[],datasets:[{data:[],borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,.08)',
      fill:true,tension:.25,pointRadius:0,borderWidth:2}]},
    options:{responsive:true,plugins:{legend:{display:false}},
      scales:{x:{display:false},y:{grid:{color:'#21262d'},ticks:{color:'#8b949e',
        callback:v=>'$'+v.toLocaleString()}}}}
  });
}
async function loadEquity(){
  try{
    const d = await (await fetch('/api/equity')).json();
    equityChart.data.labels = d.map(p=>p.t);
    equityChart.data.datasets[0].data = d.map(p=>p.v);
    const up = d.length>1 && d[d.length-1].v>=d[0].v;
    equityChart.data.datasets[0].borderColor = up?'#3fb950':'#f85149';
    equityChart.data.datasets[0].backgroundColor = up?'rgba(63,185,80,.08)':'rgba(248,81,73,.08)';
    equityChart.update('none');
  }catch(e){}
}

// ── Health + portfolio (polled) ───────────────────────────────────────────────
async function tick(){
  try{
    HEALTH = await (await fetch('/api/health')).json();
    const badge = $('modeBadge');
    badge.textContent = HEALTH.mode;
    badge.className = 'badge ' + (HEALTH.live?'live':'paper');
    $('pausedBadge').classList.toggle('hide', !HEALTH.paused);
    $('pauseBtn').textContent = HEALTH.paused ? 'Resume' : 'Pause';
    // mode explanation
    const w = $('modeWarn');
    if(HEALTH.live){ w.classList.remove('hide'); w.textContent =
      '⚠️ LIVE TRADING ARMED — real Coinbase orders will be placed. '+
      (HEALTH.exchange_balance!=null?('Exchange USD balance: $'+fmt(HEALTH.exchange_balance)):''); }
    else if(HEALTH.live_enabled && !HEALTH.auth_ok){ w.classList.remove('hide');
      w.textContent='ℹ️ Live requested but not armed ('+HEALTH.auth_msg+') — running PAPER.'; }
    else { w.classList.add('hide'); }
    renderSetup();
  }catch(e){}

  try{
    const p = await (await fetch('/api/portfolio')).json();
    $('cEquity').textContent = usd(p.equity);
    $('cPnl').innerHTML = `<span class="${cls(p.total_pnl)}">${sign(p.total_pnl)}${usd(p.total_pnl)} (${sign(p.total_pnl_pct)}${fmt(p.total_pnl_pct)}%)</span>`;
    $('cCash').textContent = usd(p.cash);
    $('cUnreal').innerHTML = `<span class="${cls(p.unrealized_pnl)}">${sign(p.unrealized_pnl)}${usd(p.unrealized_pnl)}</span>`;
    $('cWin').textContent = fmt(p.win_rate,1)+'%';
    $('cCounts').textContent = `${p.open_count} / ${p.closed_count}`;
    renderPositions(p.open_positions||[]);
    $('lastUpdate').textContent = 'updated '+new Date().toLocaleTimeString();
  }catch(e){}
  loadEquity();
  loadMarkets();
}

function renderPositions(rows){
  const tb=$('posBody');
  if(!rows.length){tb.innerHTML='<tr><td colspan="9" class="empty">No open positions</td></tr>';return;}
  tb.innerHTML = rows.map(p=>`<tr>
    <td><b>${p.symbol}</b></td><td><span class="pill buy">${p.side}</span></td>
    <td>${fmt(p.qty,6)}</td><td>${usd(p.entry,4)}</td><td>${p.price!=null?usd(p.price,4):'—'}</td>
    <td>${usd(p.value)}</td><td class="${cls(p.pnl)}">${sign(p.pnl)}${usd(p.pnl)}</td>
    <td class="${cls(p.pnl)}">${sign(p.pnl_pct)}${fmt(p.pnl_pct)}%</td>
    <td class="mut">${p.strategy||'—'}</td></tr>`).join('');
}

async function loadMarkets(){
  try{
    const m = await (await fetch('/api/markets')).json();
    const keys = Object.keys(m);
    if(!keys.length){$('markets').innerHTML='<div class="empty">No market data</div>';return;}
    $('markets').innerHTML = keys.sort().map(s=>{
      const t=m[s];const c=cls(t.change_pct);
      return `<div class="m"><div class="s">${s}</div>
        <div class="p">${usd(t.price, t.price<10?4:2)} <span class="${c}">${sign(t.change_pct)}${fmt(t.change_pct,2)}%</span></div></div>`;
    }).join('');
  }catch(e){}
}

async function loadTrades(){
  try{
    const d = await (await fetch('/api/trades?size=200')).json();
    const tb=$('tradesBody');
    if(!d.trades.length){tb.innerHTML='<tr><td colspan="10" class="empty">No trades yet</td></tr>';return;}
    tb.innerHTML = d.trades.map(t=>`<tr>
      <td class="mut">${(t.closed_at||'').replace('T',' ').slice(5,16)}</td>
      <td><b>${t.symbol}</b></td><td>${fmt(t.qty,6)}</td>
      <td>${usd(t.entry,4)}</td><td>${usd(t.exit,4)}</td>
      <td class="${cls(t.pnl)}">${sign(t.pnl)}${usd(t.pnl)}</td>
      <td class="${cls(t.pnl)}">${sign(t.pnl_pct)}${fmt(t.pnl_pct,2)}%</td>
      <td class="mut">${(t.reason||'').replace('_',' ')}</td>
      <td class="mut">${t.strategy||'—'}</td>
      <td><span class="pill ${t.live?'sell':'duplicate'}">${t.live?'LIVE':'PAPER'}</span></td></tr>`).join('');
  }catch(e){}
}

async function loadAlerts(){
  try{
    const d = await (await fetch('/api/alerts?size=150')).json();
    const tb=$('alertsBody');
    if(!d.length){tb.innerHTML='<tr><td colspan="6" class="empty">No alerts received yet</td></tr>';return;}
    tb.innerHTML = d.map(a=>`<tr>
      <td class="mut">${(a.time||'').replace('T',' ').slice(5,19)}</td>
      <td><span class="pill ${a.status}">${a.status}</span></td>
      <td>${a.symbol||'—'}</td>
      <td>${a.action?`<span class="pill ${a.action==='buy'?'buy':'sell'}">${a.action}</span>`:'—'}</td>
      <td style="text-align:left">${a.message||''}</td>
      <td class="mut">${a.ip||''}</td></tr>`).join('');
  }catch(e){}
}

// ── Price chart ───────────────────────────────────────────────────────────────
let lwChart, candleSeries, volSeries, ema21S, ema50S;
function initPrice(){
  const el=$('priceChart'); el.innerHTML='';
  lwChart = LightweightCharts.createChart(el,{
    layout:{background:{color:'#161b22'},textColor:'#8b949e'},
    grid:{vertLines:{color:'#21262d'},horzLines:{color:'#21262d'}},
    rightPriceScale:{borderColor:'#30363d'},timeScale:{borderColor:'#30363d',timeVisible:true},
    width:el.clientWidth,height:420});
  candleSeries=lwChart.addCandlestickSeries({upColor:'#3fb950',downColor:'#f85149',
    borderUpColor:'#3fb950',borderDownColor:'#f85149',wickUpColor:'#3fb950',wickDownColor:'#f85149'});
  volSeries=lwChart.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'',
    scaleMargins:{top:.82,bottom:0}});
  ema21S=lwChart.addLineSeries({color:'#d29922',lineWidth:1});
  ema50S=lwChart.addLineSeries({color:'#58a6ff',lineWidth:1});
  new ResizeObserver(()=>lwChart.applyOptions({width:el.clientWidth})).observe(el);
}
function buildSymbolOptions(){
  const sel=$('chartSym'); if(sel.options.length) return;
  const syms=(HEALTH.symbol_allowlist&&HEALTH.symbol_allowlist.length)?HEALTH.symbol_allowlist:['BTC/USD','ETH/USD'];
  sel.innerHTML=syms.map(s=>`<option value="${s}">${s}</option>`).join('');
}
async function loadChart(){
  if(!lwChart) initPrice();
  const sym=$('chartSym').value||'BTC/USD', tf=$('chartTf').value;
  try{
    const d=await (await fetch('/api/candles/'+sym.replace('/','-')+'?tf='+tf)).json();
    if(d.error){$('chartPrice').textContent='No data';return;}
    candleSeries.setData(d.candles); volSeries.setData(d.volume);
    ema21S.setData(d.ema21); ema50S.setData(d.ema50);
    lwChart.timeScale().fitContent();
    $('chartPrice').innerHTML = `${sym} &nbsp; <b>${usd(d.last_price,d.last_price<10?4:2)}</b> `+
      `<span class="${cls(d.change_pct)}">${sign(d.change_pct)}${fmt(d.change_pct,2)}%</span>`;
  }catch(e){$('chartPrice').textContent='Error loading chart';}
}

// ── Setup panel ───────────────────────────────────────────────────────────────
function renderSetup(){
  const origin = location.origin;
  const url = origin + (HEALTH.webhook_path||'/webhook');
  $('hookUrl').textContent = url;
  const sample = {
    passphrase: HEALTH.passphrase_set ? "<your WEBHOOK_PASSPHRASE>" : "(none set — add one before going live)",
    action: "{{strategy.order.action}}",
    symbol: "{{ticker}}",
    order_size: 150,
    price: "{{close}}",
    id: "{{timenow}}",
    strategy: "my_tv_strategy"
  };
  $('samplePayload').textContent = JSON.stringify(sample, null, 2);
  const c = HEALTH.caps||{};
  $('cfgKv').innerHTML = `
    <div class="k">Mode</div><div>${HEALTH.mode} ${HEALTH.live?'(real orders)':'(simulated)'}</div>
    <div class="k">Passphrase set</div><div>${HEALTH.passphrase_set?'✅ yes':'❌ no'}</div>
    <div class="k">IP allowlist</div><div>${HEALTH.ip_allowlist?'on':'off'}</div>
    <div class="k">Symbol allowlist</div><div>${(HEALTH.symbol_allowlist||[]).join(', ')||'(any)'}</div>
    <div class="k">Default order</div><div>$${fmt(c.default_order_usd)}</div>
    <div class="k">Max order</div><div>$${fmt(c.max_order_usd)}</div>
    <div class="k">Max open positions</div><div>${c.max_open_positions}</div>
    <div class="k">Daily loss limit</div><div>${c.max_daily_loss_usd>0?('$'+fmt(c.max_daily_loss_usd)):'off'}</div>
    <div class="k">Alerts received</div><div>${HEALTH.alerts_total||0}</div>`;
}
function copyHook(){ navigator.clipboard.writeText($('hookUrl').textContent); }

// ── Controls ──────────────────────────────────────────────────────────────────
async function ctl(action){ return (await (await fetch('/api/control',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({action})})).json()); }
async function togglePause(){ await ctl(HEALTH.paused?'resume':'pause'); tick(); }
async function flatten(){ if(confirm('Close ALL open positions at market?')){ await ctl('flatten'); tick(); loadTrades(); } }
async function resetPortfolio(){
  if(confirm('Reset the portfolio to starting capital? This clears positions. Keep trade history?')){
    const keep = confirm('OK = keep trade/alert history, Cancel = wipe everything');
    await ctl(keep?'reset_keep':'reset'); tick(); loadTrades(); loadAlerts();
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
initEquity();
tick();
setInterval(tick, 5000);
</script>
</body>
</html>"""
