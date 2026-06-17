"""Central configuration for the trading bot."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE          = os.getenv("EXCHANGE", "coinbase")
API_KEY           = os.getenv("API_KEY", "")
API_SECRET        = os.getenv("API_SECRET", "")
TESTNET           = os.getenv("TESTNET", "false").lower() == "true"

# ── Capital / Risk ────────────────────────────────────────────────────────────
TOTAL_CAPITAL_USDT      = float(os.getenv("TOTAL_CAPITAL_USDT", 1000))
MAX_OPEN_POSITIONS      = int(os.getenv("MAX_OPEN_POSITIONS", 5))
RISK_PER_TRADE_PCT      = float(os.getenv("RISK_PER_TRADE_PCT", 2.0))
MAX_PORTFOLIO_RISK_PCT  = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", 10.0))

# ── Universe of coins to scan ─────────────────────────────────────────────────
# Coinbase trades against USD (not USDT)
WATCHLIST = [
    "BTC/USD", "ETH/USD", "SOL/USD",
]
# Trimmed to quality assets only (evidence-driven): in the 13-coin backtest the
# small-cap alts bled worst (CRV -66%, AAVE -63%, DOGE -53%, ADA -50%) while
# BTC/ETH lost least. Low-quality alts add fee churn and tail risk with no edge.
# Full former list kept here for reference / dashboard, not for trading:
#   XRP, DOGE, ADA, LINK, DOT, UNI, ATOM, NEAR, AAVE, CRV  (all PF<0.7)
# Removed earlier: AVAX/USD (consistently -10%), OP/USD (0% win rate).

# Quote currency used for capital accounting
QUOTE_CURRENCY = "USD"

# ── Timeframes used in multi-timeframe analysis ───────────────────────────────
TF_PRIMARY   = "1h"   # signal generation
TF_TREND     = "6h"   # trend filter (Coinbase supports 1h/2h/6h/1d — not 4h)
TF_ENTRY     = "15m"  # entry timing

# ── Technical indicator parameters ───────────────────────────────────────────
EMA_FAST     = 9
EMA_MID      = 21
EMA_SLOW     = 50
EMA_TREND    = 200

RSI_PERIOD   = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

BB_PERIOD    = 20
BB_STD       = 2.0

ATR_PERIOD   = 14
ADX_PERIOD   = 14
ADX_THRESHOLD = 22        # minimum ADX to confirm trend (lowered to not over-filter)

STOCH_K      = 14
STOCH_D      = 3
STOCH_SMOOTH = 3

VOLUME_MA_PERIOD = 20
VOLUME_SURGE_MULTIPLIER = 1.5   # volume must be >1.5x 20-period average

# ── Trade management ─────────────────────────────────────────────────────────
ATR_STOP_MULTIPLIER    = 1.5   # stop = entry ± ATR*1.5
ATR_TARGET_MULTIPLIER  = 3.0   # reduced from 4.0 — more achievable, RR = 2.0
TRAILING_STOP_ATR      = 1.5   # trail at 1.5 ATR — lock in profits sooner

# ── Signal scoring thresholds ─────────────────────────────────────────────────
MIN_SIGNAL_SCORE = 5.5         # sweet spot — filters noise without killing BTC/ETH longs
STRONG_SIGNAL_SCORE = 7.5      # raised — scale up only on very strong setups

# ── Mean-reversion engine (core/mean_reversion.py — the contrarian rewrite) ───
# Fades 1h extremes instead of chasing trend. Trades ONLY in ranges (low ADX).
# Geometry is flipped: tight target (revert to mean), wider stop (beyond extreme).
MR_MIN_SCORE   = 6.0    # conviction threshold (long ≥ this, short ≤ 10-this)
MR_ADX_MAX     = 25.0   # only trade when ADX < this (ranging). Strong trend = block
MR_ATR_TARGET  = 1.2    # take-profit distance in ATRs (small — back to the mean)
MR_ATR_STOP    = 2.0    # stop distance in ATRs (wide — beyond the extreme)

# ── Daily macro-trend gate (Fix 2) ────────────────────────────────────────────
# Block longs below the daily 200-SMA and shorts above it — the most robust
# trend filter in our research. SMA_BUFFER_PCT dead-band avoids whipsaw at the
# line. Set False to disable (gate degrades to 'unknown' = allow both).
DAILY_TREND_GATE     = True
DAILY_GATE_SMA_PERIOD = 200

# ── Candles to fetch per symbol ───────────────────────────────────────────────
CANDLE_LIMIT = 300

# ── Sentiment ─────────────────────────────────────────────────────────────────
# No API keys required — uses Alternative.me + CoinGecko + RSS feeds
SENTIMENT_WEIGHT    = 0.0      # DISABLED (evidence): news/sentiment gives retail
                               # no edge — it's priced in by the time we read an
                               # RSS feed. It only added noise to the score.
                               # Set >0 only if forward-testing proves it helps.

# ── Machine Learning (FreqAI-inspired, own implementation) ────────────────────
ML_ENABLED          = True     # master switch for ML features
ML_WEIGHT           = 1.0      # how much the ML signal contributes to the score
EXTREMA_WEIGHT      = 1.0      # weight of the extrema (top/bottom) predictor
ML_USE_REGIME       = True     # use ML regime classifier instead of EMA rules
ML_SCALE_POSITIONS  = False    # DISABLED (evidence): the confidence score is
                               # non-monotonic with realised returns — sizing on
                               # it adds risk without adding edge. Size by risk only.
ML_MIN_CONFIDENCE   = 0.60     # below this, ML adds no directional weight
ML_RETRAIN_HOURS    = 6        # retrain every 6h (was 12) — matches the 6-candle
                               # prediction horizon; less staleness / drift risk.
ML_TRAIN_ON_START   = True     # train all models when bot starts
ML_COMPUTE_IMPORTANCE = True   # compute feature importance (dashboard only).
                               # Set False to make training even faster.

# ── Telegram notifications ────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Bot loop ──────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 300    # entry scan cadence (1h candles — 5 min is ample)
POSITION_CHECK_SECONDS = 45    # exit guard cadence — fast loop for stops/targets
DRY_RUN = True                 # True = never place real orders; just log signals

# ── Tier-0 safety: 1h trend engine is research-only ───────────────────────────
# The 1h composite signal has a statistically significant NEGATIVE Information
# Coefficient (anti-predictive, IC≈-0.037, p=0.03) — proven unprofitable on all
# 13 watchlist coins after fees. It must NEVER trade real money. This flag is an
# independent hard block checked in try_open_trade, on top of DRY_RUN.
ENGINE_1H_LIVE_ENABLED = False   # keep False — 1h engine is a DRY_RUN sandbox only

# ── Exchange-side protective orders ───────────────────────────────────────────
USE_EXCHANGE_STOPS = True      # place real stop/TP orders on the exchange (live).
                               # The exchange enforces them instantly even if the
                               # bot is slow/down. DRY_RUN simulates them.
STOP_GAP_BUFFER_PCT = 0.5      # if price gaps this % beyond the stop but the
                               # stop-limit hasn't filled → force a market exit
                               # (gap-through safety net)

# ── Fees & profit discipline ──────────────────────────────────────────────────
# Coinbase Advanced Trade taker fee per side (volume <$10k tier ≈ 0.6%).
# Lower it as your 30-day volume grows. A round trip costs ~2× this.
FEE_RATE_PCT          = 0.6    # taker fee per side, %
MAKER_FEE_RATE_PCT    = 0.4    # maker fee per side, % (lower — for limit entries)
ACCOUNT_FOR_FEES      = True   # subtract fees from realised P&L (true net)

# ── Entry order type ──────────────────────────────────────────────────────────
# "taker"  : market order — fills instantly, higher fee (FEE_RATE_PCT)
# "maker"  : post-only limit at the bid/ask — lower fee, but may not fill
ENTRY_ORDER_TYPE         = "maker"   # CHANGED → maker: post-only limit at the
                                     # bid/ask = 0.4% vs 0.6% taker (~33% less fee
                                     # drag). Falls back to taker if it doesn't
                                     # fill (ENTRY_FALLBACK_TO_TAKER below).
ENTRY_FILL_TIMEOUT_SEC   = 45        # how long to wait for a maker order to fill
ENTRY_FALLBACK_TO_TAKER  = True      # if maker doesn't fill in time → market order
                                     # (False = skip the trade, pure maker-only)

# Profit-floor gate: only OPEN a trade if hitting its take-profit would net at
# least this much AFTER round-trip fees. Trades that can't clear fees + a real
# profit are rejected. Raising this = fewer but higher-quality trades.
MIN_NET_PROFIT_USD    = 1.0    # minimum net $ profit a trade's target must clear
MIN_WIN_FEE_MULTIPLE  = 4.0    # target's gross win must be ≥ this × round-trip fee.
                               # Raised 3→4 — stricter fee discipline; only takes
                               # trades whose reward clearly dwarfs the fee.
                               # The single biggest fee-discipline lever: forces the
                               # bot to only take trades whose reward dwarfs the fee.

# Daily profit lock: once realised net profit for the UTC day reaches this,
# stop opening NEW trades for the rest of the day (lock in gains, don't give
# them back). Open positions keep running with their stops. 0 = disabled.
DAILY_PROFIT_TARGET_USD = 0.0

# ── Circuit breaker (kill switch) ─────────────────────────────────────────────
CB_ENABLED                = True   # master switch
CB_MAX_DAILY_LOSS_PCT     = 6.0    # halt new entries if equity drops 6% in a UTC day
CB_MAX_DRAWDOWN_PCT       = 15.0   # halt if equity falls 15% from its peak
CB_MAX_CONSECUTIVE_LOSSES = 5      # halt after 5 stop-outs in a row
CB_MARKET_CRASH_PCT       = 8.0    # halt if BTC drops 8% in the crash lookback window
CB_CRASH_LOOKBACK_HOURS   = 4      # window for the BTC-crash check (uses 1h candles)
CB_COOLDOWN_HOURS         = 6      # auto-resume this long after a trip


# ══════════════════════════════════════════════════════════════════════════════
#  SMA200 DAILY TREND STRATEGY  (sma_bot.py — the simple, tested-OK approach)
# ══════════════════════════════════════════════════════════════════════════════
# Rule: hold a coin while its daily close is above its 200-day SMA; move to cash
# when it drops below. Equal-weight across coins currently in uptrend. This is
# the ONLY active approach our research endorsed — it doesn't beat buy & hold on
# return, but it roughly halves the drawdown. Shares DRY_RUN with the main bot.
SMA_SYMBOLS      = [s.strip() for s in                      # configurable via .env:
                    os.getenv("SMA_SYMBOLS", "BTC/USD,ETH/USD").split(",")]
                   # e.g. SMA_SYMBOLS=BTC/USD,ETH/USD,SOL/USD
SMA_PERIOD       = 200                       # trend filter length (days)
SMA_CHECK_HOURS  = 6                         # how often to re-check (daily candle
                                             # only changes once/day; 6h catches it)
SMA_ALERT_ONLY   = False                     # False = actually trade it (PAPER while
                                             # DRY_RUN=True; LIVE only if DRY_RUN=False).
                                             # True = only Telegram alerts, no trades.
SMA_BUFFER_PCT   = 0.5                       # require price this % above/below the
                                             # SMA to flip — avoids whipsaw on tiny
                                             # crossings right at the line.
# Weekly macro-regime gate (top-down multi-timeframe, done right):
# only hold when the WEEKLY trend is also up. Coinbase has no native weekly
# candles, so we resample daily → weekly and use the 30-week SMA (the classic
# Weinstein stage-analysis trend filter). Slow TF sets direction; daily times it.
SMA_USE_WEEKLY_GATE = True
SMA_WEEKLY_PERIOD   = 30                      # weeks (30-week SMA ≈ long-term weekly trend)
# Self-served dashboard: when True, `python3 sma_bot.py` also serves the web
# dashboard (so you don't need to run bot.py just for the UI). Open the
# "📈 SMA Trend" tab. Uses the same Flask app; reads sma_state.json live.
SMA_DASHBOARD       = True
SMA_DASHBOARD_PORT  = 8081                    # same port as bot.py — run ONE of them


# ══════════════════════════════════════════════════════════════════════════════
#  TRADINGVIEW WEBHOOK BOT  (tradingview_bot.py — receive alerts, execute on CB)
# ══════════════════════════════════════════════════════════════════════════════
# TradingView sends alert messages (JSON) to a public webhook URL when an
# indicator/strategy fires. This bot receives them, validates, risk-checks, and
# either paper-trades (default) or executes real Coinbase orders. It runs its own
# Flask app serving BOTH the webhook endpoint and a dedicated dashboard.
#
# IMPORTANT — TradingView only POSTs to ports 80/443 on a PUBLIC URL. Run this
# behind a tunnel/reverse proxy (ngrok, cloudflared, Caddy) that forwards 443 →
# WEBHOOK_PORT. See WEBHOOK.md.

WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", 8090))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")   # POST path TradingView hits

# ── Security ──────────────────────────────────────────────────────────────────
# A public webhook that can place trades MUST be authenticated. TradingView has
# no signing, so the convention is a shared passphrase embedded in the alert JSON
# ({"passphrase": "..."}). Compared with hmac.compare_digest (constant-time).
# REQUIRED to be non-empty before any live trading is armed.
WEBHOOK_PASSPHRASE = os.getenv("WEBHOOK_PASSPHRASE", "")

# Optional: only accept POSTs from TradingView's published webhook IPs. Leave the
# allowlist OFF if you sit behind a tunnel/proxy (the source IP becomes the
# proxy's). These are TradingView's documented egress IPs as of 2024.
WEBHOOK_IP_ALLOWLIST_ENABLED = os.getenv("WEBHOOK_IP_ALLOWLIST", "false").lower() == "true"
WEBHOOK_ALLOWED_IPS = [
    "52.89.214.238", "34.212.75.30", "54.218.53.128", "52.32.178.7",
]

# ── Live-trading arm switch (defence in depth) ────────────────────────────────
# Real Coinbase orders are placed ONLY when ALL are true:
#   1. DRY_RUN = False           (global real-money switch)
#   2. WEBHOOK_LIVE_ENABLED = True (this bot's own arm switch)
#   3. API key passes the auth check at startup
#   4. WEBHOOK_PASSPHRASE is set  (no unauthenticated live trading)
# Otherwise the bot PAPER-trades (simulated fills at the live ticker price) —
# fully functional, just no real orders. Mirrors ENGINE_1H_LIVE_ENABLED above.
WEBHOOK_LIVE_ENABLED = os.getenv("WEBHOOK_LIVE_ENABLED", "false").lower() == "true"

# ── Risk caps (protect against malformed / hostile alerts) ────────────────────
# Only symbols on this allowlist can be traded. An empty list means "allow any
# symbol TradingView sends" — NOT recommended when live. Coinbase pairs (BASE/USD).
WEBHOOK_SYMBOL_ALLOWLIST = ["BTC/USD", "ETH/USD", "SOL/USD"]

WEBHOOK_START_CAPITAL_USD   = float(os.getenv("WEBHOOK_START_CAPITAL_USD", 1000))  # paper cash seed
WEBHOOK_DEFAULT_ORDER_USD   = float(os.getenv("WEBHOOK_DEFAULT_ORDER_USD", 100))   # if alert omits size
WEBHOOK_MAX_ORDER_USD       = float(os.getenv("WEBHOOK_MAX_ORDER_USD", 500))       # hard per-trade cap
WEBHOOK_MIN_ORDER_USD       = float(os.getenv("WEBHOOK_MIN_ORDER_USD", 5))         # exchange dust floor
WEBHOOK_MAX_OPEN_POSITIONS  = int(os.getenv("WEBHOOK_MAX_OPEN_POSITIONS", 10))
WEBHOOK_MAX_DAILY_LOSS_USD  = float(os.getenv("WEBHOOK_MAX_DAILY_LOSS_USD", 0))    # 0 = off; kill switch
WEBHOOK_ALLOW_SHORT         = False   # Coinbase spot is long-only; sell = reduce/close a long

# How often the background loop marks open positions to market (equity curve).
WEBHOOK_MARK_INTERVAL_SEC   = int(os.getenv("WEBHOOK_MARK_INTERVAL_SEC", 30))


# ══════════════════════════════════════════════════════════════════════════════
#  CARRY HARVESTER  (carry_bot.py — delta-neutral funding harvest, PAPER)
# ══════════════════════════════════════════════════════════════════════════════
# Earns the perpetual-swap funding premium via long-spot + short-perp of equal
# coin qty (DELTA-NEUTRAL — no price view). Feasibility (Binance funding 2020-2026):
# BTC ~+12%/yr, ETH ~+14%/yr net, positive every year INCLUDING the 2022 bear.
# "Smart" mode sits FLAT when funding turns negative (essential for volatile alts
# like SOL, whose 2022 always-on carry was −38% but +5% with the smart filter).
# Data: OKX public API (Binance/Bybit geo-blocked here; OKX needs no key).
#
# PHASE 1 IS PAPER-ONLY: carry_bot.py has NO live-execution code path, so it
# physically cannot place real orders regardless of any flag. CARRY_LIVE_ENABLED
# is reserved for a future Phase 2 and is a no-op today (defence in depth).
CARRY_TOKENS         = ["BTC", "ETH", "SOL"]            # majors carry best; alts rely on smart
# Data venue for the PAPER test — match the venue you intend to trade live, since
# funding differs by venue (e.g. ETH was OKX +3.6% vs Hyperliquid +11% APR on the
# same day). "hyperliquid" = perp funding/mark from Hyperliquid + spot from Coinbase
# (the real cross-venue book, and US-EC2-safe: both reachable, unlike OKX). "okx"
# = both legs on OKX (non-US hosts only).
CARRY_DATA_VENUE     = os.getenv("CARRY_DATA_VENUE", "hyperliquid")  # hyperliquid | okx
CARRY_QUOTE          = os.getenv("CARRY_QUOTE", "USDT")  # legacy; venue adapter sets its own quote
CARRY_NOTIONAL_USD   = float(os.getenv("CARRY_NOTIONAL_USD", 1000))  # per-leg size per token
CARRY_LEVERAGE       = float(os.getenv("CARRY_LEVERAGE", 2.0))       # short-perp leverage (conservative)
CARRY_SMART          = os.getenv("CARRY_SMART", "true").lower() == "true"  # sit out negative funding
CARRY_MIN_APR        = float(os.getenv("CARRY_MIN_APR", 0.0))   # enter ON when funding APR ≥ this (frac: 0.03=3%)
CARRY_EXIT_APR       = float(os.getenv("CARRY_EXIT_APR", -0.03))  # exit to FLAT when APR ≤ this (hysteresis vs churn)
CARRY_PERP_FEE_PCT   = float(os.getenv("CARRY_PERP_FEE_PCT", 0.05))  # per perp leg per side (%) — OKX taker ~0.05
CARRY_SPOT_FEE_PCT   = float(os.getenv("CARRY_SPOT_FEE_PCT", 0.10))  # per spot leg per side (%) — OKX taker ~0.10
CARRY_POLL_SECONDS   = int(os.getenv("CARRY_POLL_SECONDS", 300))
CARRY_STATE_FILE     = os.getenv("CARRY_STATE_FILE", "carry_state.json")
CARRY_LIVE_ENABLED   = os.getenv("CARRY_LIVE_ENABLED", "false").lower() == "true"  # Phase 2 only; no-op now
# Self-served dashboard: `python3 carry_bot.py` also serves the web UI (🪙 Carry
# tab). Same Flask app + port as bot.py/sma_bot — run ONE server; the others skip
# if the port is taken. The Carry tab reads carry_state.json regardless of host.
CARRY_DASHBOARD      = True
CARRY_DASHBOARD_PORT = 8081