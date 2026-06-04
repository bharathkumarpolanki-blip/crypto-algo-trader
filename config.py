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
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD",
    "DOGE/USD", "ADA/USD", "LINK/USD", "DOT/USD", "UNI/USD",
    "ATOM/USD", "NEAR/USD", "AAVE/USD", "CRV/USD",
]
# Removed: AVAX/USD (consistently -10%), OP/USD (0% win rate across all backtests)

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

# ── Candles to fetch per symbol ───────────────────────────────────────────────
CANDLE_LIMIT = 300

# ── Sentiment ─────────────────────────────────────────────────────────────────
# No API keys required — uses Alternative.me + CoinGecko + RSS feeds
SENTIMENT_WEIGHT    = 1.0      # how much sentiment contributes to signal score

# ── Machine Learning (FreqAI-inspired, own implementation) ────────────────────
ML_ENABLED          = True     # master switch for ML features
ML_WEIGHT           = 1.0      # how much the ML signal contributes to the score
EXTREMA_WEIGHT      = 1.0      # weight of the extrema (top/bottom) predictor
ML_USE_REGIME       = True     # use ML regime classifier instead of EMA rules
ML_SCALE_POSITIONS  = True     # scale position size by ML confidence
ML_MIN_CONFIDENCE   = 0.60     # below this, ML adds no directional weight
ML_RETRAIN_HOURS    = 12       # retrain models every N hours
ML_TRAIN_ON_START   = True     # train all models when bot starts

# ── Telegram notifications ────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Bot loop ──────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 300    # entry scan cadence (1h candles — 5 min is ample)
POSITION_CHECK_SECONDS = 45    # exit guard cadence — fast loop for stops/targets
DRY_RUN = True                 # True = never place real orders; just log signals

# ── Exchange-side protective orders ───────────────────────────────────────────
USE_EXCHANGE_STOPS = True      # place real stop/TP orders on the exchange (live).
                               # The exchange enforces them instantly even if the
                               # bot is slow/down. DRY_RUN simulates them.

# ── Circuit breaker (kill switch) ─────────────────────────────────────────────
CB_ENABLED                = True   # master switch
CB_MAX_DAILY_LOSS_PCT     = 6.0    # halt new entries if equity drops 6% in a UTC day
CB_MAX_DRAWDOWN_PCT       = 15.0   # halt if equity falls 15% from its peak
CB_MAX_CONSECUTIVE_LOSSES = 5      # halt after 5 stop-outs in a row
CB_MARKET_CRASH_PCT       = 8.0    # halt if BTC drops 8% in the crash lookback window
CB_CRASH_LOOKBACK_HOURS   = 4      # window for the BTC-crash check (uses 1h candles)
CB_COOLDOWN_HOURS         = 6      # auto-resume this long after a trip