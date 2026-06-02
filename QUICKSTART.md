# Crypto Trading Bot — Quick Start

## File Architecture

```
bot.py              ← Main loop (scan → signal → trade → monitor → dashboard)
strategies.py       ← 14-component signal engine + 2 hard gates (score 0-10)
candle_patterns.py  ← 24 candlestick pattern detectors
indicators.py       ← All TA indicators (EMA, MACD, RSI, BB, ADX, OBV, Ichimoku, Supertrend…)
risk_manager.py     ← Position sizing, trailing stops, portfolio risk caps
market_data.py      ← Coinbase CCXT wrapper — paginated OHLCV, order placement
sentiment.py        ← Free sentiment: Fear & Greed + CoinGecko + RSS feeds (no keys needed)
universe.py         ← Auto-discovers top 15 trending coins from 390+ Coinbase pairs hourly
notifier.py         ← Telegram alerts on open/close
backtest.py         ← Walk-forward backtester with 6h trend gate
state.py            ← Thread-safe shared state (bot ↔ dashboard)
dashboard.py        ← Flask web dashboard + REST API
config.py           ← All tunable parameters
```

---

## Signal Engine — 14 Components + 2 Hard Gates

### Hard Gates (trade BLOCKED if either fails)

| Gate | Logic |
|---|---|
| **6h Trend Gate** | Classifies 6h chart as bull/bear/neutral using EMA ordering, EMA21 slope, ADX+DI, Supertrend, and 7-day ROC. LONG blocked when bear; SHORT blocked when bull |
| **1h Momentum Gate** | EMA9 must be above EMA21 for LONG; below for SHORT. 10-bar ROC must agree |

### Scoring Components (each −2 to +2, normalised to 0-10)

| # | Component | What it checks |
|---|---|---|
| 1 | **EMA Stack** | 9/21/50/200 EMA alignment; full bearish inversion = hard −2 |
| 2 | **MACD** | Crossover + histogram expanding in signal direction |
| 3 | **RSI** | Regime-aware: bull=50-70 zone; bear=below 50 falling; neutral=oversold bounce |
| 4 | **Stochastic** | %K/%D cross in oversold (<25) or overbought (>75) zones only |
| 5 | **Bollinger Bands** | Trend-aware: upper-half riding (bull), lower-half (bear), squeeze detection |
| 6 | **ADX** | ADX ≥ 22 required; below = −1 ranging penalty; DI+/DI− direction |
| 7 | **Volume** | Vol ratio ≥ 0.8 required; surge (>1.5×) on trend-confirming candle rewarded |
| 8 | **Supertrend** | Fresh flip = ±2.0; ongoing trend = ±1.0 |
| 9 | **Ichimoku** | Price vs cloud + TK cross + cloud colour |
| 10 | **Support Bounce** | Price within 0.4×ATR of a pivot low |
| 11 | **Market Regime** | Strong bull = +1.5; confirmed bear = −3.0 (heavy veto) |
| 12 | **6h Trend Bonus** | EMA trend + Supertrend on 6h, weighted 0.75× |
| 13 | **Sentiment** | Fear & Greed (30%) + CoinGecko votes (40%) + RSS headlines (30%) |
| 14 | **Candle Patterns** | 24 patterns combined, capped at ±3.0 (see below) |

**Minimum score to trade: 5.5 / 10 | Minimum R:R: 2.0**

---

## 24 Candlestick Patterns

### Bullish Reversal
Hammer · Inverted Hammer · Bullish Engulfing · Morning Star · Piercing Line · Bullish Harami · Tweezer Bottom

### Bearish Reversal
Shooting Star · Hanging Man · Bearish Engulfing · Evening Star · Dark Cloud Cover · Bearish Harami · Tweezer Top

### Bullish Continuation
Three White Soldiers · Green Marubozu · Rising Three Methods · Three Inside Up

### Bearish Continuation
Three Black Crows · Red Marubozu · Falling Three Methods · Three Inside Down

### Indecision
Doji · Spinning Top

---

## Risk & Capital Management

| Rule | Value |
|---|---|
| Risk per trade | 2% of capital (= $20 on $1,000) |
| Max single position | 20% of capital (= $200 on $1,000) |
| Max open positions | 5 simultaneously |
| Max portfolio risk | 10% at once (= $100 on $1,000) |
| Stop loss | Entry ± 1.5 × ATR |
| Take profit | Entry ± 4.0 × ATR → R:R ≈ 2.67 |
| Trailing stop | Activates after 1 ATR profit; trails at 2.0 × ATR |
| Score scaling | 1.25× for score ≥ 6.5; 1.5× for score ≥ 7.5 |
| Cooldown | Skip 6 candles after 2 consecutive stop losses |

**Worst case on $1,000:** 5 simultaneous stop losses = $100 loss (10%)

---

## Dynamic Universe

Every hour the bot scans all 390+ active Coinbase USD pairs and selects the **top 15** ranked by:
- 24h volume (liquidity filter)
- 24h price change momentum
- 7-day trend direction
- Minimum price ($0.01) and volume filters

Within those 15, each 5-minute scan further narrows to the **top 60% by 14-bar ROC** — so the bot always focuses on the coins moving most right now.

---

## Setup

```bash
# 1. Install dependencies
pip3 install ccxt pandas numpy ta requests python-dotenv flask flask-cors \
             feedparser tabulate colorama

# 2. Copy and fill in Coinbase API keys
cp .env.example .env
# Edit .env — add your API_KEY and API_SECRET from coinbase.com → Settings → API

# 3. Run (dashboard auto-starts at http://localhost:8081)
python3 bot.py

# 4. Run a backtest from terminal
python3 backtest.py --symbol BTC/USD --days 90
python3 backtest.py --days 90 --multi        # all watchlist symbols

# 5. Or run a backtest from the web dashboard
# → Open http://localhost:8081 → click 🧪 Backtest tab → ▶ Run Backtest
```

---

## Web Dashboard — http://localhost:8081

| Tab | What you see |
|---|---|
| **📊 Live Trading** | Portfolio value, equity curve, open positions, signal scanner, trade history |
| **🧪 Backtest** | Run any symbol/period, live progress, per-symbol results, bar chart, drill-down trade log |
| **🌍 Universe** | Currently active coins, momentum scores, last refresh time |

**Click any symbol row** in the Signal Scanner, Positions, or Trade History to open a full chart drawer with:
- Candlestick chart (1h / 6h / 1d toggle)
- EMA 9/21/50/200 overlays + Bollinger Bands
- Volume bars, RSI panel, MACD panel
- Trade entry/exit markers on the chart
- Active candlestick patterns
- Full trade log for that symbol

---

## Key Config Knobs (config.py)

| Setting | Default | Meaning |
|---|---|---|
| `DRY_RUN` | `True` | No real orders placed — safe paper trading |
| `TOTAL_CAPITAL_USDT` | 1000 | Your starting capital in USD |
| `RISK_PER_TRADE_PCT` | 2.0 | % of capital risked per trade |
| `MAX_OPEN_POSITIONS` | 5 | Max concurrent open positions |
| `MIN_SIGNAL_SCORE` | 5.5 | Min score to open a trade (0-10) |
| `STRONG_SIGNAL_SCORE` | 7.5 | Scale up position size above this |
| `ATR_STOP_MULTIPLIER` | 1.5 | Stop = entry ± ATR × 1.5 |
| `ATR_TARGET_MULTIPLIER` | 4.0 | Target = entry ± ATR × 4.0 (R:R ≈ 2.67) |
| `SCAN_INTERVAL_SECONDS` | 300 | Scan every 5 minutes |
| `TF_PRIMARY` | `1h` | Signal generation timeframe |
| `TF_TREND` | `6h` | Trend gate timeframe (Coinbase: 1h/2h/6h/1d only) |

---

## Going Live (Coinbase)

1. Set `DRY_RUN = False` in `config.py`
2. Go to **coinbase.com → Settings → API** and create an API key
   - Enable: **View** + **Trade** permissions only
   - Never enable **Transfer** — the bot does not need it
3. Add keys to `.env`:
   ```
   API_KEY=your_key
   API_SECRET="-----BEGIN EC PRIVATE KEY-----
   ...your key body...
   -----END EC PRIVATE KEY-----"
   ```
4. Fund your Coinbase account with USD
5. Start with small capital ($100–$200) to validate live behaviour before scaling

> **Coinbase has no sandbox.** Always paper-trade (`DRY_RUN=True`) for at least 2 weeks before going live.

---

## Sentiment Sources (all free, no API keys needed)

| Source | Weight | Updates |
|---|---|---|
| Alternative.me Fear & Greed Index | 30% | Daily |
| CoinGecko community votes + 24h price change | 40% | Every 15 min |
| RSS headlines (Bitcoinist, CryptoSlate, Decrypt, Bitcoin.com) | 30% | Every 20 min |

---

## Telegram Alerts (optional)

1. Create a bot via **@BotFather** → get `TELEGRAM_BOT_TOKEN`
2. Get your chat ID via **@userinfobot** → set `TELEGRAM_CHAT_ID`
3. Add both to `.env`
4. You'll receive alerts on every signal, trade open, and trade close

---

## Recommended Go-Live Checklist

- [ ] Paper traded (`DRY_RUN=True`) for at least 2 weeks
- [ ] Backtest run on all watchlist symbols (`--multi --days 90`)
- [ ] Profitable symbols identified (PF > 1.0)
- [ ] Coinbase API key created with **View + Trade only** (no Transfer)
- [ ] Starting capital decided (recommend $100–$500 for first month)
- [ ] Telegram alerts configured so you know when trades open/close
- [ ] `DRY_RUN = False` set in config.py

---

## Risk Warnings

- Past backtested performance does not guarantee future results
- Crypto markets are highly volatile — prices can move 10-20% in hours
- Always start with paper trading and small capital
- The bot trades **spot markets only** — no leverage
- Never trade with money you cannot afford to lose
- A black swan event (exchange hack, regulatory ban) can bypass all stop losses
