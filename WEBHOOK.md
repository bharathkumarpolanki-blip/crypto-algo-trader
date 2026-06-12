# TradingView → Coinbase Webhook Bot

Receive buy/sell alerts from **TradingView** over a webhook URL and book the
trades on **Coinbase** — with a full dashboard for portfolio, trade history,
price charts, the live webhook log, and **paper trading** by default.

```
TradingView alert ──HTTPS POST──▶  /webhook  ──▶  risk checks  ──▶  Coinbase order
   (your indicator/strategy)        (this bot)     (caps, auth)      (or paper fill)
                                         │
                                         └──▶  dashboard  http://localhost:8090
```

`tradingview_bot.py` runs **one Flask process** that serves both the webhook
endpoint and the dashboard. It is independent of the other bots in this repo and
keeps its own portfolio ledger in `webhook_state.json`.

---

## Quick start (paper trading — safe, no real money)

```bash
# 1. (optional) set a passphrase so only your alerts are accepted
echo 'WEBHOOK_PASSPHRASE=pick-a-long-random-string' >> .env

# 2. run it
python3 tradingview_bot.py

# 3. open the dashboard
open http://localhost:8090
```

It starts in **PAPER** mode (`DRY_RUN=True` is the default). Every alert is
simulated at the live Coinbase price and tracked in the portfolio — fills,
fees, P&L and the equity curve all behave like the real thing, but no real
orders are placed.

Send a test alert from your terminal:

```bash
curl -X POST http://localhost:8090/webhook \
  -H 'Content-Type: application/json' \
  -d '{"passphrase":"pick-a-long-random-string","action":"buy","symbol":"BTCUSD","order_size":100,"id":"test1"}'
```

You should see it appear under **Webhook Log** and a position under **Overview**.

---

## Connecting TradingView

TradingView only sends webhooks to a **public URL on port 80 or 443**, so you
must expose this bot. The easiest way is a tunnel:

```bash
ngrok http 8090
#   → https://abcd-1234.ngrok-free.app
```

Then in a TradingView alert:

1. **Condition** — your indicator/strategy signal.
2. Tick **Webhook URL** and paste `https://abcd-1234.ngrok-free.app/webhook`.
3. In the **Message** box, paste JSON (the dashboard **Setup** tab shows this
   filled in for you):

```json
{
  "passphrase": "pick-a-long-random-string",
  "action": "{{strategy.order.action}}",
  "symbol": "{{ticker}}",
  "order_size": 150,
  "price": "{{close}}",
  "id": "{{timenow}}",
  "strategy": "my_tv_strategy"
}
```

TradingView substitutes the `{{...}}` placeholders at fire time. For a plain
indicator alert (not a strategy) just hard-code `"action": "buy"` / `"sell"`.

---

## Alert payload reference

| Field        | Required | Meaning |
|--------------|----------|---------|
| `passphrase` | if a server passphrase is set | Shared secret; constant-time compared. |
| `action`     | ✅ | `buy` / `long` → enter or add to a long. `sell` / `close` / `exit` → reduce/close the long. `short` → ignored (Coinbase spot is long-only). |
| `symbol`     | ✅ | `BTCUSD`, `BTC/USD`, `BTC-USD`, `BTCUSDT`, `COINBASE:BTCUSD` … all normalize to a Coinbase `BASE/USD` pair. Stablecoin quotes (USDT/USDC) fold to USD. |
| `order_size` | optional | Trade size. A number = **USD notional** (`150`). A string with `%` = **percent of equity** (`"10%"`). Omit → `WEBHOOK_DEFAULT_ORDER_USD`. |
| `qty`        | optional | Explicit **base units** instead of USD (e.g. `0.01`). |
| `id`         | recommended | Idempotency key — duplicate retries with the same id are dropped. Use `{{timenow}}`. |
| `strategy`   | optional | Free-text label shown in the dashboard. |
| `price`      | optional | Reference price from TV (informational; the bot fills at the live market price). |

`sell` with no explicit size **closes the whole position**; with a size it
reduces by that amount.

---

## Going live (real Coinbase orders)

Live trading is **off by default** and gated behind four independent switches —
all must be true or the bot stays in paper mode:

1. `DRY_RUN = False` in `config.py` (the global real-money switch)
2. `WEBHOOK_LIVE_ENABLED=true` in `.env` (this bot's own arm switch)
3. `WEBHOOK_PASSPHRASE` set (no unauthenticated live trading)
4. Your Coinbase API key passes the startup auth check (View + **Trade**)

When armed, the dashboard shows a red **LIVE** badge and a warning banner, and
the startup banner prints `⚠️ LIVE TRADING ARMED`. The bot still mirrors every
fill into its ledger and also shows your **real exchange USD balance** for
reconciliation.

---

## Risk controls (always on)

- **Passphrase auth** — constant-time compare; bad passphrase → `401`.
- **Symbol allowlist** (`WEBHOOK_SYMBOL_ALLOWLIST`) — only these pairs trade; a
  hostile/typo alert for anything else is rejected.
- **Per-trade cap** (`WEBHOOK_MAX_ORDER_USD`) — clamps notional regardless of
  what the alert asks for; also clamped to available cash.
- **Max open positions** (`WEBHOOK_MAX_OPEN_POSITIONS`).
- **Daily-loss kill switch** (`WEBHOOK_MAX_DAILY_LOSS_USD`, 0 = off) — halts new
  entries after that realized loss for the UTC day; open positions keep running.
- **Idempotency** — duplicate alert ids are dropped (TradingView retries on 5xx).
- **Pause / Flatten All** buttons in the dashboard.
- **Optional IP allowlist** (`WEBHOOK_IP_ALLOWLIST=true`) — restrict to
  TradingView's egress IPs (leave **off** when behind a tunnel/proxy, since the
  source IP becomes the proxy's).
- **Body size cap** (64 KB) and atomic, crash-safe state persistence.

All P&L shown is **TRUE NET** of estimated round-trip fees (`FEE_RATE_PCT`),
consistent with the rest of the project.

---

## Configuration

Everything is in the `TRADINGVIEW WEBHOOK BOT` section of `config.py`, overridable
via `.env` (see `.env.example`). Key knobs:

| `.env` / config | Default | Purpose |
|---|---|---|
| `WEBHOOK_PORT` | `8090` | Port for webhook + dashboard. |
| `WEBHOOK_PATH` | `/webhook` | POST path TradingView hits. |
| `WEBHOOK_PASSPHRASE` | _(empty)_ | Shared secret; required for live. |
| `WEBHOOK_LIVE_ENABLED` | `false` | Arm real trading (needs `DRY_RUN=false` too). |
| `WEBHOOK_START_CAPITAL_USD` | `1000` | Paper portfolio seed. |
| `WEBHOOK_DEFAULT_ORDER_USD` | `100` | Size when an alert omits one. |
| `WEBHOOK_MAX_ORDER_USD` | `500` | Hard per-trade cap. |
| `WEBHOOK_MAX_OPEN_POSITIONS` | `10` | Concurrent position cap. |
| `WEBHOOK_MAX_DAILY_LOSS_USD` | `0` | Daily-loss kill switch (0 = off). |
| `WEBHOOK_SYMBOL_ALLOWLIST` | `BTC,ETH,SOL /USD` | Tradeable pairs (config.py). |

---

## Dashboard tabs

- **Overview** — equity / P&L / cash / unrealized / win-rate cards, equity-curve
  chart, a live markets widget, and open positions with unrealized P&L.
- **Trades** — closed trades with net P&L, %, reason, strategy and mode.
- **Chart** — Coinbase candles (15m/1h/6h/1d) with EMA-21/50 for any allowlisted
  symbol.
- **Webhook Log** — every inbound alert with status (executed / rejected /
  duplicate / error), so you can see exactly what TradingView sent and why.
- **Setup** — your ready-to-paste webhook URL + sample payload + current config.

State lives in `webhook_state.json` (gitignored) and survives restarts.
