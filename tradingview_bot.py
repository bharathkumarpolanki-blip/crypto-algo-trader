"""
TradingView → Coinbase webhook trading bot.

Receives buy/sell alerts from TradingView over a webhook URL, risk-checks them,
and either PAPER-trades (default, simulated fills) or places REAL Coinbase orders
(only when fully armed). Serves a dashboard with portfolio, trade history, price
charts, paper trading and the live webhook log.

Run:   python3 tradingview_bot.py
Then:  open http://localhost:8090  (dashboard + Setup tab with your webhook URL)

Going live (real money) requires ALL of:
  • DRY_RUN = False              (config.py / .env — global real-money switch)
  • WEBHOOK_LIVE_ENABLED = True  (.env — this bot's own arm switch)
  • WEBHOOK_PASSPHRASE set       (.env — no unauthenticated live trading)
  • a Coinbase API key that passes the auth check at startup
Anything missing → it runs in PAPER mode (still fully functional).

TradingView only posts to ports 80/443 on a PUBLIC URL — put this behind a
tunnel/reverse proxy (ngrok, cloudflared, Caddy). See WEBHOOK.md.
"""
from __future__ import annotations

import logging
import socket

import config
from webhook import engine, server
from notifications.notifier import send_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] webhook: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("webhook_bot.log")],
)
logger = logging.getLogger("webhook")


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _banner(status: dict) -> None:
    port = config.WEBHOOK_PORT
    path = config.WEBHOOK_PATH
    mode = status["mode"]
    line = "=" * 64
    print(f"\n{line}")
    print(f"  TRADINGVIEW WEBHOOK BOT   |   mode: {mode}")
    print(line)
    print(f"  Dashboard      : http://localhost:{port}/")
    print(f"  Webhook (local): http://{_local_ip()}:{port}{path}")
    print(f"  Webhook path   : {path}   (POST JSON alerts here)")
    print(f"  Passphrase     : {'SET ✅' if status['passphrase_set'] else 'NOT SET ⚠️  (required for live)'}")
    if mode == "LIVE":
        print(f"  ⚠️  LIVE TRADING ARMED — real Coinbase orders WILL be placed.")
    else:
        why = []
        if config.DRY_RUN:                 why.append("DRY_RUN=True")
        if not config.WEBHOOK_LIVE_ENABLED: why.append("WEBHOOK_LIVE_ENABLED=False")
        if not status["passphrase_set"]:   why.append("no passphrase")
        if config.WEBHOOK_LIVE_ENABLED and not status["auth_ok"] and not config.DRY_RUN:
            why.append(f"auth: {status['auth_msg']}")
        print(f"  Paper mode (simulated fills) — {', '.join(why) or 'live not enabled'}")
    print(f"  Symbols        : {', '.join(config.WEBHOOK_SYMBOL_ALLOWLIST) or '(any)'}")
    print(f"  Order sizing   : default ${config.WEBHOOK_DEFAULT_ORDER_USD:.0f}, "
          f"max ${config.WEBHOOK_MAX_ORDER_USD:.0f}/trade")
    print(line)
    print(f"  TradingView posts only to ports 80/443 on a PUBLIC URL — expose this")
    print(f"  via a tunnel:  ngrok http {port}   (then use the https URL + {path})")
    print(f"{line}\n")


def main() -> None:
    status = engine.init()
    _banner(status)
    try:
        send_telegram(f"📡 *TradingView webhook bot started* — mode `{status['mode']}`, "
                      f"port `{config.WEBHOOK_PORT}`, path `{config.WEBHOOK_PATH}`.")
    except Exception:
        pass
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Webhook bot stopped by user (Ctrl+C). State saved.")
        print("\n👋 Webhook bot stopped. Portfolio state saved — restart any time.")


if __name__ == "__main__":
    main()
