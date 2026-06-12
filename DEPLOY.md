# Deploying the bots 24/7 on Oracle Cloud (Always Free)

This repo ships three runnable bots. To run any of them unattended you want an
**always-on** host (a laptop that sleeps drops webhooks *and* suspends the
position-monitor/stop-loss loop). Oracle Cloud's Always Free Ampere A1 VM
(**2 OCPU / 12 GB RAM**, free forever, public IP) hosts all of them.

| Bot | What it is | Web port | Needs a public URL? |
|-----|-----------|----------|---------------------|
| `bot.py` | 1h multi-timeframe engine + ML + dashboard | **8081** (dashboard) | No (dashboard only) |
| `sma_bot.py` | SMA200 daily-trend bot (alerts/paper) | none (writes `sma_state.json`, shown in `bot.py`'s dashboard) | No |
| `tradingview_bot.py` | TradingView webhook → Coinbase + dashboard | **8090** (webhook + dashboard) | **Yes** (TradingView posts here) |

> One VM can also host `LuxAlgoSMCBot` (port 8095) — see that repo's `DEPLOY.md`.
> All bots coexist on different ports behind one Caddy.

---

## 1. Create the VM + open ports + install deps

Do **Steps 1–3 of `LuxAlgoSMCBot/DEPLOY.md`** (identical): create the Ampere A1
Ubuntu 24.04 VM, open ports **80/443** (OCI Security List **and** host iptables),
and install `python3-venv git caddy`. If you already made that VM, reuse it.

---

## 2. Get the code + configure (one clone, one venv, one .env — shared by all three)

```bash
cd ~
git clone <YOUR_GIT_REMOTE> trading-bot-ccode    # or rsync the folder up
cd ~/trading-bot-ccode

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

cp .env.example .env
nano .env     # API keys (live only), Telegram, WEBHOOK_PASSPHRASE; keep DRY_RUN=true to paper
```

> **Shared config caveat:** all three bots read the same `config.py`/`.env`, so
> `DRY_RUN` is global. Each bot also has its **own** live arm-switch on top of it:
> `ENGINE_1H_LIVE_ENABLED` (keep **False** — the 1h engine is research-only),
> `SMA_ALERT_ONLY`, and `WEBHOOK_LIVE_ENABLED`. Real money needs `DRY_RUN=false`
> **and** the relevant bot's switch on.

---

## 3. systemd services (auto-restart + start on boot)

Create one unit per bot. Run only the ones you want.

**Main 1h bot + dashboard (port 8081):**
```bash
sudo tee /etc/systemd/system/tradingbot.service >/dev/null <<'EOF'
[Unit]
Description=1h Trading Bot (bot.py) + dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-bot-ccode
ExecStart=/home/ubuntu/trading-bot-ccode/venv/bin/python bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
```

**SMA200 daily bot (no port; feeds the dashboard's SMA tab):**
```bash
sudo tee /etc/systemd/system/smabot.service >/dev/null <<'EOF'
[Unit]
Description=SMA200 Daily Trend Bot (sma_bot.py)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-bot-ccode
ExecStart=/home/ubuntu/trading-bot-ccode/venv/bin/python sma_bot.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
```

**TradingView webhook bot (port 8090):**
```bash
sudo tee /etc/systemd/system/tvwebhook.service >/dev/null <<'EOF'
[Unit]
Description=TradingView Webhook Bot (tradingview_bot.py)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-bot-ccode
ExecStart=/home/ubuntu/trading-bot-ccode/venv/bin/python tradingview_bot.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
```

Enable the ones you want:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tradingbot smabot tvwebhook   # pick any subset
systemctl status tradingbot --no-pager
journalctl -u tvwebhook -f       # live logs
```

---

## 4. HTTPS with Caddy

Create free DuckDNS subdomains pointed at the VM IP (e.g. `mainbot.duckdns.org`,
`tvbot.duckdns.org`), then:
```bash
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
# Main bot dashboard (read-only UI)
mainbot.duckdns.org {
    reverse_proxy 127.0.0.1:8081
}
# TradingView webhook bot (webhook + dashboard)
tvbot.duckdns.org {
    reverse_proxy 127.0.0.1:8090
}
EOF
sudo systemctl reload caddy
```
- Main dashboard: `https://mainbot.duckdns.org/`
- **TradingView webhook URL:** `https://tvbot.duckdns.org/webhook`

> If you also run `LuxAlgoSMCBot`, add a third block (`smcbot.duckdns.org →
> 127.0.0.1:8095`) to the same Caddyfile.

---

## 5. Operate / update

```bash
sudo systemctl restart tvwebhook                 # after editing .env / new code
journalctl -u tradingbot -n 200 --no-pager       # recent logs
cd ~/trading-bot-ccode && git pull && ./venv/bin/pip install -r requirements.txt
sudo systemctl restart tradingbot smabot tvwebhook
```

## Per-bot live-trading notes
- **`bot.py`:** keep `ENGINE_1H_LIVE_ENABLED=False` — the 1h engine is a proven
  net-negative and is research-only. Run it for the dashboard/ML, not real orders.
- **`sma_bot.py`:** `SMA_ALERT_ONLY=True` sends Telegram alerts only; set it False
  (and `DRY_RUN=false`) to actually trade.
- **`tradingview_bot.py`:** real orders need `DRY_RUN=false` + `WEBHOOK_LIVE_ENABLED=true`
  + passphrase + valid API key (see `WEBHOOK.md`).

## Security
- `.env`, `positions.json`, `webhook_state.json`, `sma_state.json`, `models/`,
  logs are gitignored — keep them on the server only.
- Only 80/443 are public; bot ports (8081/8090) are localhost-only behind Caddy.
