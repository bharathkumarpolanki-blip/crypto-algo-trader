# Deploying on AWS EC2 (Free Tier) — sma_bot.py + LuxAlgoSMCBot

Run both bots 24/7 on one free EC2 instance.

- **`sma_bot.py`** (from `crypto-algo-trader`) runs on port **8081** and serves
  its own web dashboard (SMA trend status, paper P&L, backtest chart). Exposed
  via a second Caddy vhost so you get HTTPS on a second DuckDNS subdomain.
- **LuxAlgoSMCBot** needs a public HTTPS URL for the TradingView webhook,
  served via Caddy on port **8095**.

> Placeholders below: replace `YOUR_ELASTIC_IP`, `your-key.pem`, and
> `YOUR-SUBDOMAIN.duckdns.org` with your real values. Never commit your `.pem`
> key or `.env` files.

Each block is marked **(Mac)** or **(EC2)**.

---

## A. Launch the instance (AWS console)
1. **EC2 → Launch instance.**
2. **AMI:** Ubuntu Server 24.04 LTS → **64-bit (Arm)**.
3. **Instance type:** **`t4g.small`** (2 GB RAM, free trial through end-2026).
   *(t3.micro/1 GB works too — then add swap, see Notes.)*
4. **Key pair:** create new → type **ED25519**, format **.pem** → download it.
5. **Network → Edit → Security group** inbound rules:
   - SSH **22** → *My IP*
   - HTTP **80** → *Anywhere (0.0.0.0/0)*
   - HTTPS **443** → *Anywhere (0.0.0.0/0)*
   *(Caddy proxies both bots over 443 — no need to expose 8081 or 8095 directly.)*
6. **Storage:** 20–30 GB gp3.
7. **Launch.**
8. *(Recommended)* **Elastic IP:** EC2 → Network & Security → **Elastic IPs** →
   *Allocate* → select it → *Actions → Associate* → Instance → your instance.
   This gives a permanent public IP for the webhook.

## B. Connect (Mac)
```bash
chmod 400 ~/Downloads/your-key.pem
ssh -i ~/Downloads/your-key.pem ubuntu@YOUR_ELASTIC_IP
uname -m            # expect: aarch64
```

## C. Install dependencies (EC2)
```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install python3-venv python3-pip git curl

# GitHub CLI
(type -p wget >/dev/null || sudo apt install wget -y)
sudo mkdir -p -m 755 /etc/apt/keyrings
wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
sudo apt update && sudo apt -y install gh

# Caddy (auto-HTTPS reverse proxy for the webhook)
sudo apt -y install debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
sudo apt update && sudo apt -y install caddy
```

## D. GitHub auth + clone (EC2)
```bash
gh auth login          # GitHub.com → HTTPS → "Login with a one-time code"
                       # open https://github.com/login/device and enter the code
gh auth setup-git
cd ~
git clone -b dev https://github.com/bharathkumarpolanki-blip/crypto-algo-trader.git trading-bot-ccode
git clone https://github.com/bharathkumarpolanki-blip/LuxAlgoSMCBot.git
```

## E. Set up sma_bot.py — lean install (EC2)
`sma_bot.py` skips the heavy ML stack. Install only what it needs — including
`flask` and `flask-cors` for its built-in web dashboard (port 8081):
```bash
cd ~/trading-bot-ccode
python3 -m venv venv && ./venv/bin/pip install -U pip
./venv/bin/pip install ccxt pandas numpy python-dotenv requests flask flask-cors
```
**Copy your `.env` up — run on (Mac):**
```bash
scp -i ~/Downloads/your-key.pem ~/trading-bot-ccode/.env ubuntu@YOUR_ELASTIC_IP:~/trading-bot-ccode/.env
```
> `SMA_ALERT_ONLY=False` and `DRY_RUN=True` in `config.py` → **PAPER** mode.
> The dashboard starts automatically with the bot (`SMA_DASHBOARD=True`).

## F. Set up LuxAlgoSMCBot (EC2)
```bash
cd ~/LuxAlgoSMCBot
python3 -m venv venv && ./venv/bin/pip install -U pip
./venv/bin/pip install -r requirements.txt
```
**Copy its `.env` up — run on (Mac):**
```bash
scp -i ~/Downloads/your-key.pem ~/LuxAlgoSMCBot/.env ubuntu@YOUR_ELASTIC_IP:~/LuxAlgoSMCBot/.env
```

## G. systemd services (auto-restart + boot) (EC2)
```bash
sudo tee /etc/systemd/system/smabot.service >/dev/null <<'EOF'
[Unit]
Description=SMA200 Daily Trend Bot
After=network-online.target
Wants=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-bot-ccode
ExecStart=/home/ubuntu/trading-bot-ccode/venv/bin/python sma_bot.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/luxsmc.service >/dev/null <<'EOF'
[Unit]
Description=LuxAlgo SMC Bot
After=network-online.target
Wants=network-online.target
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/LuxAlgoSMCBot
ExecStart=/home/ubuntu/LuxAlgoSMCBot/venv/bin/python run.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now smabot luxsmc
systemctl status luxsmc --no-pager
```

## H. HTTPS for both dashboards — Caddy + DuckDNS (EC2)
Create **two** DuckDNS subdomains — both pointing at your Elastic IP:

| Subdomain | Bot | Port |
|-----------|-----|------|
| `YOUR-SMC-SUBDOMAIN.duckdns.org` | LuxAlgoSMCBot (webhook + dashboard) | 8095 |
| `YOUR-SMA-SUBDOMAIN.duckdns.org` | sma_bot.py dashboard | 8081 |

1. At <https://www.duckdns.org> create both subdomains → IP = your Elastic IP.
2. ```bash
   sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
   # LuxAlgo SMC Bot — TradingView webhook + dashboard
   YOUR-SMC-SUBDOMAIN.duckdns.org {
       reverse_proxy 127.0.0.1:8095
   }

   # SMA200 Trend Bot — dashboard only (no public webhook needed)
   YOUR-SMA-SUBDOMAIN.duckdns.org {
       reverse_proxy 127.0.0.1:8081
   }
   EOF
   sudo systemctl reload caddy
   ```
3. URLs:
   - **SMC webhook** (put in TradingView alerts): `https://YOUR-SMC-SUBDOMAIN.duckdns.org/webhook`
   - **SMC dashboard**: `https://YOUR-SMC-SUBDOMAIN.duckdns.org/`
   - **SMA dashboard**: `https://YOUR-SMA-SUBDOMAIN.duckdns.org/` (open the 📈 SMA Trend tab)

## I. Verify + operate (EC2)
```bash
journalctl -u luxsmc -f                 # SMC bot live logs
journalctl -u smabot -n 50 --no-pager   # SMA bot

# Test that both dashboards are up (replace with your real subdomains):
curl -s https://YOUR-SMC-SUBDOMAIN.duckdns.org/ | head -5
curl -s https://YOUR-SMA-SUBDOMAIN.duckdns.org/ | head -5

# Update bots later:
cd ~/LuxAlgoSMCBot && git pull && ./venv/bin/pip install -r requirements.txt && sudo systemctl restart luxsmc
cd ~/trading-bot-ccode && git pull -b dev && sudo systemctl restart smabot
```

---

## Notes
- **t3.micro (1 GB)?** Add swap so pip/pandas don't OOM:
  ```bash
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```
- Both bots **rotate logs daily (30-day retention)** and **restart on crash/reboot** (systemd).
- `.env` and `*.pem` stay on their machines only — never commit them (both are gitignored).
- AWS free EC2 hours and the t4g free trial are **time-limited** — Oracle's Ampere A1 is free
  forever if you can get capacity (see `DEPLOY.md`).
