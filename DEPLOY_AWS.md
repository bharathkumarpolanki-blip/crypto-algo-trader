# Deploying on AWS EC2 (Free Tier) — sma_bot.py + LuxAlgoSMCBot

Run both bots 24/7 on one free EC2 instance. `sma_bot.py` (from the
`crypto-algo-trader` repo) has no web server — it just sends Telegram alerts /
trades. **LuxAlgoSMCBot** needs a public HTTPS URL for the TradingView webhook,
served via Caddy.

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
`sma_bot.py` needs only 4 packages (skip the ML stack `bot.py` uses):
```bash
cd ~/trading-bot-ccode
python3 -m venv venv && ./venv/bin/pip install -U pip
./venv/bin/pip install ccxt pandas python-dotenv requests
```
**Copy your `.env` up — run on (Mac):**
```bash
scp -i ~/Downloads/your-key.pem ~/trading-bot-ccode/.env ubuntu@YOUR_ELASTIC_IP:~/trading-bot-ccode/.env
```
> `sma_bot.py` defaults to `SMA_ALERT_ONLY=True` (Telegram alerts only, no trades).

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

## H. HTTPS for the webhook — Caddy + DuckDNS (EC2)
1. At <https://www.duckdns.org> create `YOUR-SUBDOMAIN.duckdns.org` → set IP = your Elastic IP.
2. ```bash
   sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
   YOUR-SUBDOMAIN.duckdns.org {
       reverse_proxy 127.0.0.1:8095
   }
   EOF
   sudo systemctl reload caddy
   ```
3. Webhook URL → `https://YOUR-SUBDOMAIN.duckdns.org/webhook` (put in TradingView alerts).
   Dashboard → `https://YOUR-SUBDOMAIN.duckdns.org/`.

## I. Verify + operate (EC2)
```bash
journalctl -u luxsmc -f                 # SMC bot live logs
journalctl -u smabot -n 50 --no-pager   # SMA bot
# update later:
cd ~/LuxAlgoSMCBot && git pull && ./venv/bin/pip install -r requirements.txt && sudo systemctl restart luxsmc
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
