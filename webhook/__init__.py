"""TradingView webhook trading bot — receive alerts, execute on Coinbase.

Modules:
  store   — thread-safe persistent paper/live portfolio ledger (atomic JSON)
  engine  — parse + validate + risk-check + execute a TradingView alert
  server  — Flask app: /webhook endpoint + the dashboard + JSON API
"""
