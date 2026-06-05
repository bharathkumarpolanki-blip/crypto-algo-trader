"""
1h vs 1d strategy comparison under Coinbase One (zero trading fees).

Coinbase One ≈ $29.99/month = $360/year subscription, zero trading fees
(small residual spread). We model FEE=0 and then subtract the subscription
against $1,000 capital — which is itself a huge drag worth seeing.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import config
# Zero trading fees (Coinbase One). Disable the fee-multiple gate too (no fees).
config.FEE_RATE_PCT       = 0.0
config.MIN_WIN_FEE_MULTIPLE = 0.0
config.MIN_NET_PROFIT_USD = 0.0

from backtest import run_backtest
import daily_backtest as dbt

SUBSCRIPTION_PER_YEAR = 29.99 * 12   # Coinbase One
CAPITAL = 1000.0


def hr(): print("=" * 76)

print("\n1-HOUR strategy at ZERO fees (Coinbase One) — does removing fees help?")
hr()
oneh = {}
for sym in ["BTC/USD", "SOL/USD"]:
    t = time.time()
    r = run_backtest(sym, "1h", 175, CAPITAL, config.RISK_PER_TRADE_PCT, verbose=False)
    if r and r.get("trades"):
        oneh[sym] = r
        print(f"  {sym:9s}  {r['trades']:3d} trades  "
              f"return {r['return_pct']:+7.1f}%  PF {r['profit_factor']:.2f}  "
              f"({time.time()-t:.0f}s, ~175 days)")
    else:
        print(f"  {sym}: no trades")

print("\n1-DAY trend-following at ZERO fees (Coinbase One) — 5.5 years")
hr()
dbt.FEE = 0.0
for sym in ["BTC/USD", "ETH/USD"]:
    df = dbt.fetch_ohlcv(sym, "1d", limit=2000)
    if df.empty or len(df) < 250:
        continue
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    eq, trades = dbt.run(df, dbt.sig_sma200, CAPITAL)
    m = dbt.metrics(eq, trades, df)
    print(f"  {sym:9s}  SMA200  {m['trades']:3d} trades over {yrs:.1f}y  "
          f"total {m['ret_pct']:+.0f}%  CAGR {m['cagr_pct']:+.0f}%  MaxDD {m['maxdd_pct']:.0f}%")

print("\nThe subscription drag on $1,000 capital")
hr()
print(f"  Coinbase One costs ${SUBSCRIPTION_PER_YEAR:.0f}/year.")
print(f"  On ${CAPITAL:.0f} capital that is "
      f"{SUBSCRIPTION_PER_YEAR/CAPITAL*100:.0f}% of your capital PER YEAR,")
print(f"  just to be allowed to trade fee-free. You must out-earn that first.")
print()
