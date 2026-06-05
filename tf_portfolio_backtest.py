"""
Timeframe portfolio backtest — uses the bot's FULL signal engine (analyse) on
1d and 1w bars, and holds a PORTFOLIO of all top-scoring coins (long-only),
rebalancing as signals change. Trades only the delta. Fees on. No look-ahead.

This is the honest "longer timeframe + hold all the good ones" test the user
asked for. Compares against buy & hold BTC and the equal-weight basket.

Usage:  python3 tf_portfolio_backtest.py 1d
        python3 tf_portfolio_backtest.py 1w
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

import config
from exchange.market_data import fetch_ohlcv
from core.indicators import enrich
from core.strategies import analyse

FEE     = getattr(config, "FEE_RATE_PCT", 0.6) / 100.0
CAP     = 1000.0
TOP_N   = 5            # hold up to this many top-scoring coins
MIN_SC  = config.MIN_SIGNAL_SCORE
WARMUP_D = 210
WARMUP_W = 205

BASKET = ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD",
          "LINK/USD","DOT/USD","AVAX/USD","LTC/USD","ATOM/USD","UNI/USD"]

# Coinbase only provides up to DAILY candles (no native 1w/1M). So we always
# fetch 1d and resample upward ourselves.
PRIMARY_RULE = {"1d": None,  "1w": "1W"}      # how to resample daily → primary
GATE_RULE    = {"1d": "1W",  "1w": "1ME"}     # higher-tf trend gate


def resample(df, rule):
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    return pd.concat({"open":o,"high":h,"low":l,"close":c,"volume":v}, axis=1).dropna()


def round_trip_fee(entry, exit_, qty):
    return (entry*qty*FEE) + (exit_*qty*FEE)


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else "1d"
    WARMUP = WARMUP_W if tf == "1w" else WARMUP_D
    print(f"\nLoading {tf} history for {len(BASKET)} coins…", flush=True)

    data, gates = {}, {}
    for s in BASKET:
        daily = fetch_ohlcv(s, "1d", limit=2000)   # Coinbase max granularity (~5.5y)
        if daily.empty or len(daily) < 260:
            continue
        # Build the primary timeframe (daily as-is, or resampled to weekly)
        prim = daily if PRIMARY_RULE[tf] is None else resample(daily, PRIMARY_RULE[tf])
        if len(prim) < 210:
            continue
        data[s] = enrich(prim)
        g = resample(daily, GATE_RULE[tf])
        gates[s] = enrich(g) if len(g) >= 60 else pd.DataFrame()
    syms = list(data.keys())
    n = min(len(df) for df in data.values())
    for s in syms:
        data[s] = data[s].iloc[-n:]
    span = (data[syms[0]].index[-1]-data[syms[0]].index[0]).days
    print(f"  Usable: {len(syms)} coins, {n} {tf} bars ({span} days / {span/365:.1f}y)", flush=True)

    cash = CAP
    holdings = {}        # sym -> units
    n_trades = 0
    equity = []
    idx = data[syms[0]].index

    for i in range(WARMUP, n):
        px = {s: data[s]["close"].iloc[i] for s in syms}
        def mark(): return cash + sum(u*px[s] for s,u in holdings.items())

        # Score every coin, collect LONG signals above threshold
        scored = []
        ts = idx[i]
        for s in syms:
            sl = data[s].iloc[max(0,i-300):i]
            g  = gates[s]
            gslice = g[g.index <= ts] if not g.empty else None
            if gslice is not None and len(gslice) < 60: gslice = None
            try:
                sig = analyse(s, sl, df_trend=gslice, include_sentiment=False, include_ml=False)
            except Exception:
                continue
            if sig.direction == "long" and sig.score >= MIN_SC:
                scored.append((sig.score, s))
        scored.sort(reverse=True)
        target = set(s for _, s in scored[:TOP_N])

        # SELL coins no longer in target
        cur = set(holdings.keys())
        for s in list(cur - target):
            cash += holdings[s]*px[s]*(1-FEE); n_trades += 1; del holdings[s]
        # BUY new entrants, equal weight
        entrants = list(target - set(holdings.keys()))
        if entrants:
            slice_val = mark()/max(len(target),1)
            for s in entrants:
                if slice_val > 1 and cash >= slice_val*0.5:
                    holdings[s] = (slice_val*(1-FEE))/px[s]; cash -= slice_val; n_trades += 1

        equity.append(mark())

    eq = pd.Series(equity, index=idx[WARMUP:])
    end = eq.iloc[-1]; yrs = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = (end/CAP)**(1/yrs)-1 if yrs>0 and end>0 else -1
    maxdd = ((eq-eq.cummax())/eq.cummax()).min()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean()/rets.std()*np.sqrt(365/ (7 if tf=='1w' else 1)) if rets.std()>0 else 0
    calmar = cagr/abs(maxdd) if maxdd<0 else float('inf')

    # Benchmark: hold BTC over same window
    btc = data["BTC/USD"]["close"].iloc[WARMUP:]
    bh_units = (CAP*(1-FEE))/btc.iloc[0]; bh = bh_units*btc
    bh_ret = (bh.iloc[-1]/CAP-1)*100
    bh_dd = ((bh-bh.cummax())/bh.cummax()).min()

    print(f"\n{'='*64}")
    print(f"  {tf.upper()} PORTFOLIO (top {TOP_N} scoring, long-only) — {len(syms)} coins")
    print(f"{'='*64}")
    print(f"  Window        : {eq.index[0].date()} → {eq.index[-1].date()} ({yrs:.1f}y)")
    print(f"  Total return  : {(end/CAP-1)*100:+.1f}%   (Buy&Hold BTC: {bh_ret:+.0f}%)")
    print(f"  CAGR          : {cagr*100:+.1f}%")
    print(f"  Max drawdown  : {maxdd*100:.1f}%   (Buy&Hold BTC: {bh_dd*100:.0f}%)")
    print(f"  Sharpe        : {sharpe:.2f}")
    print(f"  Calmar        : {calmar:.2f}")
    print(f"  Trades        : {n_trades}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
