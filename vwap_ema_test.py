"""
STANDALONE test — VWAP entry / EMA9 exit. Touches nothing in the project except
the read-only data fetcher.

Strategy (long-only, the user's idea):
  • ENTRY: price crosses ABOVE the daily-anchored VWAP (intraday fair value).
  • EXIT : price crosses BELOW the EMA9.
  • Fees 0.6%/side + 0.05% slippage. 1h candles (VWAP is an intraday tool).

Reports per-coin backtest vs Buy & Hold, plus an out-of-sample (70/30) split as
a forward-test proxy. No optimization.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

FEE = 0.006; SLIP = 0.0005; RT = FEE + SLIP
CAP = 1000.0
COINS = ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","LINK/USD","ADA/USD"]


def daily_vwap(df):
    """Daily-anchored VWAP on intraday bars (resets each UTC day)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    day = df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_v  = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def signals(df):
    vwap = daily_vwap(df)
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    c = df["close"]
    above_vwap = c > vwap
    below_ema  = c < ema9
    cross_up   = above_vwap & ~above_vwap.shift(1).fillna(False)
    cross_down = below_ema & ~below_ema.shift(1).fillna(False)
    return cross_up.values, cross_down.values


def run(df):
    c = df["close"].values; cu, cd = signals(df)
    cash = CAP; units = 0.0; entry = 0.0
    eq = []; trades = []
    for i in range(len(df)):
        if units == 0 and cu[i]:
            buy = c[i]*(1+SLIP); units = (cash*(1-FEE))/buy; entry = buy; cash = 0
        elif units > 0 and cd[i]:
            sell = c[i]*(1-SLIP); cash = units*sell*(1-FEE)
            trades.append((sell-entry)/entry); units = 0
        eq.append(cash + units*c[i])
    if units > 0:
        sell=c[-1]*(1-SLIP); cash=units*sell*(1-FEE); trades.append((sell-entry)/entry)
        eq[-1]=cash
    return pd.Series(eq, index=df.index), trades


def stats(eq, trades):
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1
    end=eq.iloc[-1]; ret=(end/CAP-1)*100
    cagr=((end/CAP)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sharpe=r.mean()/r.std()*np.sqrt(24*365) if r.std()>0 else 0
    wins=[t for t in trades if t>0]; wr=len(wins)/len(trades)*100 if trades else 0
    ls=abs(sum(t for t in trades if t<=0)); pf=sum(wins)/ls if ls>0 else (float('inf') if wins else 0)
    return ret,cagr,sharpe,dd,len(trades),wr,pf


def bh(df):
    eq=(CAP*(1-FEE))/df["close"].iloc[0]*df["close"]
    return stats(eq, [])


def main():
    print("VWAP-entry / EMA9-exit | 1h | fees 0.6%/side + 0.05% slip | long-only\n")
    print(f"| {'Coin':5s} | {'Strat Ret':>9s} | {'CAGR':>6s} | {'Sharpe':>6s} | {'MaxDD':>6s} | {'Trades':>6s} | {'WR':>4s} | {'PF':>4s} | {'B&H Ret':>7s} |")
    print("|"+"-"*7+"|"+"-"*11+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|"+"-"*6+"|"+"-"*6+"|"+"-"*9+"|")
    agg=[]
    store={}
    for sym in COINS:
        df=fetch_ohlcv(sym,"1h",limit=5000)
        if df.empty or len(df)<500: print(f"| {sym.split('/')[0]:5s} | no data |"); continue
        store[sym]=df
        eq,tr=run(df); ret,cagr,sh,dd,n,wr,pf=stats(eq,tr)
        br=bh(df)[0]
        agg.append((sym,ret,pf,br))
        print(f"| {sym.split('/')[0]:5s} | {ret:+8.1f}% | {cagr:+5.0f}% | {sh:+6.2f} | {dd:5.0f}% | {n:6d} | {wr:3.0f}% | {pf:.2f} | {br:+6.0f}% |")
    span=(store[COINS[0]].index[-1]-store[COINS[0]].index[0]).days if store else 0
    print(f"\n  (tested span ≈ {span} days of 1h data — Coinbase retains ~{span}d)")

    # ── Out-of-sample (forward-test proxy): first 70% vs last 30% ──────────────
    print("\nOUT-OF-SAMPLE (first 70% vs last 30% — forward-test proxy):")
    print(f"| {'Coin':5s} | {'IS Ret':>7s} | {'IS PF':>5s} | {'OOS Ret':>7s} | {'OOS PF':>6s} |")
    print("|"+"-"*7+"|"+"-"*9+"|"+"-"*7+"|"+"-"*9+"|"+"-"*8+"|")
    for sym,df in store.items():
        cut=int(len(df)*0.7)
        ins=df.iloc[:cut]; oos=df.iloc[cut:]
        ei,ti=run(ins); eo,to=run(oos)
        _,_,_,_,_,_,pfi=stats(ei,ti); ri=(ei.iloc[-1]/CAP-1)*100
        _,_,_,_,_,_,pfo=stats(eo,to); ro=(eo.iloc[-1]/CAP-1)*100
        print(f"| {sym.split('/')[0]:5s} | {ri:+6.1f}% | {pfi:.2f} | {ro:+6.1f}% | {pfo:.2f} |")

    prof=sum(1 for _,r,_,_ in agg if r>0); beat=sum(1 for _,r,_,b in agg if r>b)
    print(f"\n  Profitable on {prof}/{len(agg)} coins.  Beat Buy&Hold on {beat}/{len(agg)}.")


if __name__=="__main__":
    main()
