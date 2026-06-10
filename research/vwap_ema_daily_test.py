"""
STANDALONE — VWAP entry / EMA9 exit on DAILY candles (lower frequency).

Daily-anchored VWAP is meaningless on daily bars (1 bar/day), so entries use a
ROLLING N-day VWAP (volume-weighted avg price over the window) — a legit HTF
'rolling VWAP'. Exit on EMA9 (9-day) cross-down. Long-only, fees+slip.

~6 years of daily data. Reports backtest vs Buy&Hold + 70/30 out-of-sample.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

FEE=0.006; SLIP=0.0005
CAP=1000.0
COINS=["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","LINK/USD","ADA/USD"]
VWAP_WIN=20    # rolling VWAP window (days)


def rolling_vwap(df, win=VWAP_WIN):
    tp=(df["high"]+df["low"]+df["close"])/3.0
    pv=(tp*df["volume"]).rolling(win).sum()
    v =df["volume"].rolling(win).sum().replace(0,np.nan)
    return pv/v


def signals(df):
    vwap=rolling_vwap(df); ema9=df["close"].ewm(span=9,adjust=False).mean(); c=df["close"]
    above=c>vwap; below=c<ema9
    cu=above & ~above.shift(1).fillna(False)
    cd=below & ~below.shift(1).fillna(False)
    return cu.values, cd.values


def run(df):
    c=df["close"].values; cu,cd=signals(df); cash=CAP; units=0.0; entry=0.0; eq=[]; tr=[]
    for i in range(len(df)):
        if units==0 and cu[i]:
            buy=c[i]*(1+SLIP); units=(cash*(1-FEE))/buy; entry=buy; cash=0
        elif units>0 and cd[i]:
            sell=c[i]*(1-SLIP); cash=units*sell*(1-FEE); tr.append((sell-entry)/entry); units=0
        eq.append(cash+units*c[i])
    if units>0: sell=c[-1]*(1-SLIP); eq[-1]=units*sell*(1-FEE); tr.append((sell-entry)/entry)
    return pd.Series(eq,index=df.index), tr


def stats(eq,tr):
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1
    end=eq.iloc[-1]; ret=(end/CAP-1)*100
    cagr=((end/CAP)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    wins=[t for t in tr if t>0]; wr=len(wins)/len(tr)*100 if tr else 0
    ls=abs(sum(t for t in tr if t<=0)); pf=sum(wins)/ls if ls>0 else (float('inf') if wins else 0)
    return ret,cagr,sh,dd,len(tr),wr,pf


def bh(df):
    eq=(CAP*(1-FEE))/df["close"].iloc[0]*df["close"]; return stats(eq,[])[0]


def main():
    print(f"VWAP({VWAP_WIN}d rolling) entry / EMA9 exit | DAILY | fees 0.6%/side+slip | long-only\n")
    print(f"| {'Coin':5s} | {'Strat':>8s} | {'CAGR':>6s} | {'Sharpe':>6s} | {'MaxDD':>6s} | {'Trades':>6s} | {'WR':>4s} | {'PF':>5s} | {'B&H':>7s} |")
    print("|"+"-"*7+"|"+"-"*10+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|"+"-"*6+"|"+"-"*7+"|"+"-"*9+"|")
    store={}; prof=0; beat=0; tot=0
    for sym in COINS:
        df=fetch_ohlcv(sym,"1d",limit=2500)
        if df.empty or len(df)<200: print(f"| {sym.split('/')[0]:5s} | no data |"); continue
        store[sym]=df; eq,tr=run(df); ret,cagr,sh,dd,n,wr,pf=stats(eq,tr); b=bh(df)
        tot+=1; prof+=ret>0; beat+=ret>b
        print(f"| {sym.split('/')[0]:5s} | {ret:+7.0f}% | {cagr:+5.0f}% | {sh:+6.2f} | {dd:5.0f}% | {n:6d} | {wr:3.0f}% | {pf:5.2f} | {b:+6.0f}% |")
    yrs=(store[COINS[0]].index[-1]-store[COINS[0]].index[0]).days/365.25 if store else 0
    print(f"\n  (~{yrs:.1f} years of daily data)")
    print("\nOUT-OF-SAMPLE (first 70% vs last 30%):")
    print(f"| {'Coin':5s} | {'IS Ret':>7s} | {'IS PF':>5s} | {'OOS Ret':>7s} | {'OOS PF':>6s} |")
    print("|"+"-"*7+"|"+"-"*9+"|"+"-"*7+"|"+"-"*9+"|"+"-"*8+"|")
    for sym,df in store.items():
        cut=int(len(df)*0.7); ins=df.iloc[:cut]; oos=df.iloc[cut:]
        ei,ti=run(ins); eo,to=run(oos)
        ri=(ei.iloc[-1]/CAP-1)*100; pfi=stats(ei,ti)[6]
        ro=(eo.iloc[-1]/CAP-1)*100; pfo=stats(eo,to)[6]
        print(f"| {sym.split('/')[0]:5s} | {ri:+6.0f}% | {pfi:5.2f} | {ro:+6.0f}% | {pfo:5.2f} |")
    print(f"\n  Profitable on {prof}/{tot} coins.  Beat Buy&Hold on {beat}/{tot}.")


if __name__=="__main__":
    main()
