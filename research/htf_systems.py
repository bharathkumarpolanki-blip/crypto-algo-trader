"""
PHASE 7 — HIGHER-TIMEFRAME SYSTEMS on DAILY / WEEKLY candles, vs benchmarks.

Separate systems (not the 1h engine):
  1. Weekly trend following  (hold while weekly close > 30-week SMA)
  2. Daily mean reversion     (buy RSI<35 oversold, exit RSI>55)
  3. Daily breakout           (Donchian 20-day high entry / 10-day low exit)
  4. Weekly momentum          (hold while 12-week return > 0)
Benchmarks: BTC Buy&Hold, ETH Buy&Hold, Simple 200-day SMA.

Long-only (spot), fees on (0.6%/side). Daily history (~5y on Coinbase).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
import config
from exchange.market_data import fetch_ohlcv

FEE = config.FEE_RATE_PCT/100.0
CAP = 1000.0
SYMS = ["BTC/USD","ETH/USD"]


def metrics(eq, trades, idx):
    eq = pd.Series(eq, index=idx)
    yrs = (idx[-1]-idx[0]).days/365.25
    end = eq.iloc[-1]
    ret = (end/CAP-1)*100
    cagr = ((end/CAP)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd = ((eq-eq.cummax())/eq.cummax()).min()*100
    rets = eq.pct_change().dropna()
    sharpe = rets.mean()/rets.std()*np.sqrt(365) if rets.std()>0 else 0
    wins=[t for t in trades if t>0]; wr=len(wins)/len(trades)*100 if trades else 0
    pf = sum(wins)/abs(sum(t for t in trades if t<=0)) if any(t<=0 for t in trades) else float("inf")
    return ret,cagr,dd,sharpe,len(trades),wr,pf


def run_signal(df, inpos):
    c=df["close"].values; cash=CAP; units=0.0; entry=0.0; eq=[]; trades=[]
    for i in range(len(df)):
        if inpos[i] and units==0:
            units=(cash*(1-FEE))/c[i]; entry=c[i]; cash=0
        elif not inpos[i] and units>0:
            cash=units*c[i]*(1-FEE); trades.append((c[i]-entry)/entry); units=0
        eq.append(cash+units*c[i])
    if units>0: eq[-1]=units*c[-1]*(1-FEE); trades.append((c[-1]-entry)/entry)
    return eq, trades


def weekly_trend(df):
    w = df["close"].resample("1W").last().dropna()
    sma = w.rolling(30).mean()
    sig = (w > sma).reindex(df.index, method="ffill").fillna(False).values
    return sig

def daily_meanrev(df):
    d=df["close"]; delta=d.diff()
    up=delta.clip(lower=0).rolling(14).mean(); dn=(-delta.clip(upper=0)).rolling(14).mean()
    rsi=100-100/(1+up/dn.replace(0,np.nan))
    inpos=np.zeros(len(df),bool); hold=False
    for i in range(len(df)):
        if not hold and rsi.iloc[i]<35: hold=True
        elif hold and rsi.iloc[i]>55: hold=False
        inpos[i]=hold
    return inpos

def daily_breakout(df):
    hi=df["high"].rolling(20).max().shift(1).values
    lo=df["low"].rolling(10).min().shift(1).values
    c=df["close"].values; inpos=np.zeros(len(df),bool); hold=False
    for i in range(len(df)):
        if not hold and not np.isnan(hi[i]) and c[i]>hi[i]: hold=True
        elif hold and not np.isnan(lo[i]) and c[i]<lo[i]: hold=False
        inpos[i]=hold
    return inpos

def weekly_momentum(df):
    w=df["close"].resample("1W").last().dropna()
    mom=w.pct_change(12)
    sig=(mom>0).reindex(df.index, method="ffill").fillna(False).values
    return sig

def sma200(df):
    return (df["close"] > df["close"].rolling(200).mean()).values


def main():
    print(f"PHASE 7 — Daily/Weekly systems vs benchmarks (fees {FEE*100}%/side)\n")
    for sym in SYMS:
        df=fetch_ohlcv(sym,"1d",limit=2500)
        if df.empty or len(df)<300: print(f"{sym}: no data"); continue
        yrs=(df.index[-1]-df.index[0]).days/365.25
        print(f"{'='*78}\n{sym}  {df.index[0].date()}→{df.index[-1].date()} ({yrs:.1f}y)\n{'='*78}")
        # benchmark buy&hold
        bh_u=(CAP*(1-FEE))/df['close'].iloc[0]; bh=bh_u*df['close']
        bret=(bh.iloc[-1]/CAP-1)*100; bdd=((bh-bh.cummax())/bh.cummax()).min()*100
        br=bh.pct_change().dropna(); bsh=br.mean()/br.std()*np.sqrt(365) if br.std()>0 else 0
        print(f"| {'System':22s} | {'Ret':>8s} | {'CAGR':>7s} | {'Sharpe':>6s} | {'MaxDD':>7s} | {'Trades':>6s} | {'WR':>4s} | {'PF':>4s} |")
        print("|"+"-"*24+"|"+"-"*10+"|"+"-"*9+"|"+"-"*8+"|"+"-"*9+"|"+"-"*8+"|"+"-"*6+"|"+"-"*6+"|")
        print(f"| {'BUY & HOLD':22s} | {bret:+7.0f}% | {((bh.iloc[-1]/CAP)**(1/yrs)-1)*100:+6.1f}% | {bsh:+6.2f} | {bdd:6.0f}% | {'1':>6s} | {'—':>4s} | {'—':>4s} |")
        systems = {
            "200-day SMA":       sma200(df),
            "Weekly TrendFollow": weekly_trend(df),
            "Daily MeanReversion":daily_meanrev(df),
            "Daily Breakout":     daily_breakout(df),
            "Weekly Momentum":    weekly_momentum(df),
        }
        for nm, sig in systems.items():
            eq,trades=run_signal(df, sig)
            ret,cagr,dd,sh,nt,wr,pf=metrics(eq,trades,df.index)
            print(f"| {nm:22s} | {ret:+7.0f}% | {cagr:+6.1f}% | {sh:+6.2f} | {dd:6.0f}% | {nt:6d} | {wr:3.0f}% | {pf:4.2f} |")
        print()


if __name__=="__main__":
    main()
