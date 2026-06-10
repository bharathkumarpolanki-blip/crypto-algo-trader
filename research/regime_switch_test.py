"""
STANDALONE — tests the user's idea: switch strategy by market regime.

Idea: when market flips Bear->Bull, trade ACTIVELY; when Bull->Bear, go DEFENSIVE
(cash / slow trend). Compares, head-to-head over 6+ years of DAILY data (multiple
bull/bear cycles — 1h data only covers ~162d = one regime, so daily is the only
honest way to test a regime SWITCH):

  1. BUY & HOLD                  — baseline
  2. SMA200 slow trend (sma_bot) — the endorsed defensive system
  3. ALWAYS-ACTIVE               — active engine all the time (no switch)
  4. REGIME-SWITCH (the idea)    — ACTIVE in bull, CASH in bear

"Active engine" here = a fast EMA9/EMA21 momentum trader (a fair, generous proxy
for bot.py-style active trading: trades the short-term swings). Regime = BTC vs a
rising 200-day SMA. Fees 0.6%/side + slippage. Long-only spot.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

FEE=0.006; SLIP=0.0005; CAP=1000.0
COINS=["BTC/USD","ETH/USD"]


def regime_bull(btc):
    """Daily bull flag: BTC above a RISING 200-day SMA."""
    sma=btc.rolling(200).mean()
    return (btc>sma) & (sma.diff(20)>0)


def active_signal(close):
    """Active engine proxy: long while EMA9>EMA21 (fast momentum). In/out daily."""
    e9=close.ewm(span=9,adjust=False).mean(); e21=close.ewm(span=21,adjust=False).mean()
    return (e9>e21).values


def sma200_signal(close):
    """Slow trend: long while close>200d SMA."""
    return (close>close.rolling(200).mean()).fillna(False).values


def sim(close, inpos):
    """Long-only daily spot with fees+slip. inpos: bool array (hold or cash)."""
    c=close.values; cash=CAP; units=0.0; entry=0.0; eq=[]; trades=[]
    for i in range(len(c)):
        if inpos[i] and units==0:
            buy=c[i]*(1+SLIP); units=(cash*(1-FEE))/buy; entry=buy; cash=0
        elif not inpos[i] and units>0:
            sell=c[i]*(1-SLIP); cash=units*sell*(1-FEE); trades.append((sell-entry)/entry); units=0
        eq.append(cash+units*c[i])
    if units>0:
        sell=c[-1]*(1-SLIP); eq[-1]=units*sell*(1-FEE); trades.append((sell-entry)/entry)
    return pd.Series(eq,index=close.index), trades


def stats(eq,trades):
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1
    end=eq.iloc[-1]; ret=(end/CAP-1)*100
    cagr=((end/CAP)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    wins=[t for t in trades if t>0]; wr=len(wins)/len(trades)*100 if trades else 0
    ls=abs(sum(t for t in trades if t<=0)); pf=sum(wins)/ls if ls>0 else (float('inf') if wins else 0)
    return ret,cagr,sh,dd,len(trades),wr,pf


def run_coin(sym, btc_close):
    df=fetch_ohlcv(sym,"1d",limit=2500)
    if df.empty or len(df)<260: return None
    close=df["close"]
    bull=regime_bull(btc_close.reindex(close.index, method="ffill")).reindex(close.index).fillna(False).values

    # 1. Buy & Hold
    bh=(CAP*(1-FEE))/close.iloc[0]*close
    # 2. SMA200 slow
    sma_eq,sma_tr=sim(close, sma200_signal(close))
    # 3. Always-active
    act=active_signal(close)
    act_eq,act_tr=sim(close, act)
    # 4. Regime-switch: active ONLY in bull, cash in bear  (the idea)
    switch = act & bull
    sw_eq,sw_tr=sim(close, switch)

    return {
        "sym":sym,
        "BuyHold": stats(bh,[]),
        "SMA200":  stats(sma_eq,sma_tr),
        "Active":  stats(act_eq,act_tr),
        "Switch":  stats(sw_eq,sw_tr),
        "yrs": (close.index[-1]-close.index[0]).days/365.25,
    }


def fmt(s): return f"{s[0]:+7.0f}% | CAGR {s[1]:+5.0f}% | Sh {s[2]:+5.2f} | DD {s[3]:5.0f}% | {s[4]:3d}tr | PF {s[6]:4.2f}"


def main():
    print("REGIME-SWITCH TEST — daily, fees+slip, long-only\n"
          "Idea: trade ACTIVE in bull, CASH in bear. vs Hold / SMA200 / Always-Active\n")
    btc=fetch_ohlcv("BTC/USD","1d",limit=2500)["close"]
    for sym in COINS:
        r=run_coin(sym, btc)
        if not r: print(f"{sym}: no data"); continue
        print(f"{'='*78}\n{sym}  ({r['yrs']:.1f} years)\n{'='*78}")
        print(f"  Buy & Hold      : {fmt(r['BuyHold'])}")
        print(f"  SMA200 (slow)   : {fmt(r['SMA200'])}")
        print(f"  Always-Active   : {fmt(r['Active'])}")
        print(f"  REGIME-SWITCH   : {fmt(r['Switch'])}   <-- the idea")
        # verdict
        sw=r['Switch'][0]; best=max(r['BuyHold'][0],r['SMA200'][0])
        print(f"  -> Switch vs best-of(Hold,SMA200): {sw:+.0f}% vs {best:+.0f}%  "
              f"({'BEATS' if sw>best else 'loses'})\n")


if __name__=="__main__":
    main()
