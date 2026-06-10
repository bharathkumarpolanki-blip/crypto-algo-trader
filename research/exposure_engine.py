"""
PRODUCTION EXPOSURE ENGINE — Fragility Score is FROZEN & immutable.

Converts the frozen Fragility Score into {exposure 0-100%, risk budget, position
size} and benchmarks it against Buy&Hold BTC / 200d-SMA / Vol-Targeting on
walk-forward. Optimizes ONLY Calmar, MaxDD, exposure-efficiency — never returns.
No tuning anywhere (all parameters fixed).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

CAP=1000.0; FEE=0.006; SLIP=0.0005
TARGET_VOL=0.55            # annualized target (fixed, no tuning)
MAX_LEV=1.0               # long-only spot, cap at 100%
PANEL=["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD",
       "LTC/USD","BCH/USD","XLM/USD","ETC/USD"]
_c={}
def load(s):
    if s not in _c:
        try: d=fetch_ohlcv(s,"1d",limit=2500)
        except Exception: d=None
        _c[s]=d if (d is not None and not d.empty and len(d)>700) else None
    return _c[s]


def z(s,w=252): return (s-s.rolling(w,min_periods=60).mean())/s.rolling(w,min_periods=60).std()


def build():
    data={s:load(s) for s in PANEL}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    vol=pd.DataFrame({k:v["volume"].reindex(idx) for k,v in data.items()})
    return close, vol, idx


def frozen_fragility(close, vol):
    """IMMUTABLE — identical formula to the frozen audit. BTC-anchored market read."""
    ret=close.pct_change()
    v14=ret["BTC/USD"].rolling(14).std()
    vrank=v14.rolling(252,min_periods=60).rank(pct=True)
    amihud=(ret["BTC/USD"].abs()/(vol["BTC/USD"]*close["BTC/USD"]+1e-9)).rolling(14).mean()
    mom=close["BTC/USD"].pct_change(30)
    volcompress=1-vrank
    breadth=(close>close.rolling(50).mean()).astype(float).mean(axis=1)
    liquidity=-amihud.rolling(252,min_periods=60).rank(pct=True)
    trend=close["BTC/USD"]/close["BTC/USD"].rolling(200).mean()-1
    frag=( 0.30*z(mom).clip(-3,3) + 0.25*z(volcompress).clip(-3,3)
         + 0.20*z(1-breadth).clip(-3,3) + 0.15*z(-liquidity).clip(-3,3)
         + 0.10*z(trend).clip(-3,3) )
    return frag, v14


# ── Exposure engines (each returns exposure ∈ [0,1] at close[T]) ───────────────
def expo_buyhold(btc, v14, frag): return pd.Series(1.0, index=btc.index)
def expo_sma(btc, v14, frag):     return (btc>btc.rolling(200).mean()).astype(float)
def expo_voltarget(btc, v14, frag):
    ann=v14*np.sqrt(365)
    return (TARGET_VOL/ann).clip(0, MAX_LEV).fillna(0)
def expo_fragility(btc, v14, frag):
    # FROZEN mapping: exposure = 1 − rolling percentile of fragility (high frag → low expo)
    fr=frag.rolling(252,min_periods=60).rank(pct=True)
    return (1-fr).clip(0, MAX_LEV).fillna(0.5)


ENGINES={"Buy&Hold BTC":expo_buyhold, "200d SMA":expo_sma,
         "Vol-Target":expo_voltarget, "Fragility Engine":expo_fragility}


def backtest(btc, exposure):
    ret=btc.pct_change().fillna(0)
    p=exposure.shift(1).fillna(0)                      # T+1
    turn=p.diff().abs().fillna(p.abs())
    daily=p*ret - turn*(FEE+SLIP)
    return daily, p


def metrics(daily, expo):
    d=daily.dropna(); eq=(1+d).cumprod()
    yrs=len(d)/365.25
    cagr=(eq.iloc[-1]**(1/yrs)-1)*100 if eq.iloc[-1]>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    sh=d.mean()/d.std()*np.sqrt(365) if d.std()>0 else 0
    calmar=cagr/abs(dd) if dd!=0 else 0
    avg_expo=expo.reindex(d.index).mean()*100
    eff=cagr/(avg_expo/100) if avg_expo>0 else 0     # return per unit of market exposure
    return dict(cagr=cagr,dd=dd,sharpe=sh,calmar=calmar,avg_expo=avg_expo,eff=eff)


def main():
    close,vol,idx=build()
    btc=close["BTC/USD"]; frag,v14=frozen_fragility(close,vol)
    print(f"PRODUCTION EXPOSURE ENGINE | BTC {idx[0].date()}→{idx[-1].date()} | "
          f"target_vol={TARGET_VOL}, no tuning\n")

    # ── Full-sample comparison ────────────────────────────────────────────────
    print("### FULL SAMPLE (optimize: Calmar, MaxDD, Exposure-efficiency — NOT returns)\n")
    print(f"| {'Engine':18s} | {'CAGR':>6s} | {'MaxDD':>6s} | {'Calmar':>6s} | {'Sharpe':>6s} | {'AvgExpo':>7s} | {'Effic.':>6s} |")
    print("|"+"-"*20+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|"+"-"*9+"|"+"-"*8+"|")
    full={}
    for nm,fn in ENGINES.items():
        e=fn(btc,v14,frag); d,p=backtest(btc,e); m=metrics(d,e); full[nm]=m
        print(f"| {nm:18s} | {m['cagr']:+5.0f}% | {m['dd']:5.0f}% | {m['calmar']:6.2f} | "
              f"{m['sharpe']:+5.2f} | {m['avg_expo']:6.0f}% | {m['eff']:6.2f} |")

    # ── Walk-forward by year (Calmar & MaxDD) ─────────────────────────────────
    print("\n### WALK-FORWARD by year — Calmar (and MaxDD)\n")
    yrs=list(range(2020,2027))
    hdr="| Engine             | "+" | ".join(f"{y}" for y in yrs)+" |"
    print(hdr); print("|"+"-"*20+"|"+("-"*7+"|")*len(yrs))
    for nm,fn in ENGINES.items():
        e=fn(btc,v14,frag); d,p=backtest(btc,e); row=[]
        for y in yrs:
            m=(d.index>=f"{y}-01-01")&(d.index<f"{y+1}-01-01")
            if m.sum()<30: row.append("  -  "); continue
            dy=d[m]; eq=(1+dy).cumprod(); ddy=((eq-eq.cummax())/eq.cummax()).min()
            cg=eq.iloc[-1]**(365.25/len(dy))-1
            cal=cg/abs(ddy) if ddy!=0 else 0
            row.append(f"{cal:+5.1f}")
        print(f"| {nm:18s} | "+" | ".join(f"{c:>5s}" for c in row)+" |")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n### DEPLOYABILITY VERDICT\n")
    bh=full["Buy&Hold BTC"]; vt=full["Vol-Target"]; fe=full["Fragility Engine"]
    print(f"  Calmar:   Buy&Hold {bh['calmar']:.2f} | Vol-Target {vt['calmar']:.2f} | Fragility {fe['calmar']:.2f}")
    print(f"  MaxDD:    Buy&Hold {bh['dd']:.0f}% | Vol-Target {vt['dd']:.0f}% | Fragility {fe['dd']:.0f}%")
    print(f"  Efficiency: Buy&Hold {bh['eff']:.2f} | Vol-Target {vt['eff']:.2f} | Fragility {fe['eff']:.2f}")
    beats_bh = fe['calmar']>bh['calmar'] and fe['dd']>bh['dd']
    beats_vt = fe['calmar']>vt['calmar']+0.05
    print(f"\n  Fragility engine beats Buy&Hold (Calmar+DD)? {'YES' if beats_bh else 'NO'}")
    print(f"  Fragility engine beats Vol-Target (the simpler tool)? {'YES' if beats_vt else 'NO'}")


if __name__=="__main__":
    main()
