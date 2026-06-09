"""
Deployability gate for the DAILY EMA9/21 trend-follower (the lead surfaced by
regime_switch_test). Same criteria as validate.py — no optimization, just judge.

Strategy: long while EMA9 > EMA21 on daily candles, cash otherwise. Long-only
spot, fees 0.6%/side + 0.05% slippage. Tested per-coin AND as an equal-weight
BTC+ETH+SOL portfolio.

Pass criteria (ALL must hold):
  OOS PF > 1.20 | OOS CAGR > 0 | Full Sharpe > 1.00 | profitable @2x fees |
  >=60% walk-forward years positive
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

FEE=0.006; SLIP=0.0005; CAP=1000.0
COINS=["BTC/USD","ETH/USD","SOL/USD"]
_cache={}


def load(sym):
    if sym not in _cache:
        d=fetch_ohlcv(sym,"1d",limit=2500); _cache[sym]=d if (not d.empty and len(d)>260) else None
    return _cache[sym]


def ema_inpos(close):
    e9=close.ewm(span=9,adjust=False).mean(); e21=close.ewm(span=21,adjust=False).mean()
    return (e9>e21).values


def sim(close, fee=FEE):
    c=close.values; ip=ema_inpos(close); cash=CAP; u=0.0; entry=0.0; eq=[]; tr=[]
    for i in range(len(c)):
        if ip[i] and u==0: buy=c[i]*(1+SLIP); u=(cash*(1-fee))/buy; entry=buy; cash=0
        elif not ip[i] and u>0: s=c[i]*(1-SLIP); cash=u*s*(1-fee); tr.append((s-entry)/entry); u=0
        eq.append(cash+u*c[i])
    if u>0: s=c[-1]*(1-SLIP); eq[-1]=u*s*(1-fee); tr.append((s-entry)/entry)
    return pd.Series(eq,index=close.index), tr


def stats(eq,tr):
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1; end=eq.iloc[-1]
    cagr=((end/CAP)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    wins=[t for t in tr if t>0]; ls=abs(sum(t for t in tr if t<=0))
    pf=sum(wins)/ls if ls>0 else (float('inf') if wins else 0)
    return {"ret":(end/CAP-1)*100,"cagr":cagr,"sharpe":sh,"maxdd":dd,"pf":pf,"n":len(tr)}


def gate(close, label):
    full=stats(*sim(close))
    cut=int(len(close)*0.8)
    oos=stats(*sim(close.iloc[cut:]))
    c2=stats(*sim(close, fee=FEE*2))
    # walk-forward by year
    wf=[]
    for y in sorted(set(close.index.year))[1:]:
        sub=close[(close.index>=f"{y}-01-01")&(close.index<f"{y+1}-01-01")]
        if len(sub)<60: continue
        wf.append((y, stats(*sim(sub))["ret"]))
    posf=sum(1 for _,rr in wf if rr>0)/len(wf) if wf else 0
    checks={
        "OOS PF > 1.20":      (oos["pf"]>1.20,   f"{oos['pf']:.2f}"),
        "OOS CAGR > 0":       (oos["cagr"]>0,     f"{oos['cagr']:+.1f}%"),
        "Sharpe > 1.00":      (full["sharpe"]>1.0,f"{full['sharpe']:+.2f}"),
        "Profitable @2x fees": (c2["cagr"]>0,     f"{c2['cagr']:+.1f}%"),
        "≥60% WF years +":    (posf>=0.60,        f"{posf*100:.0f}%"),
    }
    allp=all(v[0] for v in checks.values())
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    print(f"  Full ({full['n']}tr): ret {full['ret']:+.0f}%  CAGR {full['cagr']:+.0f}%  "
          f"Sharpe {full['sharpe']:+.2f}  MaxDD {full['maxdd']:.0f}%  PF {full['pf']:.2f}")
    print(f"  OOS: CAGR {oos['cagr']:+.1f}%  Sharpe {oos['sharpe']:+.2f}  PF {oos['pf']:.2f}")
    print(f"  WF years: " + "  ".join(f"{y}:{rr:+.0f}%" for y,rr in wf))
    print("  " + "-"*52)
    for desc,(ok,val) in checks.items():
        print(f"  {desc:22s} {val:>9s}   {'✅' if ok else '❌'}")
    print("  " + "-"*52)
    print(f"  {'✅ PASS — paper-eligible' if allp else '❌ FAIL — not deployable'}")
    return allp


def portfolio():
    data={s:load(s) for s in COINS}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    eqs=[]
    for s,df in data.items():
        e,_=sim(df["close"]); eqs.append(e.reindex(idx).ffill())
    port=sum(eqs)/len(eqs)
    # synthesize a pseudo "trades" set via daily returns for PF
    r=port.pct_change().dropna()
    tr=list(r[r!=0].values)
    return port, tr


def main():
    print("DAILY EMA9/21 TREND-FOLLOWER — DEPLOYABILITY GATE (no optimization)")
    results={}
    for s in COINS:
        df=load(s)
        if df is None: print(f"\n{s}: no data"); continue
        results[s]=gate(df["close"], f"{s}  ({(df.index[-1]-df.index[0]).days/365.25:.1f}y)")
    # equal-weight portfolio
    port,tr=portfolio()
    st=stats(port,tr); cut=int(len(port)*0.8); oos=stats(port.iloc[cut:], list(port.iloc[cut:].pct_change().dropna().values))
    print(f"\n{'='*70}\n  EQUAL-WEIGHT PORTFOLIO (BTC+ETH+SOL)\n{'='*70}")
    print(f"  Full: ret {st['ret']:+.0f}%  CAGR {st['cagr']:+.0f}%  Sharpe {st['sharpe']:+.2f}  MaxDD {st['maxdd']:.0f}%")
    print(f"  OOS : CAGR {oos['cagr']:+.1f}%  Sharpe {oos['sharpe']:+.2f}")
    print(f"\n{'#'*70}\n  SUMMARY: passed {sum(results.values())}/{len(results)} coins on the full gate\n{'#'*70}")


if __name__=="__main__":
    main()
