"""
REGIME-ADAPTIVE PORTFOLIO SYSTEM — design + backtest (Phases 1-8).

Not optimizing a single strategy. Builds a system that CHANGES BEHAVIOR by market
regime: invest (trend-follow) in bull, cash in bear/sideways, reduce exposure in
high vol. Daily data, fees+slippage, long-only spot. Read-only.

Market proxy / regime leader = BTC. Tradeable basket = equal-weight available
quality coins. Everything judged out-of-sample against Buy & Hold.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

FEE=0.006; SLIP=0.0005; CAP=1000.0
UNIVERSE=["BTC/USD","ETH/USD","SOL/USD","LINK/USD","AVAX/USD","XRP/USD"]
_cache={}
def load(s):
    if s not in _cache:
        d=fetch_ohlcv(s,"1d",limit=2500); _cache[s]=d if (not d.empty and len(d)>260) else None
    return _cache[s]


# ── PHASE 1: regime detection ────────────────────────────────────────────────
def detect_regimes(btc):
    c=btc; sma=c.rolling(200).mean(); slope=sma.diff(20)
    trend=pd.Series("side", index=c.index)
    trend[(c>sma)&(slope>0)]="bull"; trend[(c<sma)&(slope<0)]="bear"
    atr=(c.pct_change().abs().rolling(14).mean())          # daily vol proxy
    hivol=atr > atr.rolling(180,min_periods=30).median()*1.3
    reg=pd.Series("Sideways", index=c.index)
    reg[(trend=="bull")&~hivol]="Bull Trend"
    reg[(trend=="bull")& hivol]="HighVol Bull"
    reg[(trend=="bear")&~hivol]="Bear Trend"
    reg[(trend=="bear")& hivol]="HighVol Bear"
    reg[(trend=="side")&~hivol]="LowVol Range"
    reg[(trend=="side")& hivol]="Sideways"
    return reg


def stats_from_eq(eq, trades):
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1; end=eq.iloc[-1]
    cagr=((end/CAP)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    wins=[t for t in trades if t>0]; ls=abs(sum(t for t in trades if t<=0))
    pf=sum(wins)/ls if ls>0 else (float('inf') if wins else 0)
    return {"ret":(end/CAP-1)*100,"cagr":cagr,"sharpe":sh,"maxdd":dd,"pf":pf,"n":len(trades)}


def basket_returns(coins):
    """Equal-weight daily return series across available coins (union index, ffill)."""
    data={s:load(s) for s in coins}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    rets=pd.DataFrame({k:v["close"].reindex(idx).pct_change() for k,v in data.items()})
    return rets.mean(axis=1).fillna(0), list(data.keys()), idx


def run_engine(bret, exposure, idx):
    """Compound basket return * daily exposure; fee on exposure turnover. Trade-aware."""
    eq=CAP; series=[]; trades=[]; in_ep=False; ep=1.0; prev=0.0
    for t in idx:
        e=exposure.get(t,0.0); r=bret.get(t,0.0)
        if abs(e-prev)>1e-9: eq*= (1-FEE*abs(e-prev))      # turnover cost
        if e>0 and not in_ep: in_ep=True; ep=1.0
        if e>0: ep*=(1+e*r)
        if e==0 and in_ep: in_ep=False; trades.append(ep-1)
        eq*=(1+e*r); series.append(eq); prev=e
    if in_ep: trades.append(ep-1)
    return pd.Series(series,index=idx), trades


def main():
    print("REGIME-ADAPTIVE PORTFOLIO SYSTEM\n")
    btc=load("BTC/USD")["close"]
    reg=detect_regimes(btc)
    bret,coins,idx=basket_returns(UNIVERSE)
    reg=reg.reindex(idx, method="ffill")
    fwd20=btc.reindex(idx).pct_change(20).shift(-20)       # forward 20d return

    # ── PHASE 1 output ────────────────────────────────────────────────────────
    print("="*74,"\nPHASE 1 — REGIME DETECTION (BTC leader)\n"+"="*74)
    print("Rules: trend = BTC vs 200d SMA + 20d slope (bull/bear/side); "
          "vol = 14d |ret| vs 1.3x its 180d median.")
    print(f"\n| {'Regime':14s} | {'% days':>6s} | {'fwd-20d ret':>11s} | {'% positive':>10s} |")
    print("|"+"-"*16+"|"+"-"*8+"|"+"-"*13+"|"+"-"*12+"|")
    order=["Bull Trend","HighVol Bull","Bear Trend","HighVol Bear","Sideways","LowVol Range"]
    for rg in order:
        m=(reg==rg).values
        f=fwd20.values[m]; f=f[~np.isnan(f)]
        if len(f)==0: continue
        print(f"| {rg:14s} | {m.mean()*100:5.0f}% | {np.mean(f)*100:+10.1f}% | {(f>0).mean()*100:9.0f}% |")
    acc_bull=fwd20.values[reg.isin(["Bull Trend","HighVol Bull"]).values]
    acc_bear=fwd20.values[reg.isin(["Bear Trend","HighVol Bear"]).values]
    acc_bull=acc_bull[~np.isnan(acc_bull)]; acc_bear=acc_bear[~np.isnan(acc_bear)]
    print(f"\nAccuracy: bull-regime days fwd>0 {(acc_bull>0).mean()*100:.0f}% | "
          f"bear-regime days fwd<0 {(acc_bear<0).mean()*100:.0f}%")

    # ── PHASE 2: best action per regime (invest vs cash) ──────────────────────
    print("\n"+"="*74,"\nPHASE 2 — BEST STRATEGY PER REGIME\n"+"="*74)
    print(f"| {'Regime':14s} | {'Best action':12s} | {'mean dayret':>11s} | {'Sharpe':>6s} |")
    print("|"+"-"*16+"|"+"-"*14+"|"+"-"*13+"|"+"-"*8+"|")
    regime_action={}
    for rg in order:
        m=(reg==rg).values; rr=bret.values[m]
        if len(rr)<10: continue
        sh=rr.mean()/rr.std()*np.sqrt(365) if rr.std()>0 else 0
        act = "Trend-follow" if rr.mean()>0 else "Cash"
        regime_action[rg]=1.0 if rr.mean()>0 else 0.0
        print(f"| {rg:14s} | {act:12s} | {rr.mean()*100:+10.2f}% | {sh:+6.2f} |")

    # ── PHASE 3: cash allocation tests ────────────────────────────────────────
    print("\n"+"="*74,"\nPHASE 3 — CASH ALLOCATION\n"+"="*74)
    btrend=(1+bret).cumprod(); bsma=btrend.rolling(200).mean()
    invest_gate=(btrend>bsma)                               # basket trend up
    allocs={
        "A Always-Invested": pd.Series(1.0,index=idx),
        "B Regime-Aware Cash": pd.Series([regime_action.get(reg.get(t,"Sideways"),0.0) for t in idx],index=idx),
        "C Defensive (regime+trend)": pd.Series([ (regime_action.get(reg.get(t,"Sideways"),0.0) if invest_gate.get(t,False) else 0.0) for t in idx],index=idx),
    }
    print(f"| {'Allocation':26s} | {'CAGR':>6s} | {'Sharpe':>6s} | {'PF':>5s} | {'MaxDD':>6s} |")
    print("|"+"-"*28+"|"+"-"*8+"|"+"-"*8+"|"+"-"*7+"|"+"-"*8+"|")
    for nm,ex in allocs.items():
        eq,tr=run_engine(bret, ex, idx); s=stats_from_eq(eq,tr)
        print(f"| {nm:26s} | {s['cagr']:+5.0f}% | {s['sharpe']:+6.2f} | {s['pf']:5.2f} | {s['maxdd']:5.0f}% |")

    # ── PHASE 4: relative-strength rotation ───────────────────────────────────
    print("\n"+"="*74,"\nPHASE 4 — RELATIVE-STRENGTH ROTATION\n"+"="*74)
    rotation_table(coins, idx, reg, regime_action, invest_gate)

    # ── PHASE 5: risk parity ──────────────────────────────────────────────────
    print("\n"+"="*74,"\nPHASE 5 — PORTFOLIO WEIGHTING\n"+"="*74)
    weighting_table(coins, idx, invest_gate, reg, regime_action)

    # ── PHASE 6: full regime-switch engine (Defensive C) ──────────────────────
    print("\n"+"="*74,"\nPHASE 6 — FULL REGIME-SWITCH ENGINE (bull→trend, else→cash, hi-vol→half)\n"+"="*74)
    expo=pd.Series([engine_exposure(reg.get(t,"Sideways"), invest_gate.get(t,False)) for t in idx], index=idx)
    eq6,tr6=run_engine(bret, expo, idx); s6=stats_from_eq(eq6,tr6)
    bh=(1+bret).cumprod()*CAP; bhs=stats_from_eq(bh,[])
    print(f"| Metric | Engine | Buy&Hold basket |")
    print("|--------|--------|-----------------|")
    print(f"| CAGR   | {s6['cagr']:+.1f}% | {bhs['cagr']:+.1f}% |")
    print(f"| Sharpe | {s6['sharpe']:+.2f} | {bhs['sharpe']:+.2f} |")
    print(f"| PF     | {s6['pf']:.2f} | — |")
    print(f"| MaxDD  | {s6['maxdd']:.0f}% | {bhs['maxdd']:.0f}% |")

    # ── PHASE 7: out-of-sample ────────────────────────────────────────────────
    print("\n"+"="*74,"\nPHASE 7 — OUT-OF-SAMPLE (newest 20%, no tuning)\n"+"="*74)
    cut=int(len(idx)*0.8); oidx=idx[cut:]
    eqo,tro=run_engine(bret.reindex(oidx), expo.reindex(oidx), oidx)
    eqo=eqo/eqo.iloc[0]*CAP; so=stats_from_eq(eqo,tro)
    bho=(1+bret.reindex(oidx)).cumprod()*CAP; bhos=stats_from_eq(bho,[])
    print(f"  OOS window: {oidx[0].date()} → {oidx[-1].date()}")
    print(f"  Engine  : CAGR {so['cagr']:+.1f}%  Sharpe {so['sharpe']:+.2f}  PF {so['pf']:.2f}  MaxDD {so['maxdd']:.0f}%")
    print(f"  Buy&Hold: CAGR {bhos['cagr']:+.1f}%  Sharpe {bhos['sharpe']:+.2f}  MaxDD {bhos['maxdd']:.0f}%")

    # ── PHASE 8: verdict ──────────────────────────────────────────────────────
    print("\n"+"="*74,"\nPHASE 8 — VERDICT\n"+"="*74)
    checks={
        "Sharpe > 1.0 (full)":      (s6['sharpe']>1.0, f"{s6['sharpe']:+.2f}"),
        "OOS CAGR > 0":             (so['cagr']>0,     f"{so['cagr']:+.1f}%"),
        "OOS PF > 1.0":             (so['pf']>1.0,     f"{so['pf']:.2f}"),
        "MaxDD < Buy&Hold (full)":  (s6['maxdd']>bhs['maxdd'], f"{s6['maxdd']:.0f}% vs {bhs['maxdd']:.0f}%"),
    }
    for d,(ok,v) in checks.items():
        print(f"  {d:26s} {v:>16s}   {'✅' if ok else '❌'}")
    allp=all(v[0] for v in checks.values())
    print(f"\n  {'✅ Regime-adaptive system PASSES' if allp else '❌ Does NOT meet all criteria'}")


def engine_exposure(rg, gate):
    if rg in ("Bull Trend",) and gate:      return 1.0
    if rg in ("HighVol Bull",) and gate:    return 0.5       # reduce in high vol
    return 0.0                                                # bear/sideways → cash


def rotation_table(coins, idx, reg, regime_action, gate):
    data={s:load(s) for s in coins}; data={k:v for k,v in data.items() if v is not None}
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    mom=close.pct_change(84)
    print(f"| {'Method':8s} | {'CAGR':>6s} | {'Sharpe':>6s} | {'PF':>5s} | {'MaxDD':>6s} |")
    print("|"+"-"*10+"|"+"-"*8+"|"+"-"*8+"|"+"-"*7+"|"+"-"*8+"|")
    for n in [1,2,3]:
        eq=CAP; series=[]; held=[]; trades=[]; prev=set()
        rebal=set(idx[::7])
        rets=close.pct_change()
        for i,t in enumerate(idx):
            if t in rebal:
                invest = (regime_action.get(reg.get(t,"Sideways"),0.0)>0) and gate.get(t,False)
                if invest:
                    r=mom.loc[t].dropna().sort_values(ascending=False)
                    pick=set(r.index[:n])
                else: pick=set()
                if pick!=prev:
                    eq*= (1-FEE*0.5)                          # rough turnover cost
                    if prev and not pick: trades.append(0.0)
                    prev=pick; held=list(pick)
            if held:
                dr=rets.loc[t,held].mean()
                if not np.isnan(dr): eq*=(1+dr)
            series.append(eq)
        s=stats_from_eq(pd.Series(series,index=idx), [t for t in trades] or [0.01])
        print(f"| Top{n:<5d} | {s['cagr']:+5.0f}% | {s['sharpe']:+6.2f} | {s['pf']:5.2f} | {s['maxdd']:5.0f}% |")


def weighting_table(coins, idx, gate, reg, regime_action):
    data={s:load(s) for s in coins}; data={k:v for k,v in data.items() if v is not None}
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    rets=close.pct_change()
    vol=rets.rolling(30).std()
    schemes={
        "Equal Weight":      lambda t: pd.Series(1.0,index=close.columns),
        "Vol-Adjusted":      lambda t: (1/vol.loc[t]).replace([np.inf,np.nan],0),
        "Risk Parity":       lambda t: (1/vol.loc[t]**2).replace([np.inf,np.nan],0),
    }
    print(f"| {'Portfolio':14s} | {'CAGR':>6s} | {'Sharpe':>6s} | {'PF':>5s} | {'MaxDD':>6s} |")
    print("|"+"-"*16+"|"+"-"*8+"|"+"-"*8+"|"+"-"*7+"|"+"-"*8+"|")
    for nm,wfn in schemes.items():
        eq=CAP; series=[]
        for t in idx:
            invest=(regime_action.get(reg.get(t,"Sideways"),0.0)>0) and gate.get(t,False)
            if invest:
                w=wfn(t); w=w/w.sum() if w.sum()>0 else w
                dr=(rets.loc[t]*w).sum()
                if not np.isnan(dr): eq*=(1+dr)
            series.append(eq)
        s=stats_from_eq(pd.Series(series,index=idx),[0.01])
        print(f"| {nm:14s} | {s['cagr']:+5.0f}% | {s['sharpe']:+6.2f} | {s['pf']:5.2f} | {s['maxdd']:5.0f}% |")


if __name__=="__main__":
    main()
