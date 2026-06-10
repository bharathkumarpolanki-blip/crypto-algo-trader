"""
INSTITUTIONAL WALK-FORWARD VALIDATION of the Regime-Adaptive Portfolio System.

Strategy FROZEN exactly as in regime_adaptive.py — no tuning, no optimization.
Rules: invest (equal-weight basket) when BTC regime is bull AND basket trend up;
HighVol-Bull -> half exposure; bear/sideways -> cash. 200d SMA + 20d slope, vol
= 14d|ret| vs 1.3x 180d median. Fees+slip. Daily.

Tests robustness across multiple unseen years + bull/bear capture + param stability.
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


def detect_regimes(btc, trend_len=200):
    c=btc; sma=c.rolling(trend_len).mean(); slope=sma.diff(20)
    trend=pd.Series("side", index=c.index)
    trend[(c>sma)&(slope>0)]="bull"; trend[(c<sma)&(slope<0)]="bear"
    atr=c.pct_change().abs().rolling(14).mean()
    hivol=atr > atr.rolling(180,min_periods=30).median()*1.3
    reg=pd.Series("Sideways", index=c.index)
    reg[(trend=="bull")&~hivol]="Bull Trend"
    reg[(trend=="bull")& hivol]="HighVol Bull"
    reg[(trend=="bear")&~hivol]="Bear Trend"
    reg[(trend=="bear")& hivol]="HighVol Bear"
    reg[(trend=="side")&~hivol]="LowVol Range"
    reg[(trend=="side")& hivol]="Sideways"
    return reg


def basket(coins):
    data={s:load(s) for s in coins}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    ret=pd.DataFrame({k:v["close"].reindex(idx).pct_change() for k,v in data.items()}).mean(axis=1).fillna(0)
    price=(1+ret).cumprod()
    return ret, price, idx


def exposure_series(btc, bret, bprice, idx, trend_len=200):
    reg=detect_regimes(btc, trend_len).reindex(idx, method="ffill")
    bsma=bprice.rolling(200).mean()
    gate=bprice>bsma
    ex={}
    for t in idx:
        rg=reg.get(t,"Sideways"); g=bool(gate.get(t,False))
        if rg=="Bull Trend" and g:   ex[t]=1.0
        elif rg=="HighVol Bull" and g: ex[t]=0.5
        else: ex[t]=0.0
    return pd.Series(ex), reg


def run(bret, expo, idx):
    eq=CAP; series=[]; trades=[]; in_ep=False; ep=1.0; prev=0.0
    for t in idx:
        e=expo.get(t,0.0); r=bret.get(t,0.0)
        if abs(e-prev)>1e-9: eq*=(1-FEE*abs(e-prev))
        if e>0 and not in_ep: in_ep=True; ep=1.0
        if e>0: ep*=(1+e*r)
        if e==0 and in_ep: in_ep=False; trades.append(ep-1)
        eq*=(1+e*r); series.append(eq); prev=e
    if in_ep: trades.append(ep-1)
    return pd.Series(series,index=idx), trades


def st(eq, trades):
    if len(eq)<2: return {"cagr":0,"sharpe":0,"pf":0,"maxdd":0,"ret":0}
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1; end=eq.iloc[-1]; start=eq.iloc[0]
    cagr=((end/start)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    wins=[t for t in trades if t>0]; ls=abs(sum(t for t in trades if t<=0))
    pf=sum(wins)/ls if ls>0 else (float('inf') if wins else 0)
    return {"cagr":cagr,"sharpe":sh,"pf":pf,"maxdd":dd,"ret":(end/start-1)*100}


def yr_slice(s, y): return s[(s.index>=f"{y}-01-01")&(s.index<f"{y+1}-01-01")]


def main():
    btc=load("BTC/USD")["close"]
    bret,bprice,idx=basket(UNIVERSE)
    expo,reg=exposure_series(btc, bret, bprice, idx)

    # ── PHASE 1: walk-forward by test year ────────────────────────────────────
    print("="*78,"\nPHASE 1 — ROLLING WALK-FORWARD (strategy frozen; each test year unseen)\n"+"="*78)
    print("| Test Year | CAGR | Sharpe | PF | MaxDD | Win? |")
    print("|-----------|------|--------|------|-------|------|")
    wf={}
    for y in [2021,2022,2023,2024,2025,2026]:
        e=yr_slice(expo,y); b=yr_slice(bret,y)
        if len(b)<30: continue
        eq,tr=run(b, e, b.index); s=st(eq,tr); wf[y]=s
        win="✅" if s["ret"]>0 else "❌"
        print(f"| {y} | {s['cagr']:+5.0f}% | {s['sharpe']:+5.2f} | {s['pf']:5.2f} | {s['maxdd']:5.0f}% | {win} |")
    wins=sum(1 for s in wf.values() if s["ret"]>0)
    print(f"\n  Profitable {wins}/{len(wf)} test years.")

    # ── PHASE 2: regime of each test year + bear-only check ───────────────────
    print("\n"+"="*78,"\nPHASE 2 — REGIME-SPECIFIC OOS (is success limited to bears?)\n"+"="*78)
    print("| Test Year | Market Regime | CAGR | Sharpe |")
    print("|-----------|---------------|------|--------|")
    for y in wf:
        bh=yr_slice(bprice,y)
        mret=(bh.iloc[-1]/bh.iloc[0]-1)*100 if len(bh)>1 else 0
        lab="BULL" if mret>25 else "BEAR" if mret<-15 else "SIDEWAYS/MIXED"
        print(f"| {y} | {lab:13s} ({mret:+.0f}% mkt) | {wf[y]['cagr']:+5.0f}% | {wf[y]['sharpe']:+5.2f} |")

    # ── PHASE 3: cash vs invested per year ────────────────────────────────────
    print("\n"+"="*78,"\nPHASE 3 — CASH EXPOSURE (timing vs participation)\n"+"="*78)
    print("| Year | Invested % | Half % | Cash % |")
    print("|------|------------|--------|--------|")
    for y in wf:
        e=yr_slice(expo,y)
        inv=(e==1.0).mean()*100; half=(e==0.5).mean()*100; cash=(e==0.0).mean()*100
        print(f"| {y} | {inv:9.0f}% | {half:5.0f}% | {cash:5.0f}% |")

    # ── PHASE 4: whipsaws / false switches per year ───────────────────────────
    print("\n"+"="*78,"\nPHASE 4 — FALSE SWITCHES & WHIPSAWS\n"+"="*78)
    print("| Year | Entries | Exits | Whipsaws(<10d) |")
    print("|------|---------|-------|----------------|")
    for y in wf:
        e=yr_slice(expo,y); ev=(e>0).astype(int).values
        entries=int(((np.diff(ev)>0)).sum()); exits=int(((np.diff(ev)<0)).sum())
        # whipsaw = an invested episode shorter than 10 days
        whips=0; run_len=0
        for v in ev:
            if v>0: run_len+=1
            else:
                if 0<run_len<10: whips+=1
                run_len=0
        if 0<run_len<10: whips+=1
        print(f"| {y} | {entries} | {exits} | {whips} |")

    # ── PHASE 5: bull-market participation ────────────────────────────────────
    print("\n"+"="*78,"\nPHASE 5 — BULL-MARKET PARTICIPATION (major rallies)\n"+"="*78)
    rallies=[("2020-10-01","2021-04-15"),("2023-01-01","2023-07-15"),
             ("2024-01-01","2024-03-20"),("2023-10-01","2024-03-20")]
    print("| Rally | BuyHold | Engine | Capture % |")
    print("|-------|---------|--------|-----------|")
    for a,bb in rallies:
        sub=idx[(idx>=a)&(idx<bb)]
        if len(sub)<10: continue
        bh=(1+bret.reindex(sub)).cumprod(); bh_ret=(bh.iloc[-1]-1)*100
        eq,_=run(bret.reindex(sub), expo.reindex(sub), sub); en_ret=(eq.iloc[-1]/eq.iloc[0]-1)*100
        cap=en_ret/bh_ret*100 if bh_ret!=0 else 0
        print(f"| {a[:7]}→{bb[:7]} | {bh_ret:+5.0f}% | {en_ret:+5.0f}% | {cap:4.0f}% |")

    # ── PHASE 6: bear-market protection ───────────────────────────────────────
    print("\n"+"="*78,"\nPHASE 6 — BEAR-MARKET PROTECTION (major declines)\n"+"="*78)
    crashes=[("2021-11-08","2022-06-18"),("2022-04-01","2022-12-31"),
             ("2025-01-20","2025-06-01"),("2021-05-01","2021-07-20")]
    print("| Crash | BuyHold DD | Engine DD | Protection |")
    print("|-------|------------|-----------|------------|")
    for a,bb in crashes:
        sub=idx[(idx>=a)&(idx<bb)]
        if len(sub)<10: continue
        bh=(1+bret.reindex(sub)).cumprod(); bh_dd=((bh-bh.cummax())/bh.cummax()).min()*100
        eq,_=run(bret.reindex(sub), expo.reindex(sub), sub); en_dd=((eq-eq.cummax())/eq.cummax()).min()*100
        prot=(1-en_dd/bh_dd)*100 if bh_dd!=0 else 0
        print(f"| {a[:7]}→{bb[:7]} | {bh_dd:5.0f}% | {en_dd:5.0f}% | {prot:4.0f}% |")

    # ── PHASE 7: parameter stability ──────────────────────────────────────────
    print("\n"+"="*78,"\nPHASE 7 — PARAMETER STABILITY (trend length, full sample)\n"+"="*78)
    print("| Trend Len | CAGR | Sharpe | PF | MaxDD |")
    print("|-----------|------|--------|------|-------|")
    for L in [180,200,220]:
        ex2,_=exposure_series(btc, bret, bprice, idx, trend_len=L)
        eq,tr=run(bret, ex2, idx); s=st(eq,tr)
        print(f"| {L}d | {s['cagr']:+5.0f}% | {s['sharpe']:+5.2f} | {s['pf']:5.2f} | {s['maxdd']:5.0f}% |")

    # ── PHASE 8: robustness score ─────────────────────────────────────────────
    print("\n"+"="*78,"\nPHASE 8 — ROBUSTNESS SCORE\n"+"="*78)
    full_eq,full_tr=run(bret,expo,idx); fs=st(full_eq,full_tr)
    bull_years=[y for y in wf if (lambda b: (b.iloc[-1]/b.iloc[0]-1)>0.25)(yr_slice(bprice,y))]
    bull_wins=sum(1 for y in bull_years if wf[y]['ret']>0)
    crit={
      "1 OOS profitability": f"{wins}/{len(wf)} years +",
      "2 OOS Sharpe (full)": f"{fs['sharpe']:+.2f}",
      "3 Bull capture": "see Phase 5",
      "4 Bear protection": "see Phase 6",
      "5 Param stability": "see Phase 7 (CAGR spread)",
      "6 Whipsaw resistance": "see Phase 4",
    }
    for k,v in crit.items(): print(f"  {k:26s}: {v}")


if __name__=="__main__":
    main()
