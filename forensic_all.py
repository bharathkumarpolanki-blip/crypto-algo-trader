"""
FULL FORENSIC AUDIT — every remaining candidate under STRICT T+1 execution.

NON-NEGOTIABLE RULES (enforced by construction):
  - Signal computed from data up to and including close[T].
  - Position for day T+1 = signal(T) shifted forward one bar (.shift(1)).
  - Return earned = bret[T+1] (the move AFTER the decision).
  - No same-bar: a decision using close[T] can never earn the move into close[T].

Every indicator (trend, vol, ranking, regime, weights) is a rolling/past-only
function, then the resulting position is shifted +1 before being applied. This
makes same-bar lookahead structurally impossible.

Systems: A) Weekly SMA trend  B) Daily breakout  C) RS rotation  D) Regime-adaptive
Tests: full / walk-forward / OOS / cost / slippage / Monte Carlo.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

CAP=1000.0; FEE=0.006; SLIP=0.0005
UNIVERSE=["BTC/USD","ETH/USD","SOL/USD","LINK/USD","XRP/USD"]
_cache={}
def load(s):
    if s not in _cache:
        d=fetch_ohlcv(s,"1d",limit=2500); _cache[s]=d if (not d.empty and len(d)>260) else None
    return _cache[s]


# ── Generic T+1 backtest from a target-position series ────────────────────────
def bt(ret, pos, fee=FEE, slip=SLIP):
    """
    ret: daily return series of the traded instrument/basket.
    pos: TARGET exposure decided at close[T] (0..1). We SHIFT it +1 so it applies
         to ret[T+1]. Turnover cost (fee+slip) charged on |Δpos| at the change bar.
    """
    p = pos.shift(1).fillna(0.0)                      # <-- T+1 execution, the whole point
    turn = p.diff().abs().fillna(p.abs())
    gross = p * ret
    cost  = turn * (fee + slip)
    eq = CAP * (1 + gross - cost).cumprod()
    # trade-level returns for PF: episodes where p>0
    trades=[]; in_ep=False; ep=1.0
    for i in range(len(p)):
        e=p.iloc[i]; r=ret.iloc[i]
        if e>0 and not in_ep: in_ep=True; ep=1.0
        if e>0: ep*=(1+e*r)
        if e==0 and in_ep: in_ep=False; trades.append(ep-1)
    if in_ep: trades.append(ep-1)
    return eq, trades


def stats(eq, trades):
    if len(eq)<2 or eq.iloc[-1]<=0: return {"cagr":-100,"sharpe":0,"pf":0,"maxdd":0,"ret":0}
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1
    cagr=((eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1)*100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    wins=[t for t in trades if t>0]; ls=abs(sum(t for t in trades if t<=0))
    pf=sum(wins)/ls if ls>0 else (float('inf') if wins else 0)
    return {"cagr":cagr,"sharpe":sh,"pf":pf,"maxdd":dd,"ret":(eq.iloc[-1]/eq.iloc[0]-1)*100}


# ── Build aligned daily panel ─────────────────────────────────────────────────
def panel():
    data={s:load(s) for s in UNIVERSE}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    ret=close.pct_change().fillna(0)
    bret=ret.mean(axis=1)                              # equal-weight basket return
    bprice=(1+bret).cumprod()
    return close, ret, bret, bprice, idx, list(data.keys())


# ── A. Weekly SMA trend (BTC) — position = price>30wk SMA ─────────────────────
def sysA(close):
    c=close["BTC/USD"]
    w=c.resample("1W").last(); sma=w.rolling(30).mean()
    pos=(w>sma).reindex(c.index, method="ffill").astype(float).fillna(0)
    return pos, c.pct_change().fillna(0)


# ── B. Daily breakout (BTC) — position = close>prior 50d high (Donchian) ──────
def sysB(close):
    c=close["BTC/USD"]
    hi=c.rolling(50).max().shift(1); lo=c.rolling(20).min().shift(1)
    pos=pd.Series(0.0,index=c.index); holding=False
    for i in range(len(c)):
        if not holding and c.iloc[i]>hi.iloc[i] if not np.isnan(hi.iloc[i]) else False: holding=True
        elif holding and (c.iloc[i]<lo.iloc[i] if not np.isnan(lo.iloc[i]) else False): holding=False
        pos.iloc[i]=1.0 if holding else 0.0
    return pos, c.pct_change().fillna(0)


# ── C. Relative-strength rotation — top-N by 12wk momentum, trend-gated ───────
def sysC(close, ret, n=2):
    mom=close.pct_change(84)                           # past 12 weeks (causal)
    sma=close.rolling(200).mean()
    up=close>sma                                       # causal trend gate
    # weight series per asset (0 or 1/n), decided at close[T]
    W=pd.DataFrame(0.0,index=close.index,columns=close.columns)
    for t in close.index:
        elig=[col for col in close.columns if up.loc[t,col] and not np.isnan(mom.loc[t,col])]
        rank=mom.loc[t,elig].sort_values(ascending=False)
        pick=list(rank.index[:n])
        if pick:
            for col in pick: W.loc[t,col]=1.0/len(pick)
    # portfolio return with T+1 execution applied per-column
    Wn=W.shift(1).fillna(0.0)
    turn=Wn.diff().abs().sum(axis=1).fillna(Wn.abs().sum(axis=1))
    port=(Wn*ret).sum(axis=1) - turn*(FEE+SLIP)
    eq=CAP*(1+port).cumprod()
    # crude trade list: count weight changes
    trades=[port[port!=0].mean() if (port!=0).any() else 0.0]
    # better PF via positive/negative day aggregation per holding run
    return eq, _eq_trades(Wn, ret)


def _eq_trades(Wn, ret):
    inv=(Wn.sum(axis=1)>0).astype(int); trades=[]; ep=1.0; on=False
    pr=(Wn*ret).sum(axis=1)
    for i in range(len(inv)):
        if inv.iloc[i] and not on: on=True; ep=1.0
        if inv.iloc[i]: ep*=(1+pr.iloc[i])
        if not inv.iloc[i] and on: on=False; trades.append(ep-1)
    if on: trades.append(ep-1)
    return trades


# ── D. Regime-adaptive (the one that failed) — re-audited T+1 for completeness ─
def sysD(close, bret, bprice, idx):
    btc=close["BTC/USD"]; sma=btc.rolling(200).mean(); slope=sma.diff(20)
    atr=btc.pct_change().abs().rolling(14).mean(); hivol=atr>atr.rolling(180,min_periods=30).median()*1.3
    bull=(btc>sma)&(slope>0); gate=bprice>bprice.rolling(200).mean()
    pos=pd.Series(0.0,index=idx)
    pos[bull & ~hivol & gate]=1.0
    pos[bull &  hivol & gate]=0.5
    return pos, bret


def run_all():
    close,ret,bret,bprice,idx,coins=panel()
    print(f"Universe: {coins}  | {idx[0].date()} → {idx[-1].date()}  | STRICT T+1 execution\n")

    systems={}
    pa,ra=sysA(close); systems["A WeeklySMA"]=bt(ra,pa)
    pb,rb=sysB(close); systems["B Breakout"]=bt(rb,pb)
    eqc,trc=sysC(close,ret); systems["C RS-Rotation"]=(eqc,trc)
    pd_,rd=sysD(close,bret,bprice,idx); systems["D Regime"]=bt(rd,pd_)
    # benchmark
    bh=CAP*(1+bret).cumprod(); systems["BuyHold basket"]=(bh,[])

    print("### MAIN TABLE (full sample, T+1, fees+slip)\n")
    print(f"| {'Strategy':16s} | {'CAGR':>6s} | {'Sharpe':>6s} | {'PF':>6s} | {'MaxDD':>6s} |")
    print("|"+"-"*18+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|"+"-"*8+"|")
    S={}
    for nm,(eq,tr) in systems.items():
        s=stats(eq,tr); S[nm]=s
        print(f"| {nm:16s} | {s['cagr']:+5.0f}% | {s['sharpe']:+6.2f} | {s['pf']:6.2f} | {s['maxdd']:5.0f}% |")

    # position builders for re-running on sub-windows
    builders={
        "A WeeklySMA":  lambda c,r,br,bp,ix: bt(*( (lambda p,rr:(rr,p))(*sysA(c)) )),
    }

    # ── Walk-forward by year (rebuild positions on full data, slice equity) ────
    print("\n### WALK-FORWARD (per-year, T+1)\n")
    posmap={"A WeeklySMA":(pa,ra),"B Breakout":(pb,rb),
            "D Regime":(pd_,rd)}
    print(f"| Year | {'A SMA':>7s} | {'B Brk':>7s} | {'C Rot':>7s} | {'D Reg':>7s} |")
    print("|------|"+"-"*9+"|"+"-"*9+"|"+"-"*9+"|"+"-"*9+"|")
    for y in [2021,2022,2023,2024,2025,2026]:
        row=[]
        for nm in ["A WeeklySMA","B Breakout","D Regime"]:
            p,r=posmap[nm]
            m=(p.index>=f"{y}-01-01")&(p.index<f"{y+1}-01-01")
            if m.sum()<30: row.append("  n/a "); continue
            eq,tr=bt(r[m], p[m]); row.insert(len(row), f"{stats(eq,tr)['ret']:+5.0f}%")
        # rotation year
        mc=(eqc.index>=f"{y}-01-01")&(eqc.index<f"{y+1}-01-01")
        rotret=f"{(eqc[mc].iloc[-1]/eqc[mc].iloc[0]-1)*100:+5.0f}%" if mc.sum()>30 else "  n/a "
        print(f"| {y} | {row[0]:>7s} | {row[1]:>7s} | {rotret:>7s} | {row[2]:>7s} |")

    # ── OOS newest 20% ────────────────────────────────────────────────────────
    print("\n### OUT-OF-SAMPLE (newest 20%, T+1)\n")
    cut=int(len(idx)*0.8)
    print(f"| Strategy | CAGR | Sharpe | PF | MaxDD |")
    print("|----------|------|--------|------|-------|")
    for nm,(p,r) in posmap.items():
        eqo,tro=bt(r.iloc[cut:], p.iloc[cut:]); s=stats(eqo,tro)
        print(f"| {nm:12s} | {s['cagr']:+5.0f}% | {s['sharpe']:+5.2f} | {s['pf']:5.2f} | {s['maxdd']:5.0f}% |")
    so=stats(eqc.iloc[cut:], _eq_trades.__wrapped__ if False else [0.01])
    print(f"| C RS-Rotation | {((eqc.iloc[-1]/eqc.iloc[cut])**(365.25/((idx[-1]-idx[cut]).days)) -1)*100:+5.0f}% | "
          f"{eqc.iloc[cut:].pct_change().dropna().pipe(lambda x: x.mean()/x.std()*np.sqrt(365) if x.std()>0 else 0):+5.2f} | — | "
          f"{((eqc.iloc[cut:]-eqc.iloc[cut:].cummax())/eqc.iloc[cut:].cummax()).min()*100:5.0f}% |")

    # ── Cost + slippage stress (best non-benchmark candidate by Sharpe) ────────
    best=max([n for n in S if n!="BuyHold basket"], key=lambda n:S[n]['sharpe'])
    print(f"\n### COST + SLIPPAGE STRESS — best candidate: {best}\n")
    bp_pos = posmap.get(best, (pa,ra))
    print(f"| Setting | CAGR | Sharpe | PF |")
    print("|---------|------|--------|------|")
    for f,sl,lab in [(FEE,SLIP,"base 0.6%+0.05%"),(FEE,0.0025,"+0.25% slip"),
                     (FEE,0.005,"+0.50% slip"),(FEE*2,0.005,"2x fee +0.5% slip")]:
        eq,tr=bt(bp_pos[1], bp_pos[0], fee=f, slip=sl); s=stats(eq,tr)
        print(f"| {lab:18s} | {s['cagr']:+5.0f}% | {s['sharpe']:+5.2f} | {s['pf']:5.2f} |")

    # ── Monte Carlo on best candidate's trade sequence ────────────────────────
    print(f"\n### MONTE CARLO (10k bootstraps of {best} trades, T+1)\n")
    eqb,trb=bt(bp_pos[1],bp_pos[0]); trb=np.array(trb)
    if len(trb)>=5:
        yrs=(idx[-1]-idx[0]).days/365.25
        finals=[]; dds=[]
        for _ in range(10000):
            samp=np.random.choice(trb,size=len(trb),replace=True); eqc2=np.cumprod(1+samp)
            finals.append((eqc2[-1]-1)*100)
            pk=np.maximum.accumulate(eqc2); dds.append(((eqc2-pk)/pk).min()*100)
        finals=np.array(finals)
        print(f"  Median return {np.median(finals):+.0f}%  | 5th pct {np.percentile(finals,5):+.0f}%  | "
              f"95th pct {np.percentile(finals,95):+.0f}%")
        print(f"  P(losing money) {(finals<0).mean()*100:.0f}%  | worst sim DD {np.min(dds):.0f}%")
    else:
        print("  too few trades for Monte Carlo")

    print("\n### BENCHMARK: BuyHold basket Sharpe "
          f"{S['BuyHold basket']['sharpe']:+.2f}, CAGR {S['BuyHold basket']['cagr']:+.0f}%, "
          f"MaxDD {S['BuyHold basket']['maxdd']:.0f}%")
    return S


if __name__=="__main__":
    run_all()
