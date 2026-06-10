"""
ADVERSARIAL EDGE HUNT — cross-sectional, DOLLAR-NEUTRAL long/short. The strongest
untested avenue: market-neutral relative strategies (strip out beta entirely).

Tests, on a broad daily universe, strict T+1, fees+slippage:
  • Cross-sectional MOMENTUM   (long relative winners, short relative losers)
  • Cross-sectional REVERSAL   (long relative losers, short relative winners)
  across lookbacks. Each is dollar-neutral (sum w = 0) and gross-normalized.

Reports GROSS (signal exists?) and NET (tradeable after cost?), then for the best:
walk-forward by year, OOS, permutation, bootstrap. Trying hard to FIND an edge.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from exchange.market_data import fetch_ohlcv

UNIV=["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD",
      "DOT/USD","AVAX/USD","UNI/USD","ATOM/USD","LTC/USD","BCH/USD","XLM/USD",
      "ETC/USD","ALGO/USD","FIL/USD","AAVE/USD","NEAR/USD","CRV/USD"]
FEE=0.006; SLIP=0.0005           # spot taker
_cache={}
def load(s):
    if s not in _cache:
        try: d=fetch_ohlcv(s,"1d",limit=2500)
        except Exception: d=None
        _cache[s]=d if (d is not None and not d.empty and len(d)>700) else None
    return _cache[s]


def panel():
    data={s:load(s) for s in UNIV}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    return close, close.pct_change()


def xs_weights(signal):
    """Cross-sectional dollar-neutral weights from a signal df: demean each row,
       normalize gross exposure to 1. Only rows with >=4 valid names."""
    s=signal.copy()
    valid=s.notna().sum(axis=1)>=4
    dm=s.sub(s.mean(axis=1),axis=0)                 # demean -> dollar neutral
    gross=dm.abs().sum(axis=1).replace(0,np.nan)
    w=dm.div(gross,axis=0)
    w[~valid]=0
    return w.fillna(0)


def backtest(w, ret, fee=FEE, slip=SLIP):
    p=w.shift(1).fillna(0)                           # T+1 execution
    turn=(p-p.shift(1)).abs().sum(axis=1).fillna(p.abs().sum(axis=1))
    daily=(p*ret).sum(axis=1) - turn*(fee+slip)
    return daily


def sh(daily, ann=365):
    d=daily.dropna()
    return d.mean()/d.std()*np.sqrt(ann) if d.std()>0 else 0


def stats_(daily):
    d=daily.dropna(); eq=(1+d).cumprod()
    yrs=len(d)/365.25
    cg=(eq.iloc[-1]**(1/yrs)-1)*100 if eq.iloc[-1]>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    pos=d[d>0].sum(); neg=abs(d[d<0].sum()); pf=pos/neg if neg>0 else 0
    return cg, sh(d), pf, dd


def main():
    close,ret=panel()
    print(f"ADVERSARIAL cross-sectional L/S | {close.shape[1]} coins | "
          f"{close.index[0].date()}→{close.index[-1].date()} | strict T+1\n")

    print("### SCAN — gross (no fees) vs net (0.6%/side) Sharpe\n")
    print(f"| {'Strategy':22s} | {'lookback':>8s} | {'gross Sh':>8s} | {'net Sh':>7s} | {'net CAGR':>8s} |")
    print("|"+"-"*24+"|"+"-"*10+"|"+"-"*10+"|"+"-"*9+"|"+"-"*10+"|")
    cands={}
    for L in [3,5,10,20,30,60,90]:
        wm=xs_weights(close.pct_change(L))            # momentum: long winners
        dm_g=backtest(wm,ret,fee=0,slip=0); dm_n=backtest(wm,ret)
        cands[f"MOM L={L}"]=(wm,dm_n)
        print(f"| {'XS Momentum':22s} | {L:8d} | {sh(dm_g):+8.2f} | {sh(dm_n):+7.2f} | {stats_(dm_n)[0]:+7.0f}% |")
    for L in [1,2,3,5,10]:
        wr=-xs_weights(close.pct_change(L))           # reversal: long losers
        dr_g=backtest(wr,ret,fee=0,slip=0); dr_n=backtest(wr,ret)
        cands[f"REV L={L}"]=(wr,dr_n)
        print(f"| {'XS Reversal':22s} | {L:8d} | {sh(dr_g):+8.2f} | {sh(dr_n):+7.2f} | {stats_(dr_n)[0]:+7.0f}% |")

    # pick best by GROSS sharpe (does the signal exist at all?) and best by NET
    gross_rank=sorted(cands.items(), key=lambda kv: -sh(backtest(kv[1][0],ret,fee=0,slip=0)))
    best_gross=gross_rank[0]
    best_net=max(cands.items(), key=lambda kv: sh(kv[1][1]))
    print(f"\n  Best GROSS signal: {best_gross[0]}  (gross Sharpe {sh(backtest(best_gross[1][0],ret,fee=0,slip=0)):+.2f})")
    print(f"  Best NET strategy: {best_net[0]}  (net Sharpe {sh(best_net[1][1]):+.2f})")

    # ── Deep-validate the best GROSS signal (the one most likely to be real) ───
    name,(w,_)=best_gross
    print(f"\n### DEEP VALIDATION — {name}\n")
    # also test on cheaper-cost assumption (perp-like 0.05%/side) to separate
    # "no signal" from "signal eaten by spot fees"
    for lab,f,s in [("GROSS (0 cost)",0,0),("perp-like 0.05%/side",0.0005,0.0003),
                    ("spot 0.6%/side",FEE,SLIP)]:
        d=backtest(w,ret,fee=f,slip=s); cg,shh,pf,dd=stats_(d)
        print(f"  {lab:22s}: CAGR {cg:+6.1f}%  Sharpe {shh:+5.2f}  PF {pf:4.2f}  MaxDD {dd:5.0f}%")

    d=backtest(w,ret)                                 # net (spot)
    dg=backtest(w,ret,fee=0,slip=0)                   # gross
    # walk-forward by year (gross — is the SIGNAL stable across years?)
    print(f"\n  Walk-forward by year (GROSS Sharpe — is the signal time-stable?):")
    for y in range(2020,2027):
        m=(dg.index>=f"{y}-01-01")&(dg.index<f"{y+1}-01-01")
        if m.sum()<60: continue
        print(f"    {y}: gross Sharpe {sh(dg[m]):+.2f}   net Sharpe {sh(d[m]):+.2f}")
    # OOS
    cut=int(len(dg)*0.7)
    print(f"\n  OOS (newest 30%): gross Sharpe {sh(dg.iloc[cut:]):+.2f}  net Sharpe {sh(d.iloc[cut:]):+.2f}")
    # permutation: shuffle each row's weights across coins (destroys signal, keeps neutrality)
    rng=np.random.default_rng(0); base=sh(dg); perm=[]
    W=w.values
    for _ in range(1000):
        Wp=W.copy()
        for t in range(len(Wp)):
            Wp[t]=rng.permutation(Wp[t])
        pp=pd.DataFrame(Wp,index=w.index,columns=w.columns).shift(1).fillna(0)
        perm.append((pp*ret).sum(axis=1).pipe(sh))
    perm=np.array(perm); pval=(perm>=base).mean()
    print(f"  Permutation (shuffle weights across coins, 1000x): gross Sharpe {base:+.2f} "
          f"vs random mean {perm.mean():+.2f}  ->  p={pval:.4f}")
    # bootstrap CI on gross daily Sharpe
    arr=dg.dropna().values; boot=[]
    for _ in range(2000):
        s_=rng.choice(arr,len(arr),replace=True); boot.append(s_.mean()/s_.std()*np.sqrt(365) if s_.std()>0 else 0)
    print(f"  Bootstrap gross Sharpe 95% CI: [{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}]")

    print("\n### ADVERSARIAL CRITERIA")
    cg_n,sh_n,pf_n,dd_n=stats_(d)
    survive = (pval<0.05 and sh(dg.iloc[cut:])>0 and sh_n>1.0)
    print(f"  Signal significant (perm p<0.05)?     {'YES' if pval<0.05 else 'NO'} (p={pval:.4f})")
    print(f"  Survives OOS (gross)?                 {'YES' if sh(dg.iloc[cut:])>0 else 'NO'} (OOS gross Sh {sh(dg.iloc[cut:]):+.2f})")
    print(f"  TRADEABLE after spot fees (net Sh>1)? {'YES' if sh_n>1.0 else 'NO'} (net Sh {sh_n:+.2f})")
    print(f"\n  VERDICT: {'✅ EDGE SURVIVES' if survive else '❌ no tradeable edge (signal may exist gross but fails net/OOS)'}")


if __name__=="__main__":
    main()
