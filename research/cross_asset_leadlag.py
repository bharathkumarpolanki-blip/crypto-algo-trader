"""
CROSS-ASSET LEAD-LAG study. Does a leader's PAST predict a follower's FUTURE
(strictly after T+1) — or do they only move together contemporaneously (which is
NOT tradeable)?

Pairs: BTC->ETH, BTC->SOL, ETH->SOL, ETH->LINK, plus BTC-dominance (BTC -> alt basket).
Horizons: 1,3,7,14,30 days.
Metrics: lagged cross-correlation, IC (leader past-H ret -> follower fwd-H ret, T+1),
mutual information, and a Granger-style F-test (does leader's lag add predictive
power beyond the follower's own lag?). Strict T+1: signal at close[T], target from T+1.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from exchange.market_data import fetch_ohlcv

_cache={}
def load(s):
    if s not in _cache:
        d=fetch_ohlcv(s,"1d",limit=2500); _cache[s]=d if (not d.empty and len(d)>260) else None
    return _cache[s]


def aligned(*syms):
    data={s:load(s) for s in syms}
    if any(v is None for v in data.values()): return None
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.intersection(v.index)
    idx=idx.sort_values()
    return {s:data[s]["close"].reindex(idx) for s in syms}, idx


def ic(a,b):
    m=~(np.isnan(a)|np.isnan(b))
    if m.sum()<50: return 0.0,1.0
    r,p=stats.spearmanr(a[m],b[m]); return (0.0 if np.isnan(r) else r),(1.0 if np.isnan(p) else p)


def granger_p(leader_ret, follower_ret, H):
    """F-test: does leader's past-H return improve forecast of follower's fwd-H
       return beyond follower's OWN past-H return? (Granger-style, T+1 target)."""
    lp=leader_ret.rolling(H).sum()                 # leader past-H (known at T)
    fp=follower_ret.rolling(H).sum()               # follower own past-H (known at T)
    fy=follower_ret.shift(-1).rolling(H).sum().shift(-(H-1))  # follower fwd-H from T+1
    df=pd.concat([lp,fp,fy],axis=1).dropna(); df.columns=["lp","fp","fy"]
    if len(df)<100: return 1.0,0.0
    y=df["fy"].values; n=len(y)
    Xr=np.column_stack([np.ones(n),df["fp"].values])             # restricted: own past
    Xf=np.column_stack([np.ones(n),df["fp"].values,df["lp"].values])  # + leader past
    br,_,_,_=np.linalg.lstsq(Xr,y,rcond=None); rss_r=((y-Xr@br)**2).sum()
    bf,_,_,_=np.linalg.lstsq(Xf,y,rcond=None); rss_f=((y-Xf@bf)**2).sum()
    q=1; dfree=n-3
    F=((rss_r-rss_f)/q)/(rss_f/dfree) if rss_f>0 else 0
    p=stats.f.sf(F,q,dfree)
    r2_gain=(rss_r-rss_f)/rss_r if rss_r>0 else 0
    return p, r2_gain


def main():
    print("CROSS-ASSET LEAD-LAG | daily | strict T+1 (leader past -> follower future)\n")
    pairs=[("BTC/USD","ETH/USD"),("BTC/USD","SOL/USD"),
           ("ETH/USD","SOL/USD"),("ETH/USD","LINK/USD")]

    # ── PHASE 1: lagged cross-correlation (is there a PREDICTIVE lag?) ─────────
    print("### PHASE 1 — LAGGED CROSS-CORRELATION  corr(leader_ret[t-k], follower_ret[t])\n")
    print(f"| {'Pair':16s} | {'lag0 (same-day)':>15s} | {'lag1':>7s} | {'lag2':>7s} | {'lag3':>7s} |")
    print("|"+"-"*18+"|"+"-"*17+"|"+"-"*9+"|"+"-"*9+"|"+"-"*9+"|")
    for ld,fl in pairs:
        al=aligned(ld,fl)
        if al is None: print(f"| {ld[:3]}->{fl[:3]} | no data |"); continue
        cl,idx=al; lr=cl[ld].pct_change(); fr=cl[fl].pct_change()
        cors=[lr.shift(k).corr(fr) for k in range(4)]
        print(f"| {ld.split('/')[0]+'→'+fl.split('/')[0]:16s} | {cors[0]:+15.3f} | "
              f"{cors[1]:+7.3f} | {cors[2]:+7.3f} | {cors[3]:+7.3f} |")
    print("  (lag0 = contemporaneous, NOT tradeable. lag≥1 = leader genuinely leads = tradeable.)")

    # ── PHASE 2: IC + MI by horizon ───────────────────────────────────────────
    print("\n### PHASE 2 — IC & MutualInfo  (leader past-H ret  ->  follower fwd-H ret, T+1)\n")
    print(f"| {'Pair':16s} | {'H':>3s} | {'IC':>8s} | {'p-value':>7s} | {'MutInfo':>7s} |")
    print("|"+"-"*18+"|"+"-"*5+"|"+"-"*10+"|"+"-"*9+"|"+"-"*9+"|")
    for ld,fl in pairs:
        al=aligned(ld,fl)
        if al is None: continue
        cl,idx=al; lr=cl[ld].pct_change(); fr=cl[fl].pct_change()
        for H in [1,3,7,14,30]:
            sig=lr.rolling(H).sum()                          # leader past-H (at T)
            fwd=fr.shift(-1).rolling(H).sum().shift(-(H-1))  # follower fwd-H from T+1
            i,p=ic(sig.values,fwd.values)
            mm=pd.concat([sig,fwd],axis=1).dropna()
            mi=float(mutual_info_regression(mm.iloc[:,[0]].values, mm.iloc[:,1].values, random_state=0)[0]) if len(mm)>100 else 0
            flag=" ✅" if p<0.05 else ""
            print(f"| {ld.split('/')[0]+'→'+fl.split('/')[0]:16s} | {H:3d} | {i:+8.4f} | {p:7.3f} | {mi:7.4f}{flag} |")

    # ── PHASE 3: Granger-style F-test ─────────────────────────────────────────
    print("\n### PHASE 3 — GRANGER-STYLE F-TEST  (does leader add power beyond follower's own past?)\n")
    print(f"| {'Pair':16s} | {'H':>3s} | {'F-test p':>8s} | {'R² gain':>8s} | significant? |")
    print("|"+"-"*18+"|"+"-"*5+"|"+"-"*10+"|"+"-"*10+"|"+"-"*14+"|")
    for ld,fl in pairs:
        al=aligned(ld,fl)
        if al is None: continue
        cl,idx=al; lr=cl[ld].pct_change(); fr=cl[fl].pct_change()
        for H in [1,7,30]:
            p,r2=granger_p(lr,fr,H)
            print(f"| {ld.split('/')[0]+'→'+fl.split('/')[0]:16s} | {H:3d} | {p:8.4f} | {r2*100:+7.3f}% | "
                  f"{'✅ leads' if p<0.05 else '❌ no':>12s} |")

    # ── PHASE 4: BTC dominance effect (BTC -> alt basket) ──────────────────────
    print("\n### PHASE 4 — BTC DOMINANCE EFFECT  (BTC past -> alt-basket future)\n")
    al=aligned("BTC/USD","ETH/USD","SOL/USD","LINK/USD")
    if al:
        cl,idx=al
        btc=cl["BTC/USD"].pct_change()
        alt=pd.concat([cl["ETH/USD"].pct_change(),cl["SOL/USD"].pct_change(),
                       cl["LINK/USD"].pct_change()],axis=1).mean(axis=1)
        print(f"| {'Signal':28s} | {'H':>3s} | {'IC':>8s} | {'p-value':>7s} |")
        print("|"+"-"*30+"|"+"-"*5+"|"+"-"*10+"|"+"-"*9+"|")
        for H in [1,3,7,14,30]:
            sig=btc.rolling(H).sum()
            fwd=alt.shift(-1).rolling(H).sum().shift(-(H-1))
            i,p=ic(sig.values,fwd.values)
            print(f"| {'BTC past-H → alt-basket fwd':28s} | {H:3d} | {i:+8.4f} | {p:7.3f} |")
        # BTC relative strength -> alt forward (rotation/dominance rotation)
        rs=btc.rolling(30).sum()-alt.rolling(30).sum()    # BTC outperformance (dominance up)
        for H in [7,30]:
            fwd=alt.shift(-1).rolling(H).sum().shift(-(H-1))
            i,p=ic(rs.values,fwd.values)
            print(f"| {'BTC dominance (30d RS) → alt':28s} | {H:3d} | {i:+8.4f} | {p:7.3f} |")

    print("\n  Tradeable lead-lag needs: significant IC at lag≥1 / horizon (p<0.05) AND a")
    print("  Granger F-test p<0.05 (leader adds power beyond follower's own past).")


if __name__=="__main__":
    main()
