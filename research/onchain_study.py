"""
ON-CHAIN EDGE STUDY — real data from Coin Metrics Community API (free, no key),
12 years of daily history (unlike the 99-day derivatives cap → REAL OOS possible).

FREE / TESTED:   MVRV (CapMVRVCur), NUPL (= 1 - 1/MVRV), Active Addresses, NVT proxy.
PAYWALLED / NOT: Exchange in/out flows, whale-tier accumulation, dormant-coin
                 movement, realized P/L (SOPR) — Glassnode/CryptoQuant only.

Signals (sign convention: positive = predicts price UP):
  MVRV fade   : -zscore(MVRV)        (high MVRV = overvalued = bearish)
  NUPL fade   : -zscore(NUPL)        (monotone in MVRV — reported for completeness)
  ActiveAddr  : 30d growth of active addresses (network momentum)
  NVT fade    : -zscore(MktCap/Tx)   (high NVT = overvalued)

Strict T+1. Forward returns at 7/30/90d. Metrics: IC, MI, OOS, significance.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, requests
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

CM="https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FEE=0.006; SLIP=0.0005; CAP=1000.0


def fetch(asset, metrics, start="2014-01-01"):
    p={"assets":asset,"metrics":",".join(metrics),"frequency":"1d",
       "page_size":10000,"start_time":start}
    r=requests.get(CM,params=p,timeout=60); j=r.json()
    if "data" not in j:
        print("  fetch err:", j.get("error")); return None
    df=pd.DataFrame(j["data"])
    df["time"]=pd.to_datetime(df["time"]).dt.tz_localize(None)
    for m in metrics:
        if m in df: df[m]=pd.to_numeric(df[m],errors="coerce")
    return df.set_index("time")


def z(s,w=180): return (s-s.rolling(w).mean())/s.rolling(w).std()


def build():
    df=fetch("btc",["CapMVRVCur","CapMrktCurUSD","AdrActCnt","TxCnt"])
    if df is None: return None
    # price proxy from market cap / supply not available free -> use market cap as price proxy
    # (market cap ∝ price since supply changes slowly); returns ≈ price returns.
    df["price"]=df["CapMrktCurUSD"]
    df["MVRV"]=df["CapMVRVCur"]
    df["NUPL"]=1-1/df["MVRV"]
    df["NVT"]=df["CapMrktCurUSD"]/df["TxCnt"]
    df=df.dropna(subset=["MVRV","price"])
    print(f"  BTC on-chain: {len(df)} days, {df.index.min().date()} → {df.index.max().date()}")
    print(f"  MVRV range [{df['MVRV'].min():.2f}, {df['MVRV'].max():.2f}]  (high>3.5=top, low<1=bottom)")
    return df


def signals(df):
    return {
        "MVRV fade":        -z(df["MVRV"]),
        "NUPL fade":        -z(df["NUPL"]),
        "ActiveAddr 30d":   df["AdrActCnt"].pct_change(30),
        "NVT fade":         -z(df["NVT"]),
    }


def ic(a,b):
    m=~(np.isnan(a)|np.isnan(b))
    if m.sum()<100: return 0.0,1.0
    r,p=stats.spearmanr(a[m],b[m]); return (0.0 if np.isnan(r) else r),(1.0 if np.isnan(p) else p)


def main():
    print("ON-CHAIN EDGE STUDY — Coin Metrics Community (free, 12y)\n")
    df=build()
    if df is None: print("NO DATA."); return
    price=df["price"]
    sigs=signals(df)

    print("\n### PHASE 1 — IC & MutualInfo (full sample, T+1)\n")
    print(f"| {'Signal':16s} | {'H':>3s} | {'IC':>8s} | {'p-value':>7s} | {'MI':>6s} |")
    print("|"+"-"*18+"|"+"-"*5+"|"+"-"*10+"|"+"-"*9+"|"+"-"*8+"|")
    best=None
    for H in [7,30,90]:
        fwd=price.shift(-(H+1))/price.shift(-1)-1
        for nm,sg in sigs.items():
            i,p=ic(sg.values,fwd.values)
            mm=pd.concat([sg,fwd],axis=1).dropna()
            mi=float(mutual_info_regression(mm.iloc[:,[0]].values,mm.iloc[:,1].values,random_state=0)[0]) if len(mm)>200 else 0
            flag=" ✅" if p<0.05 else ""
            print(f"| {nm:16s} | {H:3d} | {i:+8.4f} | {p:7.3f} | {mi:6.3f}{flag} |")
            if best is None or abs(i)>abs(best[2]): best=(nm,H,i,sg)

    # ── PHASE 2 — OUT-OF-SAMPLE (older 70% train sign, newest 30% test) ───────
    nm,H,i0,sg=best
    print(f"\n### PHASE 2 — OUT-OF-SAMPLE on strongest: '{nm}' (H={H})\n")
    fwd=price.shift(-(H+1))/price.shift(-1)-1
    d=pd.concat([sg,fwd],axis=1).dropna(); d.columns=["sig","fwd"]
    cut=int(len(d)*0.7); tr=d.iloc[:cut]; te=d.iloc[cut:]
    ic_tr=ic(tr["sig"].values,tr["fwd"].values); ic_te=ic(te["sig"].values,te["fwd"].values)
    print(f"  Train IC: {ic_tr[0]:+.4f} (p={ic_tr[1]:.3f})   |   TEST IC: {ic_te[0]:+.4f} (p={ic_te[1]:.3f})")
    print(f"  Train: {tr.index.min().date()}→{tr.index.max().date()}  Test: {te.index.min().date()}→{te.index.max().date()}")

    # ── PHASE 3 — strategy (long when signal>0, cash else), T+1, fees ─────────
    print(f"\n### PHASE 3 — STRATEGY '{nm}>0 → long, else cash' (rebalance ~monthly, T+1)\n")
    pos=pd.Series(np.where(sg>0,1.0,0.0),index=sg.index)
    pos=pos.reindex(price.index).ffill()
    # monthly rebalance to cut churn
    keep=pos.copy(); keep[:]=np.nan; keep.iloc[::21]=pos.iloc[::21]; pos=keep.ffill().fillna(0)
    ret=price.pct_change().fillna(0)
    p1=pos.shift(1).fillna(0)
    turn=p1.diff().abs().fillna(p1.abs())
    daily=p1*ret - turn*(FEE+SLIP)
    eq=CAP*(1+daily).cumprod()
    bh=CAP*(1+ret).cumprod()
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    def stt(e,r):
        cg=(e.iloc[-1]/CAP)**(1/yrs)-1; dd=((e-e.cummax())/e.cummax()).min()
        sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
        return cg*100,sh,dd*100
    cg,sh,dd=stt(eq,daily); bcg,bsh,bdd=stt(bh,ret)
    # OOS
    cut2=int(len(daily)*0.7); od=daily.iloc[cut2:]; oe=(1+od).cumprod()
    oyrs=(od.index[-1]-od.index[0]).days/365.25; ocg=(oe.iloc[-1])**(1/oyrs)-1
    osh=od.mean()/od.std()*np.sqrt(365) if od.std()>0 else 0
    # permutation
    rng=np.random.default_rng(0); arr=daily.values; perm=[]
    for _ in range(2000):
        s_=rng.permutation(arr); e=np.cumprod(1+s_); rr=np.diff(e)/e[:-1]
        perm.append(rr.mean()/rr.std()*np.sqrt(365) if rr.std()>0 else 0)
    pp=(np.array(perm)>=sh).mean()
    print(f"  Strategy : CAGR {cg:+.1f}%  Sharpe {sh:+.2f}  MaxDD {dd:.0f}%")
    print(f"  Buy&Hold : CAGR {bcg:+.1f}%  Sharpe {bsh:+.2f}  MaxDD {bdd:.0f}%  (note: mktcap proxy)")
    print(f"  OOS (newest 30%): CAGR {ocg*100:+.1f}%  Sharpe {osh:+.2f}")
    print(f"  Permutation p (random ≥ strategy Sharpe): {pp:.4f}")

    print("\nNOTE: price proxy = BTC market cap (∝ price; supply ~flat). MVRV & NUPL are")
    print("rank-identical (NUPL = 1-1/MVRV), so their IC matches by construction.")


if __name__=="__main__":
    main()
