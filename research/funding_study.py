"""
FUNDING-RATE EDGE STUDY — real perpetual-futures funding data from OKX (public,
no keys). Tests whether derivatives positioning (funding) predicts forward spot
returns — genuinely NEW information vs price-only signals.

Signals:
  1. Extreme positive funding  (crowded longs  -> expect reversal DOWN)
  2. Extreme negative funding  (crowded shorts -> expect reversal UP)
  3. Funding acceleration      (change in funding)
  4. Funding divergence        (funding z-score vs price-momentum z-score)

Strict T+1. Funding aggregated to daily; predicts daily forward returns at 1/3/7d.
Metrics: IC, p-value, MI, plus a tradeable strategy (Sharpe/PF/OOS) on the best.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import ccxt

COINS=["BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT"]
CACHE="/tmp/okx_funding"
os.makedirs(CACHE, exist_ok=True)
FEE=0.0005; SLIP=0.0003          # perps are cheaper: ~0.05% taker + slip


def ex():
    return ccxt.okx({"enableRateLimit":True,"timeout":20000})


def fetch_funding(e, sym, days=900):
    f=f"{CACHE}/fund_{sym.replace('/','_').replace(':','_')}.csv"
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["dt"]).set_index("dt")["funding"]
    since=e.milliseconds()-days*86400000; allrows=[]; cursor=since
    while True:
        try:
            batch=e.fetch_funding_rate_history(sym, since=cursor, limit=100)
        except Exception as ex_:
            print(f"  {sym} funding fetch err: {str(ex_)[:80]}"); break
        if not batch: break
        allrows+=batch
        last=batch[-1]["timestamp"]
        if last<=cursor: break
        cursor=last+1
        if cursor>e.milliseconds(): break
        time.sleep(e.rateLimit/1000)
    if not allrows: return None
    df=pd.DataFrame([(r["timestamp"], r["fundingRate"]) for r in allrows], columns=["ts","funding"]).drop_duplicates("ts")
    df["dt"]=pd.to_datetime(df["ts"], unit="ms", utc=True)
    s=df.set_index("dt")["funding"].sort_index()
    s.to_csv(f, header=True)
    return s


def fetch_price(e, sym, days=900):
    f=f"{CACHE}/px_{sym.replace('/','_').replace(':','_')}.csv"
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["dt"]).set_index("dt")["close"]
    since=e.milliseconds()-days*86400000; allrows=[]; cursor=since
    while True:
        try:
            o=e.fetch_ohlcv(sym, "1d", since=cursor, limit=300)
        except Exception as ex_:
            print(f"  {sym} ohlcv err: {str(ex_)[:80]}"); break
        if not o: break
        allrows+=o; last=o[-1][0]
        if last<=cursor: break
        cursor=last+86400000
        if cursor>e.milliseconds(): break
        time.sleep(e.rateLimit/1000)
    if not allrows: return None
    df=pd.DataFrame(allrows, columns=["ts","o","h","l","close","v"]).drop_duplicates("ts")
    df["dt"]=pd.to_datetime(df["ts"], unit="ms", utc=True)
    s=df.set_index("dt")["close"].sort_index()
    s.to_csv(f, header=True)
    return s


def build():
    e=ex()
    rows=[]
    for sym in COINS:
        fund=fetch_funding(e,sym); px=fetch_price(e,sym)
        if fund is None or px is None: print(f"  {sym}: missing data"); continue
        # aggregate 8h funding to DAILY sum (daily carry), align to daily close
        fd=fund.resample("1D").sum()
        c=px.resample("1D").last()
        df=pd.DataFrame({"funding":fd,"close":c}).dropna()
        if len(df)<80: print(f"  {sym}: only {len(df)} days"); continue
        df["sym"]=sym; df["ret"]=df["close"].pct_change()
        rows.append(df)
        print(f"  {sym}: {len(df)} days  funding range [{df['funding'].min():.4f}, {df['funding'].max():.4f}]")
    return pd.concat(rows) if rows else None


def signals(df):
    f=df["funding"]; c=df["close"]; ret=df["ret"]
    fz=(f-f.rolling(30).mean())/f.rolling(30).std()           # funding z-score (extremeness)
    mom=c.pct_change(7); mz=(mom-mom.rolling(30).mean())/mom.rolling(30).std()
    return {
        # sign convention: POSITIVE signal = predicts price UP
        "Extreme +Funding (fade)":  -fz,                       # high funding -> short bias
        "Extreme -Funding (fade)":  -fz,                       # symmetric (same z, low funding -> long)
        "Funding Acceleration":     -f.diff(),                 # rising funding -> bearish
        "Funding Divergence":       -(fz - mz),                # funding hot vs price weak -> bearish
    }


def ic(a,b):
    m=~(np.isnan(a)|np.isnan(b))
    if m.sum()<100: return 0.0,1.0,0
    r,p=stats.spearmanr(a[m],b[m]); return (0.0 if np.isnan(r) else r),(1.0 if np.isnan(p) else p),int(m.sum())


def fwd(df,H):
    return df.groupby("sym")["close"].transform(lambda c: c.shift(-(H+1))/c.shift(-1)-1)


def main():
    print("FUNDING-RATE EDGE STUDY — OKX perps (real funding data)\n")
    D=build()
    if D is None: print("NO DATA — could not acquire funding/price."); return
    print(f"\nPooled: {len(D)} coin-days across {D['sym'].nunique()} coins, "
          f"{D.index.min().date()} → {D.index.max().date()}\n")

    print("### PHASE 1 — IC by signal & horizon (funding -> forward return, T+1)\n")
    print(f"| {'Signal':26s} | {'H':>3s} | {'IC':>8s} | {'p-value':>7s} | {'MI':>6s} | {'n':>5s} |")
    print("|"+"-"*28+"|"+"-"*5+"|"+"-"*10+"|"+"-"*9+"|"+"-"*8+"|"+"-"*7+"|")
    sigs=signals(D)
    best=None
    for H in [1,3,7]:
        fr=fwd(D,H)
        for nm,sg in sigs.items():
            i,p,n=ic(sg.values,fr.values)
            mm=pd.concat([sg,fr],axis=1).dropna()
            mi=float(mutual_info_regression(mm.iloc[:,[0]].values,mm.iloc[:,1].values,random_state=0)[0]) if len(mm)>100 else 0
            flag=" ✅" if p<0.05 else ""
            print(f"| {nm:26s} | {H:3d} | {i:+8.4f} | {p:7.3f} | {mi:6.3f} | {n:5d}{flag} |")
            if best is None or abs(i)>abs(best[2]): best=(nm,H,i,sg,fr)

    # ── PHASE 2 — tradeable strategy on the strongest signal (T+1) ────────────
    nm,H,i0,sg,fr=best
    print(f"\n### PHASE 2 — STRATEGY on strongest signal: '{nm}' (H={H}, T+1)\n")
    # long/short by signal sign, hold H days, per-coin, equal weight, perps fees
    df2=D.copy(); df2["sig"]=sg
    eqs=[]
    for s in df2["sym"].unique():
        sub=df2[df2["sym"]==s].copy()
        pos=np.sign(sub["sig"]).replace(0,np.nan).ffill().fillna(0)
        pos=pos.shift(1).fillna(0)                  # T+1
        turn=pos.diff().abs().fillna(pos.abs())
        daily=pos*sub["ret"] - turn*(FEE+SLIP)
        eqs.append(daily)
    P=pd.concat(eqs,axis=1).mean(axis=1).dropna()
    eq=(1+P).cumprod()
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1
    cagr=(eq.iloc[-1]**(1/yrs)-1)*100
    sh=P.mean()/P.std()*np.sqrt(365) if P.std()>0 else 0
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    pf=P[P>0].sum()/abs(P[P<0].sum()) if (P<0).any() else 0
    # OOS newest 30%
    cut=int(len(P)*0.7); oe=(1+P.iloc[cut:]).cumprod(); oP=P.iloc[cut:]
    osh=oP.mean()/oP.std()*np.sqrt(365) if oP.std()>0 else 0
    ocagr=(oe.iloc[-1]**(365.25/((oe.index[-1]-oe.index[0]).days))-1)*100 if len(oe)>2 else 0
    # permutation significance
    rng=np.random.default_rng(0); arr=P.values; perm=[]
    for _ in range(2000):
        s_=rng.permutation(arr); e=np.cumprod(1+s_); rr=np.diff(e)/e[:-1]
        perm.append(rr.mean()/rr.std()*np.sqrt(365) if rr.std()>0 else 0)
    pp=(np.array(perm)>=sh).mean()
    print(f"  Full: CAGR {cagr:+.1f}%  Sharpe {sh:+.2f}  PF {pf:.2f}  MaxDD {dd:.0f}%")
    print(f"  OOS (newest 30%): CAGR {ocagr:+.1f}%  Sharpe {osh:+.2f}")
    print(f"  Permutation p (random >= strategy Sharpe): {pp:.4f}")
    print(f"  Strongest signal IC: {i0:+.4f}")


if __name__=="__main__":
    main()
