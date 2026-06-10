"""
FUNDING/OI INCREMENTAL-VALUE TEST — uses ONLY the already-collected OKX data.
Price history ~900d (200d SMA computable); funding ~92d, OI ~110d → evaluation
restricted to the funding∩OI overlap window. Strict T+1.

Three tests:
  1. REDUNDANCY — does funding/OI add predictive R² beyond momentum/vol/trend?
  2. SKEW/TAIL  — skewness & tail of funding-carry and funding-fade returns.
  3. BENCHMARKS — Buy&Hold vs 200d-SMA vs Vol-Target vs Funding/OI overlay,
                  with bootstrap CIs and permutation tests.

HONEST CAVEAT: ~92-day evaluation window = one regime, severely underpowered.
The redundancy regression (most obs) is the most informative; strategy Sharpes
have huge CIs. This is the $0 pre-check, not the multi-year confirmation.
"""
import sys, warnings, glob, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

CACHE="/tmp/okx_funding"
COINS=["BTC_USDT_USDT","ETH_USDT_USDT","SOL_USDT_USDT"]
FEE=0.0005; SLIP=0.0003     # perp costs


def load(coin):
    f=pd.read_csv(f"{CACHE}/fund_{coin}.csv",parse_dates=["dt"]).set_index("dt")["funding"]
    p=pd.read_csv(f"{CACHE}/px_{coin}.csv",parse_dates=["dt"]).set_index("dt")["close"]
    o=pd.read_csv(f"{CACHE}/oi_{coin}.csv",parse_dates=["dt"]).set_index("dt")["oi"]
    fd=f.resample("1D").sum(); px=p.resample("1D").last(); oi=o.resample("1D").last()
    df=pd.DataFrame({"funding":fd,"close":px,"oi":oi})
    df["ret"]=df["close"].pct_change()
    # price indicators on FULL history (200d SMA warmed up)
    df["sma200"]=df["close"].rolling(200).mean()
    df["mom7"]=df["close"].pct_change(7)
    df["vol14"]=df["ret"].rolling(14).std()
    df["trend200"]=df["close"]/df["sma200"]-1
    df["fund_z"]=(df["funding"]-df["funding"].rolling(30).mean())/df["funding"].rolling(30).std()
    df["oi_chg"]=df["oi"].pct_change()
    df["fwd1"]=df["close"].shift(-2)/df["close"].shift(-1)-1   # T+1 forward 1d
    return df


def zc(x):
    x=np.asarray(x,float); s=np.nanstd(x);
    return (x-np.nanmean(x))/s if s>0 else x*0


def main():
    dats={c:load(c) for c in COINS}
    # overlap window = where funding & oi both exist
    rows=[]
    for c,df in dats.items():
        w=df.dropna(subset=["funding","oi","close"])
        rows.append(w.assign(sym=c))
    P=pd.concat(rows)
    win=P.dropna(subset=["fund_z","oi_chg","mom7","vol14","trend200","fwd1"])
    print(f"Evaluation window: {win.index.min().date()} → {win.index.max().date()} "
          f"({win.index.nunique()} days, {len(win)} coin-days)\n")

    # ── TEST 1: REDUNDANCY ────────────────────────────────────────────────────
    print("="*70,"\nTEST 1 — REDUNDANCY (does funding/OI add R² beyond price?)\n"+"="*70)
    y=win["fwd1"].values
    Xp=np.column_stack([zc(win["mom7"]),zc(win["vol14"]),zc(win["trend200"])])   # price-only
    Xf=np.column_stack([Xp, zc(win["fund_z"]), zc(win["oi_chg"])])               # + funding/OI
    n=len(y)
    def ols_rss(X):
        Xc=np.column_stack([np.ones(n),X]); b=np.linalg.lstsq(Xc,y,rcond=None)[0]
        return ((y-Xc@b)**2).sum()
    rss_r=ols_rss(Xp); rss_f=ols_rss(Xf)
    tss=((y-y.mean())**2).sum()
    r2_p=1-rss_r/tss; r2_f=1-rss_f/tss
    q=2; dfree=n-Xf.shape[1]-1
    F=((rss_r-rss_f)/q)/(rss_f/dfree); pF=stats.f.sf(F,q,dfree)
    print(f"  Price-only R²:           {r2_p*100:+.3f}%")
    print(f"  Price+funding/OI R²:     {r2_f*100:+.3f}%")
    print(f"  Incremental R² (fund/OI):{(r2_f-r2_p)*100:+.3f}%")
    print(f"  F-test p (fund/OI add value beyond price): {pF:.4f}  "
          f"{'✅ adds value' if pF<0.05 else '❌ REDUNDANT'}")
    # partial correlation of fund_z with fwd1 controlling for price
    from numpy.linalg import lstsq
    def resid(t):
        Xc=np.column_stack([np.ones(n),Xp]); b=lstsq(Xc,t,rcond=None)[0]; return t-Xc@b
    pc=stats.spearmanr(resid(zc(win["fund_z"])), resid(y))
    print(f"  Partial corr funding⊥price vs fwd: {pc[0]:+.4f} (p={pc[1]:.3f})")

    # ── TEST 2: SKEW / TAIL ───────────────────────────────────────────────────
    print("\n"+"="*70,"\nTEST 2 — SKEW & TAIL (funding carry & funding-fade)\n"+"="*70)
    # carry: receive funding (≈ daily funding when positive carry)
    carry=win["funding"].values
    print(f"  Funding CARRY daily return:")
    print(f"    mean {np.mean(carry)*100:+.4f}%/d  skew {stats.skew(carry):+.2f}  "
          f"kurt {stats.kurtosis(carry):+.2f}  min {np.min(carry)*100:+.3f}%  "
          f"%neg {np.mean(carry<0)*100:.0f}%")
    # fade strategy daily P&L (per coin: pos=-sign(fund_z), T+1)
    fade=[]
    for c,df in dats.items():
        w=df.dropna(subset=["fund_z","ret"])
        pos=-np.sign(w["fund_z"]).shift(1).fillna(0)
        fade.append(pos*w["ret"]-pos.diff().abs().fillna(0)*(FEE+SLIP))
    fade=pd.concat(fade,axis=1).mean(axis=1).dropna()
    eqf=(1+fade).cumprod(); ddf=((eqf-eqf.cummax())/eqf.cummax()).min()*100
    print(f"  Funding FADE strategy daily P&L:")
    print(f"    mean {fade.mean()*100:+.4f}%/d  skew {stats.skew(fade):+.2f}  "
          f"kurt {stats.kurtosis(fade):+.2f}  worst {fade.min()*100:+.2f}%  MaxDD {ddf:.1f}%")
    print(f"    Sharpe {fade.mean()/fade.std()*np.sqrt(365):+.2f} (⚠ {len(fade)}d, huge CI)")

    # ── TEST 3: BENCHMARKS ────────────────────────────────────────────────────
    print("\n"+"="*70,"\nTEST 3 — BENCHMARKS (equal-weight BTC/ETH/SOL, T+1, over window)\n"+"="*70)
    # build per-coin daily strategy returns restricted to window dates
    wdates=win.index.unique()
    def basket(expo_fn):
        cols=[]
        for c,df in dats.items():
            d=df.loc[df.index.isin(wdates)].copy()
            e=expo_fn(df).reindex(d.index).shift(1).fillna(0)   # T+1 exposure
            turn=e.diff().abs().fillna(e.abs())
            cols.append(e*d["ret"]-turn*(FEE+SLIP))
        return pd.concat(cols,axis=1).mean(axis=1).dropna()
    target_vol=0.03
    strategies={
        "Buy & Hold":      lambda df: pd.Series(1.0,index=df.index),
        "200d SMA":        lambda df: (df["close"]>df["sma200"]).astype(float),
        "Vol-Target":      lambda df: (target_vol/df["vol14"]).clip(0,1.5).fillna(0),
        "Funding/OI ovly": lambda df: pd.Series(np.where((df["fund_z"]>1)&(df["oi_chg"]>0),0.5,
                                       np.where(df["fund_z"]>2,0.0,1.0)),index=df.index),
    }
    print(f"| {'Strategy':16s} | {'Tot ret':>7s} | {'Sharpe':>6s} | {'MaxDD':>6s} | {'Sharpe 95% CI':>16s} |")
    print("|"+"-"*18+"|"+"-"*9+"|"+"-"*8+"|"+"-"*8+"|"+"-"*18+"|")
    base=None; rng=np.random.default_rng(0)
    res={}
    for nm,fn in strategies.items():
        d=basket(fn); res[nm]=d
        eq=(1+d).cumprod(); tot=(eq.iloc[-1]-1)*100; dd=((eq-eq.cummax())/eq.cummax()).min()*100
        shp=d.mean()/d.std()*np.sqrt(365) if d.std()>0 else 0
        boot=[ (lambda s:s.mean()/s.std()*np.sqrt(365) if s.std()>0 else 0)(pd.Series(rng.choice(d.values,len(d),replace=True))) for _ in range(2000)]
        print(f"| {nm:16s} | {tot:+6.1f}% | {shp:+6.2f} | {dd:5.0f}% | "
              f"[{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}] |")
        if nm=="Buy & Hold": base=d
    # permutation: does funding/OI overlay beat Buy&Hold by chance?
    ov=res["Funding/OI ovly"]; idx=ov.index.intersection(base.index)
    diff=(ov.reindex(idx)-base.reindex(idx)).dropna()
    obs=diff.mean()/diff.std()*np.sqrt(365) if diff.std()>0 else 0
    perm=[]
    for _ in range(2000):
        s=rng.permutation(diff.values); perm.append(s.mean()/np.std(s)*np.sqrt(365) if np.std(s)>0 else 0)
    pov=(np.array(perm)>=obs).mean()
    print(f"\n  Overlay vs Buy&Hold excess-return Sharpe {obs:+.2f}, permutation p={pov:.3f} "
          f"{'✅' if pov<0.05 else '❌ not significant'}")

    print("\n"+"="*70,"\nCONCLUSION\n"+"="*70)
    print(f"  Incremental R² of funding/OI beyond price: {(r2_f-r2_p)*100:+.3f}%  (F-test p={pF:.3f})")
    print(f"  Funding carry skew: {stats.skew(carry):+.2f} | fade skew: {stats.skew(fade):+.2f}")
    verdict = (pF<0.05 and pov<0.05)
    print(f"  Funding/OI statistically significant incremental value? {'YES' if verdict else 'NO'}")


if __name__=="__main__":
    main()
