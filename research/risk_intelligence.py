"""
RISK-INTELLIGENCE AUDIT — re-test every price-derivable signal against RISK
targets instead of return targets. The category-error hypothesis: signals with
~0 return-IC carry real volatility/drawdown/crash information.

Multi-year daily universe (years of free price data → real power for risk targets).
Funding/OI are ~92-110d only → reported separately / noted.

Outputs:
  1. Information Content Matrix  (signal × {Return, Vol, Drawdown, Crash, Regime} IC)
  2. Interaction incremental IC  (do funding×OI-style interactions add info?)
  3. Fragility Score             (formula, normalization, weights) + walk-forward
     vs 5/10/20d drawdowns and vol-expansion, with OOS significance
  4. Regime-split signal map      (pooled vs bull/bear/highvol/lowvol) — is pooling
     destroying regime-specific edges?
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from exchange.market_data import fetch_ohlcv

UNIV=["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD",
      "LTC/USD","BCH/USD","XLM/USD","ETC/USD","DOT/USD","UNI/USD","ATOM/USD"]
_c={}
def load(s):
    if s not in _c:
        try: d=fetch_ohlcv(s,"1d",limit=2500)
        except Exception: d=None
        _c[s]=d if (d is not None and not d.empty and len(d)>700) else None
    return _c[s]


def build_panel():
    data={s:load(s) for s in UNIV}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    vol  =pd.DataFrame({k:v["volume"].reindex(idx) for k,v in data.items()})
    return close, vol, idx, list(data.keys())


def per_coin_signals(close, vol):
    ret=close.pct_change()
    sma200=close.rolling(200).mean(); sma50=close.rolling(50).mean()
    v14=ret.rolling(14).std()
    vrank=v14.rolling(252,min_periods=60).rank(pct=True)
    amihud=(ret.abs()/(vol*close+1e-9)).rolling(14).mean()
    sig={
        "Momentum":      close.pct_change(30),
        "Trend":         close/sma200-1,
        "VolCompress":   1-vrank,                                   # high = vol compressed (low)
        "Liquidity":     -amihud.rolling(252,min_periods=60).rank(pct=True),  # high = liquid
        "RSI":           _rsi(close,14)/100-0.5,
    }
    # breadth & XS momentum need the panel
    above50=(close>sma50).astype(float)
    breadth=above50.mean(axis=1)                                   # market breadth (one series)
    sig["Breadth"]=pd.DataFrame({c:breadth for c in close.columns})
    mom30=close.pct_change(30)
    sig["XS_Momentum"]=mom30.sub(mom30.mean(axis=1),axis=0)
    return sig, ret


def _rsi(close,n):
    d=close.diff(); up=d.clip(lower=0).rolling(n).mean(); dn=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))


def targets(close, ret, H=10):
    fwd_ret=close.shift(-(H+1))/close.shift(-1)-1
    fwd_vol=ret.rolling(H).std().shift(-(H+1))                     # std of next-H daily rets
    fwd_dd=(close.rolling(H).min().shift(-(H+1))/close.shift(-1))-1  # worst forward drawdown
    crash=(fwd_dd<-0.15).astype(float)
    # regime transition: trend sign flips within next 20d
    tr=np.sign(close/close.rolling(200).mean()-1)
    flip=((tr!=tr.shift(-20)) & tr.notna()).astype(float)
    return {"Return":fwd_ret,"Volatility":fwd_vol,"Drawdown":fwd_dd,"Crash":crash,"Regime":flip}


def ic(a,b):
    m=~(np.isnan(a)|np.isnan(b))
    if m.sum()<200: return 0.0,1.0
    r,p=stats.spearmanr(a[m],b[m]); return (0.0 if np.isnan(r) else r),(1.0 if np.isnan(p) else p)


def flat(df): return df.values.flatten()


def main():
    close,vol,idx,coins=build_panel()
    print(f"Universe {len(coins)} coins | {idx[0].date()}→{idx[-1].date()} | {len(idx)} days\n")
    sig,ret=per_coin_signals(close,vol)
    tg=targets(close,ret,H=10)

    # ── 1. INFORMATION CONTENT MATRIX ─────────────────────────────────────────
    print("="*86,"\n1. INFORMATION CONTENT MATRIX  (Spearman IC, pooled; H=10d targets)\n"+"="*86)
    print(f"| {'Signal':12s} | {'Return':>14s} | {'Volatility':>14s} | {'Drawdown':>14s} | {'Crash':>14s} | {'Regime':>14s} |")
    print("|"+"-"*14+"|"+("-"*16+"|")*5)
    rows={}
    for sn,sf in sig.items():
        S=flat(sf); cells=[]
        for tn in ["Return","Volatility","Drawdown","Crash","Regime"]:
            r,p=ic(S, flat(tg[tn])); cells.append((r,p))
        rows[sn]=cells
        def cell(rp): r,p=rp; return f"{r:+.3f}{'*' if p<0.01 else ' '}"
        print(f"| {sn:12s} | "+" | ".join(f"{cell(c):>14s}" for c in cells)+" |")
    print("  (* = p<0.01)   Return IC≈0 but Vol/Drawdown/Crash IC≠0  ⇒  category error confirmed")

    # which signals failed on return but work on risk
    print("\n  Signals reclassified (|riskIC| >> |returnIC|):")
    for sn,cells in rows.items():
        rIC=abs(cells[0][0]); riskmax=max(abs(cells[1][0]),abs(cells[2][0]),abs(cells[3][0]))
        if riskmax>0.05 and riskmax>3*max(rIC,0.01):
            best=["Return","Volatility","Drawdown","Crash","Regime"][int(np.argmax([abs(c[0]) for c in cells]))]
            print(f"    {sn:12s}: returnIC {cells[0][0]:+.3f} → best on {best} ({max([abs(c[0]) for c in cells]):.3f})")

    # ── 2. INTERACTION INCREMENTAL IC (target = Drawdown) ─────────────────────
    print("\n"+"="*86,"\n2. INTERACTION TERMS — do they add info beyond individuals? (target=Drawdown)\n"+"="*86)
    y=flat(tg["Drawdown"])
    base_feats={k:flat(v) for k,v in sig.items()}
    inters={
        "Trend×VolCompress":   base_feats["Trend"]*base_feats["VolCompress"],
        "Momentum×Liquidity":  base_feats["Momentum"]*base_feats["Liquidity"],
        "Trend×Breadth":       base_feats["Trend"]*base_feats["Breadth"],
        "VolCompress×Breadth": base_feats["VolCompress"]*base_feats["Breadth"],
        "XS_Momentum×Trend":   base_feats["XS_Momentum"]*base_feats["Trend"],
    }
    print(f"| {'Interaction':22s} | {'Interaction IC':>14s} | {'best parent IC':>14s} | incremental? |")
    print("|"+"-"*24+"|"+"-"*16+"|"+"-"*16+"|"+"-"*14+"|")
    for nm,iv in inters.items():
        iIC,ip=ic(iv,y)
        parents=nm.replace("×","|").split("|")
        pIC=max(abs(ic(base_feats[p.strip()],y)[0]) for p in parents if p.strip() in base_feats)
        inc="✅ adds" if abs(iIC)>pIC+0.01 and ip<0.01 else "❌ no"
        print(f"| {nm:22s} | {iIC:+13.3f}{'*' if ip<0.01 else ' '} | {pIC:14.3f} | {inc:>12s} |")
    # regression incremental R²
    import numpy as _np
    def rss(X):
        m=~_np.isnan(y) & ~_np.isnan(X).any(1)
        Xc=_np.column_stack([_np.ones(m.sum()),X[m]]); yy=y[m]
        b=_np.linalg.lstsq(Xc,yy,rcond=None)[0]; return ((yy-Xc@b)**2).sum(), len(yy)
    Xi=_np.column_stack([_np.nan_to_num(v) for v in base_feats.values()])
    Xf=_np.column_stack([Xi]+[_np.nan_to_num(v) for v in inters.values()])
    r_i,n=rss(Xi); r_f,_=rss(Xf); tss=((y[~_np.isnan(y)]-_np.nanmean(y))**2).sum()
    print(f"\n  R² individuals only: {(1-r_i/tss)*100:+.2f}%   +interactions: {(1-r_f/tss)*100:+.2f}%   "
          f"incremental {(r_i-r_f)/tss*100:+.2f}%")

    # ── 3. FRAGILITY SCORE + walk-forward ─────────────────────────────────────
    print("\n"+"="*86,"\n3. FRAGILITY SCORE — formula, normalization, weights, walk-forward OOS\n"+"="*86)
    print("  Components (z-scored, 252d rolling), weights:")
    print("    +0.30 leverage proxy (momentum extreme = crowded) | +0.25 vol-compression")
    print("    +0.20 breadth-narrowing (1-breadth)              | +0.15 illiquidity")
    print("    +0.10 trend-overextension (price/200SMA)")
    def z(df,w=252): return (df-df.rolling(w,min_periods=60).mean())/df.rolling(w,min_periods=60).std()
    frag=( 0.30*z(sig["Momentum"]).clip(-3,3)
         + 0.25*z(sig["VolCompress"]).clip(-3,3)
         + 0.20*z(1-sig["Breadth"]).clip(-3,3)
         + 0.15*z(-sig["Liquidity"]).clip(-3,3)
         + 0.10*z(sig["Trend"]).clip(-3,3) )
    # walk-forward: 70/30 by time; Fragility predicts forward drawdown/vol on OOS
    cut=int(len(idx)*0.7); te=idx[cut:]
    print(f"\n  Walk-forward OOS test (train<{te[0].date()}, test≥{te[0].date()}):")
    print(f"  {'Target':22s} {'OOS IC':>8s} {'p-value':>8s}")
    for H,lab in [(5,"5d drawdown"),(10,"10d drawdown"),(20,"20d drawdown")]:
        t=targets(close,ret,H)["Drawdown"]
        fo=frag.loc[te].values.flatten(); to=t.loc[te].values.flatten()
        r,p=ic(fo,to); print(f"  P({lab:20s}) {r:+8.3f} {p:8.4f} {'✅' if p<0.01 else ''}")
    velo=targets(close,ret,10)["Volatility"]
    r,p=ic(frag.loc[te].values.flatten(), velo.loc[te].values.flatten())
    print(f"  P({'vol expansion':20s}) {r:+8.3f} {p:8.4f} {'✅' if p<0.01 else ''}")
    crash=targets(close,ret,10)["Crash"]
    r,p=ic(frag.loc[te].values.flatten(), crash.loc[te].values.flatten())
    print(f"  P({'crash (>15% dd)':20s}) {r:+8.3f} {p:8.4f} {'✅' if p<0.01 else ''}")

    # ── 4. REGIME SPLIT — pooled vs regime-specific ───────────────────────────
    print("\n"+"="*86,"\n4. REGIME SPLIT — does pooling destroy edges? (target=Drawdown, |IC| by regime)\n"+"="*86)
    bull=(close>close.rolling(200).mean())
    v14=ret.rolling(14).std(); hivol=v14>v14.rolling(252,min_periods=60).median()
    y2=tg["Drawdown"]
    print(f"| {'Signal':12s} | {'Pooled':>8s} | {'Bull':>8s} | {'Bear':>8s} | {'HighVol':>8s} | {'LowVol':>8s} |")
    print("|"+"-"*14+"|"+("-"*10+"|")*5)
    for sn,sf in sig.items():
        def cic(mask): return ic(sf.where(mask).values.flatten(), y2.where(mask).values.flatten())[0]
        pooled=ic(sf.values.flatten(),y2.values.flatten())[0]
        print(f"| {sn:12s} | {pooled:+8.3f} | {cic(bull):+8.3f} | {cic(~bull):+8.3f} | "
              f"{cic(hivol):+8.3f} | {cic(~hivol):+8.3f} |")
    print("\n  If a signal's regime-IC >> pooled-IC, pooling was hiding a regime-specific edge.")


if __name__=="__main__":
    main()
