"""
ADVERSARIAL AUDIT of the Fragility Score — assume it is overfit and try to break it.
Score is FROZEN exactly (same components, same weights, same z-norm). NO tuning.

Hardest tests:
  A. Different YEARS      (per-year IC — does it hold every year or cherry-picked?)
  B. Different COINS      (expanded universe incl. coins not in original 7)
  C. Different VOL regimes(works in both high & low vol, or only one?)
  D. Different CAP buckets(large/mid/small — generalizes across cap spectrum?)
  + THE KILLER: volatility-persistence control (partial corr controlling for
    current realized vol — does Fragility's drawdown-IC survive removing vol-autocorr?)

Target = forward 10d drawdown (frozen). Reports best/expected/worst IC + P(significant).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from exchange.market_data import fetch_ohlcv

# EXPANDED universe (adds coins beyond the original 7 → out-of-universe test)
UNIV=["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD",
      "LTC/USD","BCH/USD","XLM/USD","ETC/USD","DOT/USD","UNI/USD","ATOM/USD",
      "AAVE/USD","CRV/USD","ALGO/USD","FIL/USD","NEAR/USD","MKR/USD"]
ORIG7={"BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD"}
LARGE={"BTC/USD","ETH/USD"}
MID={"SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD","LTC/USD","BCH/USD","DOT/USD"}
_c={}
def load(s):
    if s not in _c:
        try: d=fetch_ohlcv(s,"1d",limit=2500)
        except Exception: d=None
        _c[s]=d if (d is not None and not d.empty and len(d)>700) else None
    return _c[s]


def z(df,w=252): return (df-df.rolling(w,min_periods=60).mean())/df.rolling(w,min_periods=60).std()


def build(coins):
    data={s:load(s) for s in coins}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    vol=pd.DataFrame({k:v["volume"].reindex(idx) for k,v in data.items()})
    return close, vol, list(data.keys())


def fragility_FROZEN(close, vol):
    """EXACT frozen score — identical to risk_intelligence.py. No tuning."""
    ret=close.pct_change()
    v14=ret.rolling(14).std()
    vrank=v14.rolling(252,min_periods=60).rank(pct=True)
    amihud=(ret.abs()/(vol*close+1e-9)).rolling(14).mean()
    mom=close.pct_change(30)
    volcompress=1-vrank
    breadth=(close>close.rolling(50).mean()).astype(float).mean(axis=1)
    breadth=pd.DataFrame({c:breadth for c in close.columns})
    liquidity=-amihud.rolling(252,min_periods=60).rank(pct=True)
    trend=close/close.rolling(200).mean()-1
    frag=( 0.30*z(mom).clip(-3,3)
         + 0.25*z(volcompress).clip(-3,3)
         + 0.20*z(1-breadth).clip(-3,3)
         + 0.15*z(-liquidity).clip(-3,3)
         + 0.10*z(trend).clip(-3,3) )
    fwd_dd=(close.rolling(10).min().shift(-11)/close.shift(-1))-1
    return frag, fwd_dd, v14


def ic(a,b):
    m=~(np.isnan(a)|np.isnan(b))
    if m.sum()<150: return 0.0,1.0,int(m.sum())
    r,p=stats.spearmanr(a[m],b[m]); return (0.0 if np.isnan(r) else r),(1.0 if np.isnan(p) else p),int(m.sum())


def partial_ic(x,y,ctrl):
    """Partial Spearman of x,y controlling for ctrl (vol-persistence killer)."""
    m=~(np.isnan(x)|np.isnan(y)|np.isnan(ctrl))
    if m.sum()<150: return 0.0
    rxy=stats.spearmanr(x[m],y[m])[0]; rxz=stats.spearmanr(x[m],ctrl[m])[0]; ryz=stats.spearmanr(y[m],ctrl[m])[0]
    den=np.sqrt((1-rxz**2)*(1-ryz**2))
    return (rxy-rxz*ryz)/den if den>0 else 0.0


def main():
    close,vol,coins=build(UNIV)
    frag,dd,v14=fragility_FROZEN(close,vol)
    idx=close.index
    print(f"FROZEN Fragility audit | {len(coins)} coins | {idx[0].date()}→{idx[-1].date()}\n")

    base_ic,base_p,n=ic(frag.values.flatten(), dd.values.flatten())
    print(f"Baseline pooled IC (expanded universe): {base_ic:+.3f} (p={base_p:.4f}, n={n})\n")

    # ── A. DIFFERENT YEARS ────────────────────────────────────────────────────
    print("A. PER-YEAR IC (does it hold every year?):")
    yr_ics=[]
    for y in range(2020,2027):
        m=(idx>=f"{y}-01-01")&(idx<f"{y+1}-01-01")
        if m.sum()<30: continue
        r,p,nn=ic(frag.loc[m].values.flatten(), dd.loc[m].values.flatten())
        yr_ics.append(r); print(f"   {y}: IC {r:+.3f} (p={p:.3f}) {'✅' if p<0.05 else '❌'}")
    pos=sum(1 for r in yr_ics if r>0)
    print(f"   → positive in {pos}/{len(yr_ics)} years")

    # ── B. DIFFERENT COINS (out-of-original-universe) ─────────────────────────
    print("\nB. COIN GENERALIZATION:")
    for grp,label in [(ORIG7,"original 7"),(set(coins)-ORIG7,"NEW coins (never used)")]:
        cols=[c for c in coins if c in grp]
        if not cols: continue
        r,p,nn=ic(frag[cols].values.flatten(), dd[cols].values.flatten())
        print(f"   {label:24s}: IC {r:+.3f} (p={p:.4f}) {'✅' if p<0.05 else '❌'}")

    # ── C. DIFFERENT VOL REGIMES ──────────────────────────────────────────────
    print("\nC. VOL REGIME:")
    hivol=v14>v14.rolling(252,min_periods=60).median()
    for mask,label in [(hivol,"High-vol days"),(~hivol,"Low-vol days")]:
        r,p,nn=ic(frag.where(mask).values.flatten(), dd.where(mask).values.flatten())
        print(f"   {label:16s}: IC {r:+.3f} (p={p:.4f}) {'✅' if p<0.05 else '❌'}")

    # ── D. CAP BUCKETS ────────────────────────────────────────────────────────
    print("\nD. CAP BUCKET:")
    for grp,label in [(LARGE,"Large (BTC/ETH)"),(MID,"Mid"),(set(coins)-LARGE-MID,"Small")]:
        cols=[c for c in coins if c in grp]
        if not cols: continue
        r,p,nn=ic(frag[cols].values.flatten(), dd[cols].values.flatten())
        print(f"   {label:16s}: IC {r:+.3f} (p={p:.4f}) {'✅' if p<0.05 else '❌'}")

    # ── KILLER: volatility-persistence control ────────────────────────────────
    print("\n★ VOLATILITY-PERSISTENCE CONTROL (partial IC | current vol):")
    raw,_,_=ic(frag.values.flatten(), dd.values.flatten())
    pic=partial_ic(frag.values.flatten(), dd.values.flatten(), v14.values.flatten())
    print(f"   Raw IC: {raw:+.3f}  →  Partial IC (controlling for current vol): {pic:+.3f}")
    print(f"   Fraction surviving vol-control: {pic/raw*100 if raw!=0 else 0:.0f}%")

    # ── SUMMARY: best / expected / worst ──────────────────────────────────────
    all_ics=yr_ics+[ic(frag[[c for c in coins if c in (set(coins)-ORIG7)]].values.flatten(),
                       dd[[c for c in coins if c in (set(coins)-ORIG7)]].values.flatten())[0]]
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"  Best-case IC   (strongest year/regime) : {max(yr_ics):+.3f}")
    print(f"  Expected IC    (pooled, frozen)        : {base_ic:+.3f}")
    print(f"  Worst-case IC  (weakest year)          : {min(yr_ics):+.3f}")
    print(f"  Vol-controlled IC (the honest core)    : {pic:+.3f}")


if __name__=="__main__":
    main()
