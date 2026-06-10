"""
NEW-EDGE DISCOVERY (not optimization). Surveys every signal CATEGORY for genuine
predictive power against FORWARD returns. Strict T+1: signal computed at close[T]
predicts the return captured entering at T+1 (fwd = close[T+1+H]/close[T+1]-1).

DATA REALITY (honest): the keyless Coinbase client gives spot OHLCV only. So
funding rate, open interest, and on-chain metrics are NOT TESTABLE here — reported
as such rather than faked. All OHLCV-derivable categories ARE tested.

Metrics per category: IC (Spearman), p-value, Mutual Information, trade frequency,
gross edge (top-vs-bottom quintile fwd ret), net edge (gross - 1.3% round trip).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from exchange.market_data import fetch_ohlcv

COINS=["BTC/USD","ETH/USD","SOL/USD","LINK/USD","XRP/USD"]
RT_COST=1.30          # % round trip (0.6% fee + 0.05% slip, both sides)
_cache={}
def load(s):
    if s not in _cache:
        d=fetch_ohlcv(s,"1d",limit=2500); _cache[s]=d if (not d.empty and len(d)>260) else None
    return _cache[s]


def causal_signals(df):
    """All past-only. Sign convention: positive = predicts price UP."""
    c=df["close"]; o,h,l=df["open"],df["high"],df["low"]
    sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean()
    std20=c.rolling(20).std()
    atr=(h-l).rolling(14).mean()/c
    bbw=(c.rolling(20).std()*4)/c
    rng=(c.rolling(50).max()-c.rolling(50).min())
    donch=(c-c.rolling(50).min())/rng
    return {
        "Trend Following":      (c/sma50-1),
        "Mean Reversion":       -((c-sma20)/std20),
        "Momentum":             c.pct_change(90),
        "Volatility Expansion": (atr/atr.rolling(60).mean()-1),
        "Volatility Contraction": -(bbw/bbw.rolling(60).mean()-1),
        "Market Structure":     (donch-0.5),
    }


def fwd_return(c, H):
    """Return captured entering at T+1, exiting at T+1+H (strict, no same-bar)."""
    return c.shift(-(H+1))/c.shift(-1)-1


def ic_block(sig, fwd):
    s=pd.Series(sig); f=pd.Series(fwd)
    m=s.notna()&f.notna()
    if m.sum()<100 or s[m].std()==0: return (0.0,1.0,0.0,0)
    ic,p=stats.spearmanr(s[m],f[m])
    try: mi=float(mutual_info_regression(s[m].values.reshape(-1,1),f[m].values,random_state=0)[0])
    except Exception: mi=0.0
    return (0.0 if np.isnan(ic) else ic, 1.0 if np.isnan(p) else p, mi, int(m.sum()))


def quintile_edge(sig, fwd):
    s=pd.Series(sig); f=pd.Series(fwd); m=s.notna()&f.notna()
    s,f=s[m],f[m]
    if len(s)<200: return 0.0
    hi=f[s>=s.quantile(0.8)].mean(); lo=f[s<=s.quantile(0.2)].mean()
    return (hi-lo)*100   # long-top short-bottom gross, %


def pooled(category, H):
    sigs=[]; fwds=[]
    for s in COINS:
        df=load(s)
        if df is None: continue
        sg=causal_signals(df)[category]; fw=fwd_return(df["close"],H)
        sigs.append(sg.values); fwds.append(fw.values)
    return np.concatenate(sigs), np.concatenate(fwds)


def rel_strength(H):
    """Cross-sectional: coin 30d return minus cross-sectional mean. Pooled IC."""
    data={s:load(s) for s in COINS}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    mom=close.pct_change(30)
    rs=mom.sub(mom.mean(axis=1),axis=0)             # excess momentum vs basket
    fwd=close.shift(-(H+1))/close.shift(-1)-1
    return rs.values.flatten(), fwd.values.flatten()


def seasonality(H=1):
    """Month-of-year effect on next-day return; permutation p-value."""
    sigs=[]; months=[]
    for s in COINS:
        df=load(s)
        if df is None: continue
        f=fwd_return(df["close"],H).values
        mo=df.index.month.values
        sigs.append(f); months.append(mo)
    f=np.concatenate(sigs); mo=np.concatenate(months); m=~np.isnan(f)
    f,mo=f[m],mo[m]
    by=[f[mo==k] for k in range(1,13)]
    means=[x.mean() for x in by]
    spread=max(means)-min(means)
    rng=np.random.default_rng(0); perm=[]
    for _ in range(2000):
        sh=rng.permutation(mo); bym=[f[sh==k].mean() for k in range(1,13)]
        perm.append(max(bym)-min(bym))
    p=(np.array(perm)>=spread).mean()
    best=int(np.argmax(means)+1); worst=int(np.argmin(means)+1)
    return spread*100, p, best, worst


def main():
    H=10
    print(f"EDGE INVENTORY | {len(COINS)} coins daily | strict T+1 | primary horizon H={H}d | cost {RT_COST}%\n")
    print("DATA AVAILABILITY:")
    print("  ✅ OHLCV-derivable: Trend, MeanRev, RelStrength, Vol Expand/Contract, Momentum, Seasonality, Market Structure")
    print("  ❌ NOT TESTABLE (no data via Coinbase spot keyless): Funding Rate, Open Interest, On-Chain\n")

    print("### PHASE 1 — EDGE INVENTORY (pooled, H=10d)\n")
    print(f"| {'Category':22s} | {'IC':>8s} | {'p-value':>7s} | {'MI':>6s} | {'Freq/yr':>7s} | {'Gross%':>6s} | {'Net%':>6s} |")
    print("|"+"-"*24+"|"+"-"*10+"|"+"-"*9+"|"+"-"*8+"|"+"-"*9+"|"+"-"*8+"|"+"-"*8+"|")
    cats=["Trend Following","Mean Reversion","Momentum","Volatility Expansion",
          "Volatility Contraction","Market Structure"]
    rows=[]
    freq=365/H
    for cat in cats:
        sig,fwd=pooled(cat,H); ic,p,mi,n=ic_block(sig,fwd); g=quintile_edge(sig,fwd)
        net=g-RT_COST
        rows.append((cat,ic,p,mi,g,net))
        print(f"| {cat:22s} | {ic:+8.4f} | {p:7.3f} | {mi:6.3f} | {freq:7.0f} | {g:+6.2f} | {net:+6.2f} |")
    # relative strength
    rs,fwd=rel_strength(H); ic,p,mi,n=ic_block(rs,fwd); g=quintile_edge(rs,fwd)
    rows.append(("Relative Strength",ic,p,mi,g,g-RT_COST))
    print(f"| {'Relative Strength':22s} | {ic:+8.4f} | {p:7.3f} | {mi:6.3f} | {freq:7.0f} | {g:+6.2f} | {g-RT_COST:+6.2f} |")
    # seasonality
    sp,sp_p,best,worst=seasonality()
    print(f"| {'Seasonality (month)':22s} | {'n/a':>8s} | {sp_p:7.3f} | {'—':>6s} | {'—':>7s} | {sp:+6.2f} | {'—':>6s} |")
    print(f"    (best month {best}, worst {worst}, spread {sp:+.2f}% next-day, perm p={sp_p:.3f})")
    print("| Funding / OI / On-Chain | NOT TESTABLE — requires data beyond Coinbase spot OHLCV |")

    # ── PHASE 2: horizon discovery (best category by |IC|) ────────────────────
    best_cat=max(rows, key=lambda r: abs(r[1]))[0]
    print(f"\n### PHASE 2 — HORIZON DISCOVERY (strongest category: {best_cat})\n")
    print(f"| Horizon | IC | p-value | Net Edge% |")
    print("|---------|-----|---------|-----------|")
    for hl,hd in [("1D",1),("3D",3),("1W",7),("2W",14),("1M",30),("3M",90)]:
        if best_cat=="Relative Strength": sig,fwd=rel_strength(hd)
        else: sig,fwd=pooled(best_cat,hd)
        ic,p,mi,n=ic_block(sig,fwd); g=quintile_edge(sig,fwd)
        print(f"| {hl:7s} | {ic:+.4f} | {p:7.3f} | {g-RT_COST:+8.2f} |")

    # ── PHASE 3: regime study (best category) ─────────────────────────────────
    print(f"\n### PHASE 3 — MARKET-REGIME STUDY ({best_cat}, H={H}d IC by regime)\n")
    btc=load("BTC/USD")["close"]; sma=btc.rolling(200).mean(); slope=sma.diff(20)
    atrb=btc.pct_change().abs().rolling(14).mean()
    print("| Regime | IC | p-value | n |")
    print("|--------|-----|---------|---|")
    # build per-coin regime-tagged pools
    sig_all=[]; fwd_all=[]; reg_all=[]
    for s in COINS:
        df=load(s)
        if df is None: continue
        sg=(rel_strength.__wrapped__ if False else None)
        if best_cat=="Relative Strength": continue
        v=causal_signals(df)[best_cat]; f=fwd_return(df["close"],H)
        rg=pd.Series("Sideways",index=df.index)
        bsma=df["close"].rolling(200).mean(); bsl=bsma.diff(20)
        rg[(df["close"]>bsma)&(bsl>0)]="Bull"; rg[(df["close"]<bsma)&(bsl<0)]="Bear"
        av=df["close"].pct_change().abs().rolling(14).mean(); hv=av>av.rolling(180,min_periods=30).median()
        rg2=np.where(hv,"HighVol","LowVol")
        sig_all.append(v.values); fwd_all.append(f.values); reg_all.append(rg.values)
        # store vol regime too
    if sig_all:
        S=np.concatenate(sig_all); F=np.concatenate(fwd_all); R=np.concatenate(reg_all)
        for rg in ["Bull","Bear","Sideways"]:
            m=R==rg; ic,p,mi,n=ic_block(S[m],F[m])
            print(f"| {rg:6s} | {ic:+.4f} | {p:7.3f} | {int(n)} |")

    # ── PHASE 4: long/short asymmetry (best category) ─────────────────────────
    print(f"\n### PHASE 4 — LONG/SHORT ASYMMETRY ({best_cat}, H={H}d)\n")
    if best_cat=="Relative Strength": sig,fwd=rel_strength(H)
    else: sig,fwd=pooled(best_cat,H)
    s=pd.Series(sig); f=pd.Series(fwd); m=s.notna()&f.notna(); s,f=s[m],f[m]
    long_g=f[s>=s.quantile(0.8)].mean()*100; short_g=-f[s<=s.quantile(0.2)].mean()*100
    print(f"  Long  edge (top-quintile fwd): {long_g:+.2f}%  net {long_g-RT_COST:+.2f}%")
    print(f"  Short edge (bottom-quintile, inverted): {short_g:+.2f}%  net {short_g-RT_COST:+.2f}%")

    # ── PHASE 5: significance on best category ────────────────────────────────
    print(f"\n### PHASE 5 — STATISTICAL SIGNIFICANCE ({best_cat}, H={H}d)\n")
    ic0,p0,_,_=ic_block(sig,fwd)
    sv=pd.Series(sig); fv=pd.Series(fwd); m=sv.notna()&fv.notna(); sv,fv=sv[m].values,fv[m].values
    rng=np.random.default_rng(0)
    perm=[abs(stats.spearmanr(rng.permutation(sv),fv)[0]) for _ in range(2000)]
    p_perm=(np.array(perm)>=abs(ic0)).mean()
    boot=[stats.spearmanr(*(lambda i:(sv[i],fv[i]))(rng.integers(0,len(sv),len(sv))))[0] for _ in range(1000)]
    print(f"  Observed IC: {ic0:+.4f} (analytic p={p0:.4f})")
    print(f"  Permutation p (|IC_perm|>=|IC|): {p_perm:.4f}")
    print(f"  Bootstrap IC 95% CI: [{np.nanpercentile(boot,2.5):+.4f}, {np.nanpercentile(boot,97.5):+.4f}]")

    # ── PHASE 6: ranking ──────────────────────────────────────────────────────
    print("\n### PHASE 6 — EDGE RANKING (by |IC|, H=10d)\n")
    rows.sort(key=lambda r: -abs(r[1]))
    print(f"| Rank | Signal | IC | p-value | Net% |")
    print("|------|--------|-----|---------|------|")
    for i,(cat,ic,p,mi,g,net) in enumerate(rows,1):
        flag="✅sig" if p<0.05 else ""
        print(f"| {i} | {cat:22s} | {ic:+.4f} | {p:7.3f} | {net:+6.2f} {flag} |")


if __name__=="__main__":
    main()
