"""
NONLINEAR INTERACTION TEST — do the discovered signals combine into edge that
individual factors don't have? Uses decision tree / random forest / gradient
boosting. Strict T+1 target. The ONLY honest judge is OUT-OF-SAMPLE: a tree will
always find in-sample 'importance' in noise, so we time-split (train on older,
test on newest — NO shuffle) and compare the combined model's OOS predictive IC
against the best SINGLE factor's OOS IC.

Features (causal, close[T]): Trend, Volatility, Vol-Contraction, Market Structure,
Relative Strength, Momentum, Seasonality(month sin/cos), RSI.
Target: forward 20-day return entering at T+1.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from exchange.market_data import fetch_ohlcv

COINS=["BTC/USD","ETH/USD","SOL/USD","LINK/USD","XRP/USD"]
H=20
_cache={}
def load(s):
    if s not in _cache:
        d=fetch_ohlcv(s,"1d",limit=2500); _cache[s]=d if (not d.empty and len(d)>260) else None
    return _cache[s]


def build():
    # cross-sectional basket momentum for relative strength
    data={s:load(s) for s in COINS}; data={k:v for k,v in data.items() if v is not None}
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx.sort_values()
    closep=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()})
    basket_mom=closep.pct_change(30).mean(axis=1)

    frames=[]
    for s,df in data.items():
        c=df["close"]; h=df["high"]; l=df["low"]
        sma50=c.rolling(50).mean()
        atr=(h-l).rolling(14).mean()/c
        bbw=(c.rolling(20).std()*4)/c
        rng=(c.rolling(50).max()-c.rolling(50).min())
        donch=(c-c.rolling(50).min())/rng
        delta=c.diff(); up=delta.clip(lower=0).rolling(14).mean(); dn=(-delta.clip(upper=0)).rolling(14).mean()
        rsi=100-100/(1+up/dn.replace(0,np.nan))
        mo=df.index.month
        f=pd.DataFrame({
            "Trend":        (c/sma50-1),
            "Volatility":   atr,
            "VolContract":  -(bbw/bbw.rolling(60).mean()-1),
            "MarketStruct": donch-0.5,
            "RelStrength":  c.pct_change(30)-basket_mom.reindex(c.index),
            "Momentum":     c.pct_change(90),
            "Season_sin":   np.sin(2*np.pi*mo/12),
            "Season_cos":   np.cos(2*np.pi*mo/12),
            "RSI":          rsi/100,
        }, index=df.index)
        f["target"]=c.shift(-(H+1))/c.shift(-1)-1     # T+1 entry, 20d hold
        f["date"]=df.index
        frames.append(f)
    D=pd.concat(frames).dropna()
    return D


def ic(a,b):
    m=~(np.isnan(a)|np.isnan(b))
    if m.sum()<50: return 0.0
    r,_=stats.spearmanr(a[m],b[m]); return 0.0 if np.isnan(r) else r


def main():
    D=build()
    feats=["Trend","Volatility","VolContract","MarketStruct","RelStrength","Momentum","Season_sin","Season_cos","RSI"]
    # time-ordered split (NO shuffle): train older 70%, test newest 30%
    D=D.sort_values("date")
    cut=int(len(D)*0.7); tr=D.iloc[:cut]; te=D.iloc[cut:]
    Xtr,ytr=tr[feats].values, tr["target"].values
    Xte,yte=te[feats].values, te["target"].values
    print(f"NONLINEAR INTERACTION TEST | target=fwd{H}d (T+1) | {len(D)} samples "
          f"({len(tr)} train / {len(te)} test, time-split)\n")

    # ── Best SINGLE factor OOS IC (the bar to beat) ──────────────────────────
    print("### Single-factor OUT-OF-SAMPLE IC (the benchmark to beat)\n")
    print(f"| {'Feature':12s} | {'train IC':>8s} | {'OOS IC':>7s} | {'MutInfo':>7s} |")
    print("|"+"-"*14+"|"+"-"*10+"|"+"-"*9+"|"+"-"*9+"|")
    mi=mutual_info_regression(D[feats].values, D["target"].values, random_state=0)
    single_oos={}
    for i,f in enumerate(feats):
        tic=ic(tr[f].values,ytr); oic=ic(te[f].values,yte); single_oos[f]=oic
        print(f"| {f:12s} | {tic:+8.4f} | {oic:+7.4f} | {mi[i]:7.4f} |")
    best_single=max(single_oos.values(), key=abs)
    best_name=max(single_oos, key=lambda k:abs(single_oos[k]))
    print(f"\n  Best single-factor |OOS IC|: {best_name} = {single_oos[best_name]:+.4f}")

    # ── Models ───────────────────────────────────────────────────────────────
    print("\n### Combined models — does nonlinear interaction beat the best single factor?\n")
    models={
        "DecisionTree(d=4)": DecisionTreeRegressor(max_depth=4, min_samples_leaf=50, random_state=0),
        "RandomForest":      RandomForestRegressor(n_estimators=300, max_depth=5,
                                                   min_samples_leaf=50, n_jobs=-1, random_state=0),
        "GradientBoosting":  HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                           max_iter=300, l2_regularization=0.1, random_state=0),
    }
    print(f"| {'Model':18s} | {'train IC':>8s} | {'OOS IC':>7s} | {'beats best single?':>18s} |")
    print("|"+"-"*20+"|"+"-"*10+"|"+"-"*9+"|"+"-"*20+"|")
    fitted={}
    for nm,m in models.items():
        m.fit(Xtr,ytr); fitted[nm]=m
        tic=ic(m.predict(Xtr),ytr); oic=ic(m.predict(Xte),yte)
        beats="✅ YES" if oic>abs(best_single)+0.005 else "❌ no"
        print(f"| {nm:18s} | {tic:+8.4f} | {oic:+7.4f} | {beats:>18s} |")

    # ── Feature importance (RF gini) + permutation importance (OOS) ──────────
    rf=fitted["RandomForest"]
    print("\n### Feature importance — RandomForest\n")
    gini=rf.feature_importances_
    perm=permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0, n_jobs=-1)
    print(f"| {'Feature':12s} | {'Gini imp':>8s} | {'Perm imp (OOS)':>14s} |")
    print("|"+"-"*14+"|"+"-"*10+"|"+"-"*16+"|")
    order=np.argsort(-gini)
    for i in order:
        pm=perm.importances_mean[i]
        flag=" ✅" if pm>0 else ""
        print(f"| {feats[i]:12s} | {gini[i]:8.4f} | {pm:+14.5f}{flag} |")
    npos=int((perm.importances_mean>0).sum())
    print(f"\n  Features with POSITIVE out-of-sample permutation importance: {npos}/{len(feats)}")

    # ── Pairwise interaction probe (Trend × each) ─────────────────────────────
    print("\n### Pairwise OOS IC: 2-feature tree vs the better single factor\n")
    print(f"| {'Pair':28s} | {'pair OOS IC':>11s} | {'best single OOS':>15s} | {'interaction?':>12s} |")
    print("|"+"-"*30+"|"+"-"*13+"|"+"-"*17+"|"+"-"*14+"|")
    for b in ["Volatility","Season_sin","MarketStruct","RelStrength"]:
        cols=["Trend",b]; ix=[feats.index(x) for x in cols]
        t=DecisionTreeRegressor(max_depth=4,min_samples_leaf=50,random_state=0).fit(Xtr[:,ix],ytr)
        oic=ic(t.predict(Xte[:,ix]),yte)
        bs=max(single_oos["Trend"],single_oos[b],key=abs)
        inter="✅ adds" if oic>abs(bs)+0.005 else "❌ none"
        print(f"| {'Trend + '+b:28s} | {oic:+11.4f} | {bs:+15.4f} | {inter:>12s} |")


if __name__=="__main__":
    main()
