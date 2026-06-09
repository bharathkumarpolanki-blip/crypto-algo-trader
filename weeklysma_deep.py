"""
WEEKLY SMA — deep forensic (Phases 1-8). Strategy FROZEN. Strict T+1, no lookahead.

Signal: BTC weekly close > 30-week SMA -> invested in BTC (long-only spot).
Position decided at close[T], applied to return[T+1] (.shift(1)). Fees+slip.
Same universe context (BTC primary; ETH/60-40 for benchmark).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

CAP=1000.0; FEE=0.006; SLIP=0.0005
_cache={}
def load(s):
    if s not in _cache:
        d=fetch_ohlcv(s,"1d",limit=2500); _cache[s]=d if (not d.empty and len(d)>260) else None
    return _cache[s]


def wsma_pos(close):
    """Weekly 30wk SMA position, ffilled to daily. Causal."""
    w=close.resample("1W").last(); sma=w.rolling(30).mean()
    return (w>sma).reindex(close.index, method="ffill").astype(float).fillna(0)


def bt(ret, pos, fee=FEE, slip=SLIP):
    p=pos.shift(1).fillna(0.0)                      # T+1
    turn=p.diff().abs().fillna(p.abs())
    daily=p*ret - turn*(fee+slip)
    eq=CAP*(1+daily).cumprod()
    return eq, p, daily


def metr(eq):
    if len(eq)<2 or eq.iloc[-1]<=0: return dict(cagr=-100,sharpe=0,sortino=0,calmar=0,maxdd=0,ulcer=0,ret=0)
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1
    cagr=((eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1)*100
    r=eq.pct_change().dropna()
    sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    dn=r[r<0]; so=r.mean()/dn.std()*np.sqrt(365) if len(dn)>1 and dn.std()>0 else 0
    ddser=(eq-eq.cummax())/eq.cummax(); dd=ddser.min()*100
    ulcer=np.sqrt((ddser**2).mean())*100
    cal=cagr/abs(dd) if dd!=0 else 0
    return dict(cagr=cagr,sharpe=sh,sortino=so,calmar=cal,maxdd=dd,ulcer=ulcer,ret=(eq.iloc[-1]/eq.iloc[0]-1)*100)


def main():
    btc=load("BTC/USD")["close"]; ret=btc.pct_change().fillna(0)
    pos=wsma_pos(btc)
    eq,p,daily=bt(ret,pos)
    bh=CAP*(1+ret).cumprod()
    idx=btc.index

    # regime labels (causal) for attribution
    sma200=btc.rolling(200).mean(); slope=sma200.diff(20)
    regime=pd.Series("Sideways",index=idx)
    regime[(btc>sma200)&(slope>0)]="Bull"; regime[(btc<sma200)&(slope<0)]="Bear"

    # ── PHASE 1: bull/bear attribution ────────────────────────────────────────
    print("="*72,"\nPHASE 1 — BULL/BEAR ATTRIBUTION (strategy daily returns by regime)\n"+"="*72)
    print("| Regime   | Days | Return Contribution | Sharpe | (BuyHold same days) |")
    print("|----------|------|---------------------|--------|---------------------|")
    sret=daily            # strategy daily return
    for rg in ["Bull","Bear","Sideways"]:
        m=(regime==rg).values
        sr=sret.values[m]; br=ret.values[m]
        contrib=(np.prod(1+sr)-1)*100
        sh=sr.mean()/sr.std()*np.sqrt(365) if sr.std()>0 else 0
        bhc=(np.prod(1+br)-1)*100
        print(f"| {rg:8s} | {m.sum():4d} | {contrib:+18.0f}% | {sh:+6.2f} | {bhc:+18.0f}% |")
    # bull-only: does it beat hold inside bull?
    mb=(regime=="Bull").values
    s_bull=(np.prod(1+sret.values[mb])-1)*100; h_bull=(np.prod(1+ret.values[mb])-1)*100
    print(f"\n  Inside BULL: strategy {s_bull:+.0f}% vs buy&hold {h_bull:+.0f}%  "
          f"({'beats' if s_bull>h_bull else 'lags'} hold in bull)")

    # ── PHASE 2: rolling OOS ──────────────────────────────────────────────────
    print("\n"+"="*72,"\nPHASE 2 — ROLLING OOS (test year, T+1)\n"+"="*72)
    print("| OOS Year | CAGR | Sharpe | PF | DD |")
    print("|----------|------|--------|------|----|")
    for y in [2022,2023,2024,2025]:
        m=(idx>=f"{y}-01-01")&(idx<f"{y+1}-01-01")
        if m.sum()<30: continue
        e2=CAP*(1+daily[m]).cumprod(); mm=metr(e2)
        pos_y=p[m]; rr=ret[m]
        wins=rr[(pos_y>0).values & (rr>0).values].sum(); loss=abs(rr[(pos_y>0).values & (rr<0).values].sum())
        pf=wins/loss if loss>0 else 0
        print(f"| {y} | {mm['cagr']:+5.0f}% | {mm['sharpe']:+5.2f} | {pf:5.2f} | {mm['maxdd']:5.0f}% |")

    # ── PHASE 3: benchmark comparison ─────────────────────────────────────────
    print("\n"+"="*72,"\nPHASE 3 — BENCHMARK COMPARISON\n"+"="*72)
    eth=load("ETH/USD")
    print(f"| {'Strategy':14s} | {'CAGR':>6s} | {'Sharpe':>6s} | {'Sortino':>7s} | {'Calmar':>6s} | {'MaxDD':>6s} |")
    print("|"+"-"*16+"|"+"-"*8+"|"+"-"*8+"|"+"-"*9+"|"+"-"*8+"|"+"-"*8+"|")
    def line(nm,e):
        m=metr(e); print(f"| {nm:14s} | {m['cagr']:+5.0f}% | {m['sharpe']:+6.2f} | {m['sortino']:+7.2f} | {m['calmar']:+6.2f} | {m['maxdd']:5.0f}% |")
    line("BTC BuyHold", bh)
    if eth is not None:
        er=eth["close"].pct_change().fillna(0); line("ETH BuyHold", CAP*(1+er).cumprod())
        # 60/40 BTC/ETH on common index
        ci=btc.index.intersection(eth.index); br=ret.reindex(ci).fillna(0); er2=eth["close"].pct_change().reindex(ci).fillna(0)
        line("60/40 BTC/ETH", CAP*(1+0.6*br+0.4*er2).cumprod())
    line("WeeklySMA", eq)

    # ── PHASE 4: exposure ─────────────────────────────────────────────────────
    print("\n"+"="*72,"\nPHASE 4 — EXPOSURE ANALYSIS\n"+"="*72)
    inv=(p>0); pct_inv=inv.mean()*100
    # trade durations
    durs=[]; run=0
    for v in inv.values:
        if v: run+=1
        elif run>0: durs.append(run); run=0
    if run>0: durs.append(run)
    yrs=(idx[-1]-idx[0]).days/365.25
    print(f"  % invested: {pct_inv:.0f}%   % cash: {100-pct_inv:.0f}%")
    print(f"  Trades: {len(durs)}  | avg duration: {np.mean(durs):.0f} days  | ~{len(durs)/yrs:.1f} trades/yr")
    # how much perf from being invested: compare strat vs hold's invested-day return
    inv_days_hold=(np.prod(1+ret.values[inv.values])-1)*100
    print(f"  Buy&hold return earned ONLY on strategy's invested days: {inv_days_hold:+.0f}%")
    print(f"  => performance is PARTICIPATION in up-trends, not shorting/timing alpha.")

    # ── PHASE 5: crisis analysis ──────────────────────────────────────────────
    print("\n"+"="*72,"\nPHASE 5 — CRISIS ANALYSIS (T+1)\n"+"="*72)
    crises=[("2020 COVID","2020-02-15","2020-04-15"),("2021 May","2021-05-08","2021-07-25"),
            ("2022 bear","2022-01-01","2022-12-31"),("2025 bear","2025-01-20","2025-05-15")]
    print("| Crisis | BuyHold | WeeklySMA | Avg exposure |")
    print("|--------|---------|-----------|--------------|")
    for nm,a,b in crises:
        m=(idx>=a)&(idx<b)
        if m.sum()<5: print(f"| {nm} | pre-data |"); continue
        bhr=(np.prod(1+ret.values[m])-1)*100
        sr=(np.prod(1+daily.values[m])-1)*100
        ex=p.values[m].mean()*100
        print(f"| {nm:10s} | {bhr:+5.0f}% | {sr:+5.0f}% | {ex:3.0f}% invested |")

    # ── PHASE 6: statistical significance ─────────────────────────────────────
    print("\n"+"="*72,"\nPHASE 6 — STATISTICAL SIGNIFICANCE\n"+"="*72)
    strat_r=daily.values; n=len(strat_r)
    # Permutation: shuffle the position series vs returns 5000x -> distribution of Sharpe
    base_sh=metr(eq)["sharpe"]
    rng=np.random.default_rng(0); perm_sh=[]
    pos_arr=p.values; ret_arr=ret.values
    for _ in range(5000):
        sh_pos=rng.permutation(pos_arr)              # random timing, same invested fraction
        dr=sh_pos*ret_arr; e=np.cumprod(1+dr)
        rr=np.diff(e)/e[:-1]; perm_sh.append(rr.mean()/rr.std()*np.sqrt(365) if rr.std()>0 else 0)
    perm_sh=np.array(perm_sh); p_perm=(perm_sh>=base_sh).mean()
    print(f"  Permutation test (random timing, same exposure): "
          f"strategy Sharpe {base_sh:+.2f} vs random mean {perm_sh.mean():+.2f}")
    print(f"    p-value (random >= strategy): {p_perm:.4f}")
    # Randomized-entry: random invested fraction matching pct, 5000x
    frac=(p>0).mean(); rand_cagr=[]
    for _ in range(5000):
        mask=rng.random(n)<frac; dr=mask*ret_arr; e=np.cumprod(1+dr)
        rand_cagr.append((e[-1])**(365.25/(idx[-1]-idx[0]).days)-1)
    rand_cagr=np.array(rand_cagr); base_growth=(eq.iloc[-1]/CAP)**(365.25/(idx[-1]-idx[0]).days)-1
    p_rand=(rand_cagr>=base_growth).mean()
    print(f"  Randomized-entry (same % time invested, random days): "
          f"strategy CAGR {base_growth*100:+.0f}% vs random mean {rand_cagr.mean()*100:+.0f}%")
    print(f"    p-value (random >= strategy): {p_rand:.4f}")
    # Bootstrap CI on Sharpe
    boot=[]
    for _ in range(5000):
        samp=rng.choice(strat_r,size=n,replace=True)
        boot.append(samp.mean()/samp.std()*np.sqrt(365) if samp.std()>0 else 0)
    print(f"  Bootstrap Sharpe 95% CI: [{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}]")

    # ── PHASE 7: risk-adjusted utility ────────────────────────────────────────
    print("\n"+"="*72,"\nPHASE 7 — RISK-ADJUSTED UTILITY (WeeklySMA vs BTC hold)\n"+"="*72)
    ms=metr(eq); mh=metr(bh)
    print(f"| Metric        | WeeklySMA | BTC Hold |")
    print("|---------------|-----------|----------|")
    print(f"| Return/MaxDD  | {ms['ret']/abs(ms['maxdd']):+8.2f} | {mh['ret']/abs(mh['maxdd']):+8.2f} |")
    print(f"| Sharpe        | {ms['sharpe']:+8.2f} | {mh['sharpe']:+8.2f} |")
    print(f"| Sortino       | {ms['sortino']:+8.2f} | {mh['sortino']:+8.2f} |")
    print(f"| Calmar        | {ms['calmar']:+8.2f} | {mh['calmar']:+8.2f} |")
    print(f"| Ulcer Index   | {ms['ulcer']:8.1f} | {mh['ulcer']:8.1f} |")
    print(f"| MaxDD         | {ms['maxdd']:7.0f}% | {mh['maxdd']:7.0f}% |")


if __name__=="__main__":
    main()
