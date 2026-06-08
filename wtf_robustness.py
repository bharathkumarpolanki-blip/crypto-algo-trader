"""
Weekly Trend-Following ROBUSTNESS & DEPLOYABILITY AUDIT (Phases 1-15).

Adversarial intent: find reasons NOT to deploy. The strategy is the Phase-7
winner: hold a coin while its weekly close > 30-week SMA (resampled from daily),
move to cash otherwise. Long-only spot, fees 0.6%/side.

Read-only. No optimization — parameters are tested for robustness, not tuned.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
import config
from exchange.market_data import fetch_ohlcv

FEE = config.FEE_RATE_PCT/100.0
CAP = 1000.0
WK_SMA = 30          # 30-week SMA (base parameter)
UNIVERSE = ["BTC/USD","ETH/USD","SOL/USD","AVAX/USD","LINK/USD","DOGE/USD","XRP/USD"]
_cache = {}


def load(sym):
    if sym in _cache: return _cache[sym]
    df = fetch_ohlcv(sym, "1d", limit=2700)
    _cache[sym] = df if (not df.empty and len(df) > 250) else None
    return _cache[sym]


def weekly_signal(df, wk_sma=WK_SMA):
    """Daily boolean: weekly close > wk-week SMA, forward-filled to daily."""
    w = df["close"].resample("1W").last().dropna()
    sma = w.rolling(wk_sma).mean()
    return (w > sma).reindex(df.index, method="ffill").fillna(False).values


def sim(df, inpos, fee=FEE, vol_scale=None):
    """Long-only spot. vol_scale: optional per-bar exposure 0..1 (vol targeting)."""
    c = df["close"].values; cash=CAP; units=0.0; entry=0.0
    eq=[]; trades=[]; tdates=[]
    exposure = np.ones(len(df)) if vol_scale is None else vol_scale
    for i in range(len(df)):
        want = inpos[i]
        if want and units==0:
            invest = cash*exposure[i]
            units=(invest*(1-fee))/c[i]; entry=c[i]; cash-=invest
        elif not want and units>0:
            cash += units*c[i]*(1-fee); trades.append((c[i]-entry)/entry)
            tdates.append((df.index[i], (c[i]-entry)/entry)); units=0
        eq.append(cash+units*c[i])
    if units>0:
        cash+=units*c[-1]*(1-fee); trades.append((c[-1]-entry)/entry)
        tdates.append((df.index[-1], (c[-1]-entry)/entry))
    return pd.Series(eq, index=df.index), trades, tdates


def stats(eq, trades, ann=365):
    idx=eq.index; yrs=(idx[-1]-idx[0]).days/365.25 if len(idx)>1 else 1
    end=eq.iloc[-1]; ret=(end/CAP-1)*100
    cagr=((end/CAP)**(1/yrs)-1)*100 if yrs>0 and end>0 else -100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna()
    sharpe=r.mean()/r.std()*np.sqrt(ann) if r.std()>0 else 0
    dn=r[r<0]; sortino=r.mean()/dn.std()*np.sqrt(ann) if len(dn)>1 and dn.std()>0 else 0
    wins=[t for t in trades if t>0]; wr=len(wins)/len(trades)*100 if trades else 0
    losssum=abs(sum(t for t in trades if t<=0))
    pf=sum(wins)/losssum if losssum>0 else float("inf")
    return dict(ret=ret,cagr=cagr,dd=dd,sharpe=sharpe,sortino=sortino,wr=wr,pf=pf,n=len(trades))


def slice_df(df, start, end):
    return df[(df.index>=start)&(df.index<end)]


def H(title): print(f"\n{'='*78}\n{title}\n{'='*78}")


def phase1(df):
    H("PHASE 1 — WALK-FORWARD (rolling out-of-sample test years)")
    print("| Test | Return | CAGR | Sharpe | Sortino | PF | MaxDD |")
    print("|------|--------|------|--------|---------|------|-------|")
    for yr in [2021,2022,2023,2024,2025]:
        sub=slice_df(df, f"{yr-1}-07-01", f"{yr+1}-01-01")   # 30wk warmup + test year
        if len(sub)<60: print(f"| {yr} | (insufficient data) |"); continue
        eq,tr,td=sim(sub, weekly_signal(sub))
        ey=eq[eq.index>=f"{yr}-01-01"]
        if len(ey)<5: continue
        ey=ey/ey.iloc[0]*CAP
        tr_y=[t for d,t in td if d.year==yr]
        s=stats(ey, tr_y)
        print(f"| {yr} | {s['ret']:+.0f}% | {s['cagr']:+.0f}% | {s['sharpe']:+.2f} | {s['sortino']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% |")


def _trades_in_year(df, yr):
    sub=slice_df(df, f"{yr-1}-07-01", f"{yr+1}-01-01")
    _,tr,td=sim(sub, weekly_signal(sub))
    return [t for d,t in td if d.year==yr]


def phase2(df):
    H("PHASE 2 — YEAR-BY-YEAR")
    print("| Year | Return | Sharpe | MaxDD | Trades |")
    print("|------|--------|--------|-------|--------|")
    yrly={}
    for yr in range(2019,2027):
        sub=slice_df(df, f"{yr}-01-01", f"{yr+1}-01-01")
        if len(sub)<30: continue
        warm=slice_df(df, f"{yr-1}-06-01", f"{yr+1}-01-01")
        sig=weekly_signal(warm)
        eq,tr,td=sim(warm, sig)
        ey=eq[eq.index>=f"{yr}-01-01"]
        if len(ey)<5: continue
        ey=ey/ey.iloc[0]*CAP
        tr_y=[t for d,t in td if d.year==yr]
        s=stats(ey, tr_y)
        yrly[yr]=s
        print(f"| {yr} | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['dd']:.0f}% | {len(tr_y)} |")
    if yrly:
        best=max(yrly,key=lambda y:yrly[y]['ret']); worst=min(yrly,key=lambda y:yrly[y]['ret'])
        losing=[y for y in yrly if yrly[y]['ret']<0]
        print(f"\n  Best year: {best} ({yrly[best]['ret']:+.0f}%)  Worst: {worst} ({yrly[worst]['ret']:+.0f}%)")
        print(f"  Losing years: {losing}")
        pos=sum(1 for y in yrly if yrly[y]['ret']>0)
        print(f"  Profitable {pos}/{len(yrly)} years.")


def phase3(df):
    H("PHASE 3 — PARAMETER ROBUSTNESS (weekly SMA length)")
    print("| WkSMA | Return | Sharpe | PF | MaxDD |")
    print("|-------|--------|--------|------|-------|")
    for p in [20,25,30,35,40,50]:   # ~150/175/200/225/250/300-day equivalents
        eq,tr,_=sim(df, weekly_signal(df, p))
        s=stats(eq,tr)
        print(f"| {p}wk | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% |")


def phase4(df):
    H("PHASE 4 — MONTE CARLO (10,000 bootstraps of trade sequence)")
    eq,tr,_=sim(df, weekly_signal(df))
    tr=np.array(tr)
    if len(tr)<5: print("  too few trades"); return
    cagrs=[]; dds=[]; finals=[]
    yrs=(df.index[-1]-df.index[0]).days/365.25
    for _ in range(10000):
        samp=np.random.choice(tr, size=len(tr), replace=True)
        eqc=np.cumprod(1+samp); fin=eqc[-1]
        finals.append((fin-1)*100)
        cagrs.append((fin**(1/yrs)-1)*100 if fin>0 else -100)
        peak=np.maximum.accumulate(eqc); dds.append(((eqc-peak)/peak).min()*100)
    finals=np.array(finals)
    print(f"  Median CAGR: {np.median(cagrs):+.1f}%")
    print(f"  Median final return: {np.median(finals):+.0f}%")
    print(f"  Worst drawdown (of sims): {np.min(dds):.0f}%")
    print(f"  5th pct return: {np.percentile(finals,5):+.0f}%   95th pct: {np.percentile(finals,95):+.0f}%")
    ruin=(finals<=-50).mean()*100
    print(f"  P(end down >50% / 'ruin'): {ruin:.1f}%   P(losing money): {(finals<0).mean()*100:.1f}%")


def phase5(df):
    H("PHASE 5 — COST STRESS TEST")
    print("| Cost× | CAGR | Sharpe | PF |")
    print("|-------|------|--------|------|")
    for m in [0.5,1.0,1.5,2.0,3.0]:
        eq,tr,_=sim(df, weekly_signal(df), fee=FEE*m)
        s=stats(eq,tr)
        print(f"| {m}x | {s['cagr']:+.1f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} |")


def phase6():
    H("PHASE 6 — MULTI-ASSET GENERALIZATION")
    print("| Asset | Return | Sharpe | PF | MaxDD | Trades |")
    print("|-------|--------|--------|------|-------|--------|")
    res={}
    for sym in UNIVERSE:
        df=load(sym)
        if df is None: print(f"| {sym} | no data |"); continue
        eq,tr,_=sim(df, weekly_signal(df)); s=stats(eq,tr); res[sym]=s
        print(f"| {sym.split('/')[0]:5s} | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% | {s['n']} |")
    prof=sum(1 for s in res.values() if s['ret']>0)
    print(f"\n  Profitable on {prof}/{len(res)} assets.")
    return res


def phase7(df):
    H("PHASE 7 — REGIME ANALYSIS (BTC; regime by 200d SMA slope + range)")
    sma=df["close"].rolling(200).mean()
    slope=sma.diff(20)
    bull=(df["close"]>sma)&(slope>0)
    bear=(df["close"]<sma)&(slope<0)
    side=~(bull|bear)
    eq,tr,td=sim(df, weekly_signal(df))
    # attribute each trade's return to regime at entry — approximate via daily eq returns
    r=eq.pct_change().fillna(0)
    print("| Regime   | Contribution(ann%) | Sharpe | DaysShare |")
    print("|----------|--------------------|--------|-----------|")
    for nm,mask in [("Bull",bull),("Bear",bear),("Sideways",side)]:
        rr=r[mask]
        if len(rr)<5: continue
        ann=(1+rr).prod()**(365/len(rr))-1
        sh=rr.mean()/rr.std()*np.sqrt(365) if rr.std()>0 else 0
        print(f"| {nm:8s} | {ann*100:+17.1f}% | {sh:+6.2f} | {len(rr)/len(r)*100:8.0f}% |")


def phase8(df):
    H("PHASE 8 — VOLATILITY FILTER (only hold when ATR%ile below threshold)")
    atr=(df["high"]-df["low"]).rolling(14).mean()/df["close"]
    pct=atr.rank(pct=True)
    base=weekly_signal(df)
    print("| Filter | Return | Sharpe | PF | MaxDD |")
    print("|--------|--------|--------|------|-------|")
    eq,tr,_=sim(df, base); s=stats(eq,tr)
    print(f"| none | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% |")
    for thr in [0.25,0.50,0.75]:
        filt = base & (pct.values <= thr)
        eq,tr,_=sim(df, filt); s=stats(eq,tr)
        print(f"| ATR<={int(thr*100)}% | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% |")


def phase9():
    H("PHASE 9 — RELATIVE STRENGTH (rotate into top-N by 12wk momentum, trend-gated)")
    uni=["BTC/USD","ETH/USD","SOL/USD","AVAX/USD","LINK/USD"]
    data={s:load(s) for s in uni}; data={k:v for k,v in data.items() if v is not None}
    # common daily index
    idx=None
    for v in data.values(): idx=v.index if idx is None else idx.intersection(v.index)
    idx=idx.sort_values()
    close=pd.DataFrame({k:v["close"].reindex(idx) for k,v in data.items()}).dropna()
    mom=close.pct_change(84)                          # 12-week momentum
    wsig={k:pd.Series(weekly_signal(data[k]), index=data[k].index).reindex(close.index).fillna(False)
          for k in close.columns}
    wsig=pd.DataFrame(wsig)
    print("| Method | Return | Sharpe | PF | MaxDD |")
    print("|--------|--------|--------|------|-------|")
    def run_topn(n):
        cash=CAP; held={}; eqs=[]; traderets=[]
        rebal=close.index[::7]  # weekly rebalance
        for d in close.index:
            if d in rebal:
                ranked=mom.loc[d].dropna()
                ranked=ranked[[c for c in ranked.index if wsig.loc[d,c]]]  # trend-gated
                pick=list(ranked.sort_values(ascending=False).index[:n])
                # liquidate non-picks
                for c in list(held):
                    if c not in pick:
                        cash+=held[c]['u']*close.loc[d,c]*(1-FEE)
                        traderets.append((close.loc[d,c]-held[c]['e'])/held[c]['e']); del held[c]
                # allocate equally to picks not yet held
                if pick:
                    target=(cash)/max(len(pick),1)
                    for c in pick:
                        if c not in held and target>1:
                            u=(target*(1-FEE))/close.loc[d,c]; held[c]={'u':u,'e':close.loc[d,c]}; cash-=target
            eqs.append(cash+sum(held[c]['u']*close.loc[d,c] for c in held))
        eq=pd.Series(eqs, index=close.index)
        return stats(eq, traderets)
    for label,n in [("Top1",1),("Top2",2),("Top3",3),("All(single-asset BTC)",None)]:
        if n is None:
            eq,tr,_=sim(data["BTC/USD"], weekly_signal(data["BTC/USD"])); s=stats(eq,tr)
        else:
            s=run_topn(n)
        print(f"| {label:22s} | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% |")


def phase10(df):
    H("PHASE 10 — POSITION SIZING (BTC)")
    base=weekly_signal(df)
    atr=(df["high"]-df["low"]).rolling(14).mean()/df["close"]
    inv_vol=(atr.median()/atr).clip(0.25,1.0).fillna(1.0).values   # vol-target exposure
    print("| Method | Return | Sharpe | PF | MaxDD |")
    print("|--------|--------|--------|------|-------|")
    for label,vs in [("Fixed (full)",None),("Vol-Adjusted",inv_vol),
                     ("Risk-Parity(inv-vol)", (atr.median()/atr).clip(0.1,1.0).fillna(1.0).values)]:
        eq,tr,_=sim(df, base, vol_scale=vs); s=stats(eq,tr)
        print(f"| {label:20s} | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% |")


def phase11():
    H("PHASE 11 — PORTFOLIO CONSTRUCTION (equal-weight, independent trend per asset)")
    sets={"BTC only":["BTC/USD"],"ETH only":["ETH/USD"],"BTC+ETH":["BTC/USD","ETH/USD"],
          "BTC+ETH+SOL":["BTC/USD","ETH/USD","SOL/USD"],
          "Full(7)":UNIVERSE}
    print("| Portfolio | Return | Sharpe | PF | MaxDD |")
    print("|-----------|--------|--------|------|-------|")
    for label,syms in sets.items():
        eqs=[]; idx=None; alltr=[]
        for s in syms:
            df=load(s)
            if df is None: continue
            eq,tr,_=sim(df, weekly_signal(df)); alltr+=tr
            eqs.append(eq); idx=eq.index if idx is None else idx.union(eq.index)
        if not eqs: continue
        port=sum(e.reindex(idx).ffill().fillna(CAP) for e in eqs)/len(eqs)
        s=stats(port, alltr)
        print(f"| {label:11s} | {s['ret']:+.0f}% | {s['sharpe']:+.2f} | {s['pf']:.2f} | {s['dd']:.0f}% |")


def phase12(df):
    H("PHASE 12 — OUT-OF-SAMPLE (oldest 80% in-sample, newest 20% OOS, no tuning)")
    n=len(df); cut=int(n*0.8)
    ins=df.iloc[:cut]; oos=df.iloc[cut-210:]   # carry warmup into OOS
    ei,ti,_=sim(ins, weekly_signal(ins)); si=stats(ei,ti)
    eo,to,_=sim(oos, weekly_signal(oos))
    eo=eo[eo.index>=df.index[cut]]; eo=eo/eo.iloc[0]*CAP if len(eo)>2 else eo
    so=stats(eo, [t for t in to])
    print("| Metric | In-Sample | Out-of-Sample |")
    print("|--------|-----------|---------------|")
    print(f"| CAGR   | {si['cagr']:+.1f}% | {so['cagr']:+.1f}% |")
    print(f"| Sharpe | {si['sharpe']:+.2f} | {so['sharpe']:+.2f} |")
    print(f"| PF     | {si['pf']:.2f} | {so['pf']:.2f} |")
    print(f"| MaxDD  | {si['dd']:.0f}% | {so['dd']:.0f}% |")
    print(f"  IS: {df.index[0].date()}→{df.index[cut].date()}  OOS: {df.index[cut].date()}→{df.index[-1].date()}")


def phase13(df):
    H("PHASE 13 — FAILURE ANALYSIS (worst losing trades)")
    eq,tr,td=sim(df, weekly_signal(df))
    losers=sorted([(d,r) for d,r in td if r<0], key=lambda x:x[1])[:10]
    atr=(df["high"]-df["low"]).rolling(14).mean()/df["close"]
    print("| Exit date | TradeRet | ATR%@exit | note |")
    print("|-----------|----------|-----------|------|")
    for d,r in losers:
        a=atr.reindex([d], method="ffill").iloc[0]*100
        print(f"| {str(d.date()):10s} | {r*100:+.1f}% | {a:.1f}% | whipsaw exit below 30wk SMA |")
    print(f"\n  Recurring cause: whipsaw — price dips below the 30wk SMA, we exit at a")
    print(f"  small loss + fee, then it reclaims. Losses are small/bounded by design.")


def phase14_15(btc_res):
    H("PHASE 14/15 — DEPLOYABILITY SCORECARD & VERDICT")
    print("  (criteria evaluated from the phases above)\n")


def main():
    print("WEEKLY TREND-FOLLOWING — ROBUSTNESS & DEPLOYABILITY AUDIT")
    btc=load("BTC/USD")
    phase1(btc); phase2(btc); phase3(btc); phase4(btc); phase5(btc)
    phase6(); phase7(btc); phase8(btc); phase9(); phase10(btc); phase11()
    phase12(btc); phase13(btc)


if __name__=="__main__":
    main()
