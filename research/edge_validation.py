"""
Post-fix EDGE VALIDATION — does the signal predict forward returns better than
random, after fees? Read-only. One signal pass per symbol; everything (trade
sims, IC, buckets, regimes, attribution, MI) derives from the stored series.

The core scientific test is the Information Coefficient: rank-correlation between
the signal (and each component) and the FORWARD return. Trade P&L is path- and
exit-dependent; IC isolates predictive value itself.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

import config
from exchange.market_data import fetch_ohlcv
from core.indicators import enrich
from core.strategies import analyse, score_market_regime, passes_conviction

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
DAYS = 180
FEE = config.FEE_RATE_PCT / 100.0
SLIP = 0.05 / 100.0          # 0.05%/side slippage+spread (realistic taker on majors)
RISK = config.RISK_PER_TRADE_PCT / 100.0
CAP0 = 1000.0
HORIZON = 6                  # forward-return horizon (bars) for IC
COMPONENTS = ["ema_trend","macd","rsi","stochastic","bollinger","adx","volume",
              "supertrend","ichimoku","support","regime","candle_patterns","trend_4h"]


def regime_label(df):
    r = score_market_regime(df)
    if r >= 1.5:  return "strong_bull"
    if r >= 0.5:  return "bull"
    if r <= -3.0: return "confirmed_bear"
    if r <= -1.0: return "below_200ema"
    return "neutral"


def collect(symbol):
    """One pass: store per-bar signal, components, regime, OHLC."""
    df1h = fetch_ohlcv(symbol, "1h", limit=DAYS*24+300)
    df6h = fetch_ohlcv(symbol, config.TF_TREND, limit=DAYS*6+100)
    df1d = fetch_ohlcv(symbol, "1d", limit=config.DAILY_GATE_SMA_PERIOD+400)
    if df1h.empty or len(df1h) < 300:
        return None
    df1h = enrich(df1h); df6h = enrich(df6h) if not df6h.empty else pd.DataFrame()
    span = (df1h.index[-1]-df1h.index[0]).days
    close = df1h["close"].values; high = df1h["high"].values; low = df1h["low"].values
    recs = []
    for i in range(100, len(df1h)):
        sl1 = df1h.iloc[max(0,i-300):i]
        ts = df1h.index[i]
        sl6 = df6h[df6h.index <= ts] if not df6h.empty else pd.DataFrame()
        if len(sl6) < 60: sl6 = pd.DataFrame()
        sl1d = df1d[df1d.index <= ts] if not df1d.empty else None
        if sl1d is not None and len(sl1d) < config.DAILY_GATE_SMA_PERIOD: sl1d = None
        sig = analyse(symbol, sl1, df_trend=sl6 if len(sl6)>=60 else None,
                      include_sentiment=False, include_ml=False, df_daily=sl1d)
        comps = sig.components
        recs.append({
            "i": i, "price": close[i], "atr": sig.atr, "dir": sig.direction,
            "score": sig.score, "raw": sum(comps.values()),
            "regime": regime_label(sl1), "daily": sig.daily_trend,
            **{f"c_{k}": comps.get(k, 0.0) for k in COMPONENTS},
        })
    return {"symbol": symbol, "span": span, "recs": recs,
            "close": close, "high": high, "low": low, "df": df1h}


def simulate(d, mode):
    """mode: 'long' | 'short' | 'both'. ATR exits, fees+slippage. Returns metrics."""
    recs = d["recs"]; high = d["high"]; low = d["low"]; close = d["close"]
    n = len(close)
    cap = CAP0; in_trade = False
    entry=stop=target=qty=0.0; side=""; ei=0
    trades = []; consec = 0
    by_i = {r["i"]: r for r in recs}
    i = recs[0]["i"]
    while i < n:
        r = by_i.get(i)
        # exit check on bar i
        if in_trade:
            hit_sl = (side=="long" and low[i]<=stop) or (side=="short" and high[i]>=stop)
            hit_tp = (side=="long" and high[i]>=target) or (side=="short" and low[i]<=target)
            if hit_sl or hit_tp:
                ep = target if hit_tp else stop
                # slippage: fills worse than the trigger
                ep = ep*(1-SLIP) if side=="long" else ep*(1+SLIP)
                gross = (ep-entry)*qty if side=="long" else (entry-ep)*qty
                fees = (entry*qty+ep*qty)*FEE
                pnl = gross-fees; cap += pnl
                trades.append({**by_entry, "pnl": pnl, "reason":"TP" if hit_tp else "SL",
                               "bars": i-ei})
                in_trade=False; consec = consec+1 if pnl<0 else 0
        # entry check
        if not in_trade and r is not None:
            if consec>=2: consec-=1; i+=1; continue
            allow = (mode=="both") or (r["dir"]==mode)
            if allow and r["dir"] in ("long","short") and passes_conviction_rec(r) and r["atr"]>0:
                price=r["price"]; atr=r["atr"]
                # entry slippage
                eprice = price*(1+SLIP) if r["dir"]=="long" else price*(1-SLIP)
                if r["dir"]=="long":
                    fs=eprice-config.ATR_STOP_MULTIPLIER*atr; ft=eprice+config.ATR_TARGET_MULTIPLIER*atr
                else:
                    fs=eprice+config.ATR_STOP_MULTIPLIER*atr; ft=eprice-config.ATR_TARGET_MULTIPLIER*atr
                rr=abs(ft-eprice)/abs(fs-eprice)
                if rr>=2.0:
                    rpu=abs(eprice-fs)
                    if rpu>0:
                        q=cap*RISK/rpu
                        g=abs(ft-eprice)*q; f=(eprice*q+ft*q)*FEE
                        if g-f>=config.MIN_NET_PROFIT_USD and g>=f*config.MIN_WIN_FEE_MULTIPLE:
                            entry,stop,target,qty,side=eprice,fs,ft,q,r["dir"]; ei=i
                            by_entry={"side":r["dir"],"score":r["score"],"regime":r["regime"],
                                      "daily":r["daily"],"raw":r["raw"],
                                      **{k:r[k] for k in r if k.startswith("c_")}}
                            in_trade=True
        i+=1
    if in_trade:
        lp=close[-1]; gross=(lp-entry)*qty if side=="long" else (entry-lp)*qty
        pnl=gross-(entry*qty+lp*qty)*FEE; cap+=pnl
        trades.append({**by_entry,"pnl":pnl,"reason":"eot","bars":0})
    return metrics(d, trades, cap)


# passes_conviction works on SignalResult; replicate on a record dict
def passes_conviction_rec(r):
    thr=config.MIN_SIGNAL_SCORE
    if r["dir"]=="long":  return r["score"]>=thr
    if r["dir"]=="short": return r["score"]<=(10-thr)
    return False


def metrics(d, trades, cap):
    n=len(trades); yrs=d["span"]/365.25
    if n==0:
        return {"trades":0,"ret":0,"cagr":0,"wr":0,"pf":0,"sharpe":0,"sortino":0,
                "maxdd":0,"exp":0,"sl":0,"tp":0,"trades_list":[]}
    pnl=np.array([t["pnl"] for t in trades])
    wins=pnl[pnl>0]; losses=pnl[pnl<=0]
    wr=len(wins)/n*100
    pf=wins.sum()/abs(losses.sum()) if len(losses) else float("inf")
    ret=(cap/CAP0-1)*100
    cagr=((cap/CAP0)**(1/yrs)-1)*100 if yrs>0 and cap>0 else -100
    tpy=n/yrs if yrs>0 else n
    sharpe=pnl.mean()/pnl.std()*np.sqrt(tpy) if pnl.std()>0 else 0
    downside=pnl[pnl<0]
    dd_std=downside.std() if len(downside)>1 else (abs(downside.mean()) if len(downside) else 0)
    sortino=pnl.mean()/dd_std*np.sqrt(tpy) if dd_std>0 else 0
    eq=CAP0+np.cumsum(pnl); peak=np.maximum.accumulate(eq); maxdd=((eq-peak)/peak).min()*100
    return {"trades":n,"ret":ret,"cagr":cagr,"wr":wr,"pf":pf,"sharpe":sharpe,
            "sortino":sortino,"maxdd":maxdd,"exp":pnl.mean(),
            "sl":int((np.array([t["reason"] for t in trades])=="SL").sum()),
            "tp":int((np.array([t["reason"] for t in trades])=="TP").sum()),
            "trades_list":trades}


def buy_hold(d):
    c=d["close"]; yrs=d["span"]/365.25
    units=(CAP0*(1-FEE))/c[0]; eq=units*c
    ret=(eq[-1]/CAP0-1)*100; cagr=((eq[-1]/CAP0)**(1/yrs)-1)*100 if yrs>0 else 0
    peak=np.maximum.accumulate(eq); maxdd=((eq-peak)/peak).min()*100
    rets=np.diff(c)/c[:-1]; sharpe=rets.mean()/rets.std()*np.sqrt(24*365) if rets.std()>0 else 0
    dn=rets[rets<0]; sortino=rets.mean()/dn.std()*np.sqrt(24*365) if len(dn)>1 and dn.std()>0 else 0
    return {"ret":ret,"cagr":cagr,"maxdd":maxdd,"sharpe":sharpe,"sortino":sortino}


def ic_analysis(d):
    """Information Coefficient: Spearman(signal, forward H-bar return)."""
    recs=d["recs"]; close=d["close"]; n=len(close)
    rows=[r for r in recs if r["i"]+HORIZON < n]
    fwd=np.array([(close[r["i"]+HORIZON]-close[r["i"]])/close[r["i"]] for r in rows])
    out={}
    # overall signed-signal IC (raw centered score predicts forward return)
    sig=np.array([r["raw"] for r in rows])
    ic,p=stats.spearmanr(sig, fwd)
    out["__signal__"]=(ic, p, len(rows))
    # per-component IC
    for k in COMPONENTS:
        col=np.array([r[f"c_{k}"] for r in rows])
        if col.std()==0:
            out[k]=(0.0,1.0,0); continue
        ic,p=stats.spearmanr(col, fwd)
        out[k]=(ic,p,len(rows))
    # mutual information (nonlinear) per component
    X=np.array([[r[f"c_{k}"] for k in COMPONENTS] for r in rows])
    try:
        mi=mutual_info_regression(X, fwd, random_state=0)
    except Exception:
        mi=np.zeros(len(COMPONENTS))
    out["__mi__"]=dict(zip(COMPONENTS, mi))
    out["__fwd__"]=fwd; out["__rows__"]=rows
    return out


def score_buckets(d):
    """Forward-return by confidence bucket — does higher conviction pay?"""
    recs=d["recs"]; close=d["close"]; n=len(close)
    buckets={"5.5-6.0":[], "6.0-6.5":[], "6.5-7.0":[], "7.0-7.5":[], "7.5+":[]}
    for r in recs:
        if r["i"]+HORIZON>=n or r["dir"] not in ("long","short"): continue
        if not passes_conviction_rec(r): continue
        fwd=(close[r["i"]+HORIZON]-close[r["i"]])/close[r["i"]]
        # express forward return in the TRADE's direction (long=+, short=-)
        signed = fwd if r["dir"]=="long" else -fwd
        s=r["score"] if r["dir"]=="long" else 10-r["score"]  # map short score to conviction scale
        b=("7.5+" if s>=7.5 else "7.0-7.5" if s>=7.0 else "6.5-7.0" if s>=6.5
           else "6.0-6.5" if s>=6.0 else "5.5-6.0")
        buckets[b].append(signed*100)
    return buckets


def regime_attr(d):
    """Per-regime forward return in trade direction + loss attribution from sims."""
    return None  # handled in report via trade lists


# ── Report ────────────────────────────────────────────────────────────────────

def fmt(m, bh=None):
    return (f"ret {m['ret']:+6.1f}%  CAGR {m['cagr']:+6.1f}%  WR {m['wr']:4.1f}%  "
            f"PF {m['pf']:4.2f}  Sharpe {m['sharpe']:+5.2f}  Sortino {m['sortino']:+5.2f}  "
            f"MaxDD {m['maxdd']:6.1f}%  Exp ${m['exp']:+6.2f}  n={m['trades']}")


def main():
    print(f"EDGE VALIDATION | {DAYS}d req | fee {FEE*100}%/side + slip {SLIP*100}%/side | "
          f"IC horizon {HORIZON} bars\n")
    all_ic={}; all_buckets={k:[] for k in ["5.5-6.0","6.0-6.5","6.5-7.0","7.0-7.5","7.5+"]}
    loss_attr={}
    for sym in SYMBOLS:
        d=collect(sym)
        if d is None: print(f"{sym}: no data"); continue
        print(f"{'='*92}\n{sym}  ({d['span']}d, {len(d['recs'])} bars)\n{'='*92}")
        bh=buy_hold(d)
        print(f"  BUY & HOLD      : ret {bh['ret']:+6.1f}%  CAGR {bh['cagr']:+6.1f}%  "
              f"Sharpe {bh['sharpe']:+5.2f}  Sortino {bh['sortino']:+5.2f}  MaxDD {bh['maxdd']:6.1f}%")
        for mode in ("long","short","both"):
            m=simulate(d, mode)
            print(f"  {mode.upper():15s}: {fmt(m)}")
            # loss attribution
            for t in m["trades_list"]:
                if t["pnl"]<=0 and t["reason"]!="eot":
                    key=(t["regime"], t["daily"])
                    loss_attr[key]=loss_attr.get(key,0)+1
        # IC
        ic=ic_analysis(d); all_ic[sym]=ic
        sic,sp,nn=ic["__signal__"]
        sig_flag="✅ predictive" if sp<0.05 and abs(sic)>0.02 else "❌ ~random"
        print(f"\n  SIGNAL IC (raw score vs {HORIZON}-bar fwd ret): "
              f"IC={sic:+.4f}  p={sp:.3f}  n={nn}   {sig_flag}")
        # buckets
        bk=score_buckets(d)
        print(f"  Confidence buckets (mean signed {HORIZON}-bar fwd ret %, in trade direction):")
        for b,v in bk.items():
            all_buckets[b]+=v
            if v: print(f"    {b:9s}: {np.mean(v):+.3f}%  (n={len(v)})")
            else: print(f"    {b:9s}: —")
        print()

    # ── Cross-symbol component IC ranking ─────────────────────────────────────
    print(f"{'#'*92}\n  COMPONENT PREDICTIVE VALUE  (mean Spearman IC across BTC/ETH/SOL)\n{'#'*92}")
    rows=[]
    for k in COMPONENTS:
        ics=[all_ic[s][k][0] for s in all_ic if k in all_ic[s]]
        ps =[all_ic[s][k][1] for s in all_ic if k in all_ic[s]]
        mis=[all_ic[s]["__mi__"].get(k,0) for s in all_ic]
        rows.append((k, np.mean(ics), np.mean(ps), np.mean(mis)))
    rows.sort(key=lambda x:-x[1])
    print(f"  {'component':16s} {'mean IC':>9s} {'mean p':>8s} {'mean MI':>9s}  verdict")
    for k,ic,p,mi in rows:
        v=("POSITIVE edge" if ic>0.02 and p<0.10 else
           "NEGATIVE (anti-signal)" if ic<-0.02 and p<0.10 else "no edge (~random)")
        print(f"  {k:16s} {ic:+9.4f} {p:8.3f} {mi:9.4f}  {v}")
    sigics=[all_ic[s]["__signal__"][0] for s in all_ic]
    sigps =[all_ic[s]["__signal__"][1] for s in all_ic]
    print(f"\n  COMPOSITE SIGNAL: mean IC={np.mean(sigics):+.4f}  mean p={np.mean(sigps):.3f}")

    # ── Confidence monotonicity ───────────────────────────────────────────────
    print(f"\n{'#'*92}\n  DOES HIGHER CONFIDENCE PAY?  (pooled mean signed fwd ret per bucket)\n{'#'*92}")
    seq=[]
    for b in ["5.5-6.0","6.0-6.5","6.5-7.0","7.0-7.5","7.5+"]:
        v=all_buckets[b]
        if v: print(f"  {b:9s}: {np.mean(v):+.3f}%  (n={len(v)})"); seq.append(np.mean(v))
        else: print(f"  {b:9s}: —")
    mono = all(seq[i]<=seq[i+1] for i in range(len(seq)-1)) if len(seq)>1 else False
    print(f"  Monotonic increasing (more conviction → more return)? {'YES' if mono else 'NO'}")

    # ── Loss attribution ──────────────────────────────────────────────────────
    print(f"\n{'#'*92}\n  LOSS ATTRIBUTION  (losing trades by [1h regime, daily gate])\n{'#'*92}")
    for (reg,daily),c in sorted(loss_attr.items(), key=lambda x:-x[1])[:10]:
        print(f"  {c:4d}  regime={reg:15s} daily={daily}")


if __name__=="__main__":
    main()
