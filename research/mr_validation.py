"""
Mean-reversion edge validation — same rigor and cost model as edge_validation.py.
Proves (or kills) the contrarian 1h rewrite BEFORE any live wiring.

Gate to advance: (1) composite IC positive & significant (p<0.05),
                 (2) net-positive after fees+slippage on BTC/ETH/SOL,
                 (3) higher-confidence buckets pay (monotonic-ish).
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from scipy import stats

import config
from exchange.market_data import fetch_ohlcv
from core.indicators import enrich
from core.mean_reversion import analyse_mr

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
DAYS = 180
FEE = config.FEE_RATE_PCT / 100.0
SLIP = 0.05 / 100.0
RISK = config.RISK_PER_TRADE_PCT / 100.0
CAP0 = 1000.0
HORIZON = 6
COMPONENTS = ["zscore", "rsi_revert", "bb_revert", "stoch", "volume"]


def passes_mr(score, direction):
    thr = config.MR_MIN_SCORE
    if direction == "long":  return score >= thr
    if direction == "short": return score <= (10 - thr)
    return False


def collect(symbol):
    df1h = fetch_ohlcv(symbol, "1h", limit=DAYS*24+300)
    if df1h.empty or len(df1h) < 300:
        return None
    df1h = enrich(df1h)
    span = (df1h.index[-1]-df1h.index[0]).days
    close = df1h["close"].values; high = df1h["high"].values; low = df1h["low"].values
    recs = []
    for i in range(100, len(df1h)):
        sl = df1h.iloc[max(0, i-300):i]
        sig = analyse_mr(symbol, sl)
        recs.append({"i": i, "price": close[i], "atr": sig.atr, "dir": sig.direction,
                     "score": sig.score, "raw": sum(sig.components.values()),
                     **{f"c_{k}": sig.components.get(k, 0.0) for k in COMPONENTS}})
    return {"symbol": symbol, "span": span, "recs": recs,
            "close": close, "high": high, "low": low}


def simulate(d):
    recs = d["recs"]; high = d["high"]; low = d["low"]; close = d["close"]; n = len(close)
    cap = CAP0; in_trade = False
    entry=stop=target=qty=0.0; side=""; ei=0; consec=0
    trades = []; by_entry = {}
    by_i = {r["i"]: r for r in recs}
    i = recs[0]["i"]
    while i < n:
        r = by_i.get(i)
        if in_trade:
            hit_sl = (side=="long" and low[i]<=stop) or (side=="short" and high[i]>=stop)
            hit_tp = (side=="long" and high[i]>=target) or (side=="short" and low[i]<=target)
            if hit_sl or hit_tp:
                ep = target if hit_tp else stop
                ep = ep*(1-SLIP) if side=="long" else ep*(1+SLIP)
                gross = (ep-entry)*qty if side=="long" else (entry-ep)*qty
                pnl = gross-(entry*qty+ep*qty)*FEE; cap += pnl
                trades.append({**by_entry, "pnl": pnl, "reason": "TP" if hit_tp else "SL"})
                in_trade=False; consec = consec+1 if pnl<0 else 0
        if not in_trade and r is not None:
            if consec >= 2: consec -= 1; i += 1; continue
            if r["dir"] in ("long","short") and passes_mr(r["score"], r["dir"]) and r["atr"]>0:
                price=r["price"]; atr=r["atr"]
                eprice = price*(1+SLIP) if r["dir"]=="long" else price*(1-SLIP)
                if r["dir"]=="long":
                    fs=eprice-config.MR_ATR_STOP*atr; ft=eprice+config.MR_ATR_TARGET*atr
                else:
                    fs=eprice+config.MR_ATR_STOP*atr; ft=eprice-config.MR_ATR_TARGET*atr
                rpu=abs(eprice-fs)
                if rpu>0:
                    q=cap*RISK/rpu
                    g=abs(ft-eprice)*q; f=(eprice*q+ft*q)*FEE
                    if g-f>=config.MIN_NET_PROFIT_USD and g>=f*config.MIN_WIN_FEE_MULTIPLE:
                        entry,stop,target,qty,side=eprice,fs,ft,q,r["dir"]; ei=i
                        by_entry={"side":r["dir"],"score":r["score"]}
                        in_trade=True
        i += 1
    if in_trade:
        lp=close[-1]; gross=(lp-entry)*qty if side=="long" else (entry-lp)*qty
        cap += gross-(entry*qty+lp*qty)*FEE
        trades.append({**by_entry,"pnl":gross-(entry*qty+lp*qty)*FEE,"reason":"eot"})
    return metrics(d, trades, cap)


def metrics(d, trades, cap):
    n=len(trades); yrs=d["span"]/365.25
    if n==0:
        return {"trades":0,"ret":0,"cagr":0,"wr":0,"pf":0,"sharpe":0,"sortino":0,"maxdd":0,"exp":0,"sl":0,"tp":0}
    pnl=np.array([t["pnl"] for t in trades])
    wins=pnl[pnl>0]; losses=pnl[pnl<=0]
    wr=len(wins)/n*100; pf=wins.sum()/abs(losses.sum()) if len(losses) else float("inf")
    ret=(cap/CAP0-1)*100; cagr=((cap/CAP0)**(1/yrs)-1)*100 if yrs>0 and cap>0 else -100
    tpy=n/yrs if yrs>0 else n
    sharpe=pnl.mean()/pnl.std()*np.sqrt(tpy) if pnl.std()>0 else 0
    dn=pnl[pnl<0]; dstd=dn.std() if len(dn)>1 else (abs(dn.mean()) if len(dn) else 0)
    sortino=pnl.mean()/dstd*np.sqrt(tpy) if dstd>0 else 0
    eq=CAP0+np.cumsum(pnl); peak=np.maximum.accumulate(eq); maxdd=((eq-peak)/peak).min()*100
    return {"trades":n,"ret":ret,"cagr":cagr,"wr":wr,"pf":pf,"sharpe":sharpe,"sortino":sortino,
            "maxdd":maxdd,"exp":pnl.mean(),
            "sl":int((np.array([t["reason"] for t in trades])=="SL").sum()),
            "tp":int((np.array([t["reason"] for t in trades])=="TP").sum())}


def ic_analysis(d):
    recs=d["recs"]; close=d["close"]; n=len(close)
    rows=[r for r in recs if r["i"]+HORIZON < n]
    fwd=np.array([(close[r["i"]+HORIZON]-close[r["i"]])/close[r["i"]] for r in rows])
    out={}
    sig=np.array([r["raw"] for r in rows])
    out["__signal__"]=stats.spearmanr(sig, fwd)
    for k in COMPONENTS:
        col=np.array([r[f"c_{k}"] for r in rows])
        out[k]=(0.0,1.0) if col.std()==0 else tuple(stats.spearmanr(col, fwd))
    return out


def buckets(d):
    recs=d["recs"]; close=d["close"]; n=len(close)
    bk={"6.0-6.5":[], "6.5-7.0":[], "7.0-7.5":[], "7.5-8.0":[], "8.0+":[]}
    for r in recs:
        if r["i"]+HORIZON>=n or r["dir"] not in ("long","short"): continue
        if not passes_mr(r["score"], r["dir"]): continue
        fwd=(close[r["i"]+HORIZON]-close[r["i"]])/close[r["i"]]
        signed=(fwd if r["dir"]=="long" else -fwd)*100
        s=r["score"] if r["dir"]=="long" else 10-r["score"]
        b=("8.0+" if s>=8 else "7.5-8.0" if s>=7.5 else "7.0-7.5" if s>=7
           else "6.5-7.0" if s>=6.5 else "6.0-6.5")
        bk[b].append(signed)
    return bk


def main():
    print(f"MEAN-REVERSION VALIDATION | {DAYS}d | fee {FEE*100}%/side + slip {SLIP*100}%/side | "
          f"IC horizon {HORIZON} | target {config.MR_ATR_TARGET}ATR stop {config.MR_ATR_STOP}ATR\n")
    all_ic={}; all_bk={k:[] for k in ["6.0-6.5","6.5-7.0","7.0-7.5","7.5-8.0","8.0+"]}
    for sym in SYMBOLS:
        d=collect(sym)
        if d is None: print(f"{sym}: no data"); continue
        m=simulate(d); ic=ic_analysis(d); all_ic[sym]=ic
        sic,sp=ic["__signal__"]
        flag="✅ predictive" if sp<0.05 and sic>0.02 else "❌ not(+)significant"
        print(f"{'='*90}\n{sym}  ({d['span']}d, {len(d['recs'])} bars)\n{'='*90}")
        print(f"  BACKTEST: ret {m['ret']:+6.1f}%  CAGR {m['cagr']:+6.1f}%  WR {m['wr']:4.1f}%  "
              f"PF {m['pf']:4.2f}  Sharpe {m['sharpe']:+5.2f}  Sortino {m['sortino']:+5.2f}  "
              f"MaxDD {m['maxdd']:6.1f}%  Exp ${m['exp']:+5.2f}  n={m['trades']} (TP{m['tp']}/SL{m['sl']})")
        print(f"  SIGNAL IC vs {HORIZON}-bar fwd ret: IC={sic:+.4f}  p={sp:.3f}   {flag}")
        bk=buckets(d)
        print(f"  Confidence buckets (mean signed fwd ret %):")
        for b,v in bk.items():
            all_bk[b]+=v
            print(f"    {b:9s}: {np.mean(v):+.3f}%  (n={len(v)})" if v else f"    {b:9s}: —")
        print()
    # component IC ranking
    print(f"{'#'*90}\n  MR COMPONENT PREDICTIVE VALUE  (mean Spearman IC across BTC/ETH/SOL)\n{'#'*90}")
    for k in COMPONENTS:
        ics=[all_ic[s][k][0] for s in all_ic]; ps=[all_ic[s][k][1] for s in all_ic]
        v=("POSITIVE edge" if np.mean(ics)>0.02 and np.mean(ps)<0.10 else
           "NEGATIVE" if np.mean(ics)<-0.02 and np.mean(ps)<0.10 else "no edge")
        print(f"  {k:12s} IC={np.mean(ics):+.4f}  p={np.mean(ps):.3f}  {v}")
    sigi=[all_ic[s]['__signal__'][0] for s in all_ic]; sigp=[all_ic[s]['__signal__'][1] for s in all_ic]
    print(f"\n  COMPOSITE MR SIGNAL: mean IC={np.mean(sigi):+.4f}  mean p={np.mean(sigp):.3f}")
    print(f"\n{'#'*90}\n  DOES HIGHER MR CONFIDENCE PAY?\n{'#'*90}")
    seq=[]
    for b in ["6.0-6.5","6.5-7.0","7.0-7.5","7.5-8.0","8.0+"]:
        v=all_bk[b]
        if v: print(f"  {b:9s}: {np.mean(v):+.3f}%  (n={len(v)})"); seq.append(np.mean(v))
        else: print(f"  {b:9s}: —")
    mono=all(seq[i]<=seq[i+1] for i in range(len(seq)-1)) if len(seq)>1 else False
    print(f"  Monotonic (more conviction → more return)? {'YES' if mono else 'NO'}")


if __name__ == "__main__":
    main()
