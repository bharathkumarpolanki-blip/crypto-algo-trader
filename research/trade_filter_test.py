"""
"FIX THE LOSING TRADES" TEST — does filtering out the losers generalize, or is it
curve-fitting? Generate the 1h engine's trades with entry features, split by time
(in-sample / out-of-sample), learn a loser-avoidance filter on IN-SAMPLE trades,
FREEZE it, and apply to OUT-OF-SAMPLE trades.

If the win-rate/expectancy gain only appears in-sample → overfitting (confirmed).
If it survives OOS → a real, generalizable improvement.

Read-only. Strict: the filter never sees OOS data when it's built.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier

import config
from exchange.market_data import fetch_ohlcv
from core.indicators import enrich
from core.strategies import analyse, score_market_regime, passes_conviction

COINS = ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD","LTC/USD"]
DAYS = 180
FEE = config.FEE_RATE_PCT/100.0
RISK = config.RISK_PER_TRADE_PCT/100.0
CAP = 1000.0
FEATS = ["score","conviction","atr_pct","rr","regime","trend6h","daily_num",
         "hour","side_num","roc","adx","rsi","bb_pct","vol_ratio"]


def num_trend(t): return {"bull":1,"neutral":0,"bear":-1,"unknown":0}.get(t,0)


def gen_trades():
    trades=[]
    for sym in COINS:
        d1=fetch_ohlcv(sym,"1h",limit=DAYS*24+300)
        d6=fetch_ohlcv(sym,config.TF_TREND,limit=DAYS*6+100)
        dd=fetch_ohlcv(sym,"1d",limit=config.DAILY_GATE_SMA_PERIOD+400)
        if d1.empty or len(d1)<300: continue
        d1=enrich(d1); d6=enrich(d6) if not d6.empty else pd.DataFrame()
        in_t=False; entry=stop=target=qty=0.0; side=""; feat={}; consec=0
        for i in range(100,len(d1)):
            cur=d1.iloc[i]; price=cur["close"]; ts=d1.index[i]
            s6=d6[d6.index<=ts] if not d6.empty else pd.DataFrame()
            if len(s6)<60: s6=pd.DataFrame()
            sd=dd[dd.index<=ts] if not dd.empty else None
            if sd is not None and len(sd)<config.DAILY_GATE_SMA_PERIOD: sd=None
            if in_t:
                hit_sl=(side=="long" and cur["low"]<=stop) or (side=="short" and cur["high"]>=stop)
                hit_tp=(side=="long" and cur["high"]>=target) or (side=="short" and cur["low"]<=target)
                if hit_sl or hit_tp:
                    ep=target if hit_tp else stop
                    gross=(ep-entry)*qty if side=="long" else (entry-ep)*qty
                    pnl=gross-(entry*qty+ep*qty)*FEE
                    trades.append({**feat,"pnl":pnl,"win":int(pnl>0),"time":ts})
                    in_t=False; consec=consec+1 if pnl<0 else 0
            if not in_t:
                if consec>=2: consec-=1; continue
                sl1=d1.iloc[max(0,i-300):i]
                sig=analyse(sym,sl1,df_trend=s6 if len(s6)>=60 else None,
                            include_sentiment=False,include_ml=False,df_daily=sd)
                if sig.direction in ("long","short") and passes_conviction(sig) and sig.atr>0:
                    atr=sig.atr
                    if sig.direction=="long":
                        fs=price-config.ATR_STOP_MULTIPLIER*atr; ft=price+config.ATR_TARGET_MULTIPLIER*atr
                    else:
                        fs=price+config.ATR_STOP_MULTIPLIER*atr; ft=price-config.ATR_TARGET_MULTIPLIER*atr
                    rr=abs(ft-price)/abs(fs-price)
                    if rr>=2.0:
                        rpu=abs(price-fs)
                        if rpu>0:
                            qy=CAP*RISK/rpu; g=abs(ft-price)*qy; f=(price*qy+ft*qy)*FEE
                            if g-f>=config.MIN_NET_PROFIT_USD and g>=f*config.MIN_WIN_FEE_MULTIPLE:
                                entry,stop,target,qty,side=price,fs,ft,qy,sig.direction
                                feat={
                                    "score":sig.score,"conviction":sig.conviction,
                                    "atr_pct":atr/price*100,"rr":rr,
                                    "regime":score_market_regime(sl1),
                                    "trend6h":num_trend(sig.note.split()[0].split(":")[-1] if ":" in sig.note else "neutral"),
                                    "daily_num":num_trend(sig.daily_trend),
                                    "hour":ts.hour,"side_num":1 if side=="long" else -1,
                                    "roc":float(cur.get("roc",0) or 0),"adx":float(cur.get("adx",0) or 0),
                                    "rsi":float(cur.get("rsi",50) or 50),"bb_pct":float(cur.get("bb_pct",0.5) or 0.5),
                                    "vol_ratio":float(cur.get("vol_ratio",1) or 1),
                                }
                                in_t=True
    return pd.DataFrame(trades)


def stats(df,label):
    n=len(df); w=df["win"].sum(); wr=w/n*100 if n else 0
    exp=df["pnl"].mean() if n else 0; tot=df["pnl"].sum()
    return f"{label:28s} n={n:4d}  WR={wr:4.1f}%  exp=${exp:+6.2f}  total=${tot:+8.1f}"


def main():
    print(f"Generating 1h-engine trades across {len(COINS)} coins ({DAYS}d)…")
    T=gen_trades().sort_values("time").reset_index(drop=True)
    if len(T)<60:
        print(f"Only {len(T)} trades — too few."); return
    cut=int(len(T)*0.70)
    tr=T.iloc[:cut]; te=T.iloc[cut:]
    print(f"\n{len(T)} trades total | train {len(tr)} (≤{tr['time'].iloc[-1].date()}) "
          f"| test {len(te)} (≥{te['time'].iloc[0].date()})\n")

    print("### BASELINE (no filter)")
    print("  "+stats(tr,"TRAIN all trades"))
    print("  "+stats(te,"TEST  all trades"))

    # ── Build loser-avoidance filter on TRAIN only (charitable: shallow tree) ──
    clf=DecisionTreeClassifier(max_depth=3,min_samples_leaf=15,random_state=0)
    clf.fit(tr[FEATS].fillna(0), tr["win"])
    tr_keep=tr[clf.predict(tr[FEATS].fillna(0))==1]
    te_keep=te[clf.predict(te[FEATS].fillna(0))==1]    # FROZEN filter on OOS

    print("\n### AFTER LOSER-AVOIDANCE FILTER (learned on train, frozen)")
    print("  "+stats(tr_keep,"TRAIN kept (in-sample)"))
    print("  "+stats(te_keep,"TEST  kept (OUT-OF-SAMPLE)"))

    # ── Verdict ───────────────────────────────────────────────────────────────
    base_te_exp=te["pnl"].mean(); filt_te_exp=te_keep["pnl"].mean() if len(te_keep) else 0
    base_tr_exp=tr["pnl"].mean(); filt_tr_exp=tr_keep["pnl"].mean() if len(tr_keep) else 0
    print(f"\n### VERDICT")
    print(f"  In-sample expectancy lift:  ${base_tr_exp:+.2f} → ${filt_tr_exp:+.2f}  "
          f"(Δ ${filt_tr_exp-base_tr_exp:+.2f})")
    print(f"  OUT-OF-SAMPLE lift:         ${base_te_exp:+.2f} → ${filt_te_exp:+.2f}  "
          f"(Δ ${filt_te_exp-base_te_exp:+.2f})")
    kept_pct=len(te_keep)/len(te)*100 if len(te) else 0
    print(f"  Filter kept {kept_pct:.0f}% of OOS trades")
    if filt_te_exp>0 and filt_te_exp>base_te_exp+1:
        print(f"  → ✅ Filter IMPROVED out-of-sample — possibly real, investigate further.")
    else:
        print(f"  → ❌ Filter did NOT generalize. In-sample gain was curve-fitting.")
        print(f"     (The losers and winners are not separable by entry features OOS.)")


if __name__=="__main__":
    main()
