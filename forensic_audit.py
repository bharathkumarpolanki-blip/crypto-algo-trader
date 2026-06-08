"""
Forensic audit harness — instruments the REAL signal engine + backtest loop to
answer the long/short, regime, stop-loss and expectancy questions with actual
numbers (not assumptions). Read-only: places no orders, changes no config.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

import config
from exchange.market_data import fetch_ohlcv
from core.indicators import enrich
from core.strategies import analyse, trend_direction_4h, score_market_regime, passes_conviction

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
DAYS = 180
FEE = config.FEE_RATE_PCT / 100.0
RISK = config.RISK_PER_TRADE_PCT / 100.0
MIN_RR = 2.0
CAP0 = 1000.0


def regime_label(df):
    r = score_market_regime(df)
    if r >= 1.5:  return "strong_bull"
    if r >= 0.5:  return "bull"
    if r <= -3.0: return "confirmed_bear"
    if r <= -1.0: return "below_200ema"
    return "neutral"


def audit(symbol):
    df1h = fetch_ohlcv(symbol, "1h", limit=DAYS*24+300)
    df6h = fetch_ohlcv(symbol, config.TF_TREND, limit=DAYS*6+100)
    df1d = fetch_ohlcv(symbol, "1d", limit=config.DAILY_GATE_SMA_PERIOD+400)
    if df1h.empty or len(df1h) < 200:
        print(f"{symbol}: insufficient data"); return None
    df1h = enrich(df1h); df6h = enrich(df6h) if not df6h.empty else pd.DataFrame()
    span = (df1h.index[-1]-df1h.index[0]).days

    # Counters
    opp = {"long": 0, "short": 0, "neutral": 0}
    blocked_short_by_gate = 0
    raw_short_candidates = 0
    raw_long_candidates = 0
    regime_hist = {}
    trend_hist = {}

    trades = []
    cap = CAP0
    in_trade = False
    entry=stop=target=qty=0.0; side=""; t_regime=""; t_trend=""; t_score=0.0; t_atr=0.0
    consec = 0

    for i in range(100, len(df1h)):
        sl1 = df1h.iloc[max(0,i-300):i]
        cur = df1h.iloc[i]; price = cur["close"]
        ts = df1h.index[i]
        sl6 = df6h[df6h.index <= ts] if not df6h.empty else pd.DataFrame()
        if len(sl6) < 60: sl6 = pd.DataFrame()
        sl1d = df1d[df1d.index <= ts] if not df1d.empty else None
        if sl1d is not None and len(sl1d) < config.DAILY_GATE_SMA_PERIOD: sl1d = None

        # exit
        if in_trade:
            hit_sl = (side=="long" and cur["low"]<=stop) or (side=="short" and cur["high"]>=stop)
            hit_tp = (side=="long" and cur["high"]>=target) or (side=="short" and cur["low"]<=target)
            if hit_sl or hit_tp:
                ep = target if hit_tp else stop
                gross = (ep-entry)*qty if side=="long" else (entry-ep)*qty
                fees = (entry*qty+ep*qty)*FEE
                pnl = gross-fees; cap += pnl
                trades.append({"side":side,"reason":"TP" if hit_tp else "SL","pnl":pnl,
                               "regime":t_regime,"trend6h":t_trend,"score":t_score,
                               "atr":t_atr,"entry":entry,"stop":stop,"target":target,
                               "stop_dist_pct":abs(entry-stop)/entry*100})
                in_trade=False; consec = consec+1 if pnl<0 else 0

        if not in_trade:
            if consec>=2: consec-=1; continue
            sig = analyse(symbol, sl1, df_trend=sl6 if len(sl6)>=60 else None,
                          include_sentiment=False, include_ml=False, df_daily=sl1d)
            # tally opportunity direction (post-gate)
            opp[sig.direction] = opp.get(sig.direction,0)+1
            # tally raw pre-gate candidates
            raw = sum(sig.components.values())
            if sig.score >= config.MIN_SIGNAL_SCORE and raw>0: raw_long_candidates+=1
            if sig.score <= (10-config.MIN_SIGNAL_SCORE) and raw<0:
                raw_short_candidates+=1
                if sig.direction != "short": blocked_short_by_gate+=1
            tr = trend_direction_4h(sl6 if len(sl6)>=60 else None)
            trend_hist[tr] = trend_hist.get(tr,0)+1
            reg = regime_label(sl1)
            regime_hist[reg] = regime_hist.get(reg,0)+1

            if sig.direction in ("long","short") and passes_conviction(sig) and sig.atr>0:
                atr=sig.atr
                if sig.direction=="long":
                    fs=price-config.ATR_STOP_MULTIPLIER*atr; ft=price+config.ATR_TARGET_MULTIPLIER*atr
                else:
                    fs=price+config.ATR_STOP_MULTIPLIER*atr; ft=price-config.ATR_TARGET_MULTIPLIER*atr
                rr=abs(ft-price)/abs(fs-price)
                if rr>=MIN_RR:
                    rpu=abs(price-fs)
                    if rpu>0:
                        q=cap*RISK/rpu
                        gross_tp=abs(ft-price)*q; fee_tp=(price*q+ft*q)*FEE
                        if gross_tp-fee_tp>=config.MIN_NET_PROFIT_USD and gross_tp>=fee_tp*config.MIN_WIN_FEE_MULTIPLE:
                            entry,stop,target,qty,side=price,fs,ft,q,sig.direction
                            t_regime=regime_label(sl1); t_trend=tr
                            t_score=sig.score; t_atr=atr; in_trade=True

    return {"symbol":symbol,"span":span,"cap":cap,"trades":trades,"opp":opp,
            "raw_long":raw_long_candidates,"raw_short":raw_short_candidates,
            "blocked_short":blocked_short_by_gate,"regime_hist":regime_hist,
            "trend_hist":trend_hist}


def report(r):
    if not r: return
    t=r["trades"]; n=len(t)
    longs=[x for x in t if x["side"]=="long"]; shorts=[x for x in t if x["side"]=="short"]
    wins=[x for x in t if x["pnl"]>0]; losses=[x for x in t if x["pnl"]<=0]
    sls=[x for x in t if x["reason"]=="SL"]; tps=[x for x in t if x["reason"]=="TP"]
    aw=np.mean([x["pnl"] for x in wins]) if wins else 0
    al=np.mean([x["pnl"] for x in losses]) if losses else 0
    wr=len(wins)/n*100 if n else 0
    pf=sum(x["pnl"] for x in wins)/abs(sum(x["pnl"] for x in losses)) if losses else float("inf")
    exp=np.mean([x["pnl"] for x in t]) if n else 0
    print(f"\n{'='*66}\n  {r['symbol']}  ({r['span']} days tested)\n{'='*66}")
    print(f"  Final capital: ${r['cap']:.2f}  ({(r['cap']/CAP0-1)*100:+.1f}%)")
    print(f"  Trades: {n}  |  LONG {len(longs)}  SHORT {len(shorts)}")
    print(f"  Exits: TP {len(tps)}  SL {len(sls)}  ({len(sls)/n*100 if n else 0:.0f}% stopped out)")
    print(f"  Win rate: {wr:.1f}%   PF: {pf:.2f}   Expectancy/trade: ${exp:+.2f}")
    print(f"  Avg win ${aw:+.2f}   Avg loss ${al:+.2f}   "
          f"payoff {abs(aw/al) if al else 0:.2f}x")
    print(f"\n  Opportunity scan (per-candle direction AFTER gates):")
    print(f"    long {r['opp'].get('long',0)}  short {r['opp'].get('short',0)}  neutral {r['opp'].get('neutral',0)}")
    print(f"  Raw candidates BEFORE 1h/6h gate:  long {r['raw_long']}  short {r['raw_short']}")
    print(f"  SHORT candidates KILLED by gates:  {r['blocked_short']}")
    print(f"  6h trend distribution at scan: {r['trend_hist']}")
    print(f"  Regime distribution at scan:   {r['regime_hist']}")
    if sls:
        print(f"\n  Stop-loss forensics (all {len(sls)} SL hits):")
        print(f"    avg stop distance: {np.mean([x['stop_dist_pct'] for x in sls]):.2f}% "
              f"(= {config.ATR_STOP_MULTIPLIER}×ATR)")
        by_reg={}
        for x in sls: by_reg[x["regime"]]=by_reg.get(x["regime"],0)+1
        print(f"    SL by regime at entry: {by_reg}")
    return {"symbol":r["symbol"],"n":n,"wr":wr,"pf":pf,"exp":exp,"aw":aw,"al":al,
            "sl":len(sls),"tp":len(tps),"longs":len(longs),"shorts":len(shorts)}


if __name__=="__main__":
    print(f"Forensic audit | risk {RISK*100}% | stop {config.ATR_STOP_MULTIPLIER}×ATR | "
          f"target {config.ATR_TARGET_MULTIPLIER}×ATR | fee {FEE*100}%/side")
    agg=[]
    for s in SYMBOLS:
        agg.append(report(audit(s)))
    print(f"\n{'#'*66}\n  PORTFOLIO ROLLUP\n{'#'*66}")
    A=[a for a in agg if a]
    tn=sum(a["n"] for a in A); tl=sum(a["longs"] for a in A); ts=sum(a["shorts"] for a in A)
    tsl=sum(a["sl"] for a in A); ttp=sum(a["tp"] for a in A)
    print(f"  Total trades {tn}  |  LONG {tl}  SHORT {ts}  ({ts/tn*100 if tn else 0:.0f}% short)")
    print(f"  Exits: TP {ttp}  SL {tsl}  ({tsl/tn*100 if tn else 0:.0f}% stopped out)")
