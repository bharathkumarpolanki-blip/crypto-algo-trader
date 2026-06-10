"""
OPEN-INTEREST / PRICE-DIVERGENCE STUDY — real OI data from OKX perps (public).

Classic market-structure quadrants (decided at close[T]):
  Price↑ + OI↑  = new longs (trend confirmation, bullish continuation)
  Price↑ + OI↓  = short covering (weak rally → expect reversal DOWN)
  Price↓ + OI↑  = new shorts (trend confirmation, bearish continuation)
  Price↓ + OI↓  = long liquidation (capitulation → expect reversal UP)

Measures forward returns (strict T+1) at 1/3/7/30d per quadrant + significance.
DATA NOTE: OKX public OI history caps at ~99 days → underpowered, one regime.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
import ccxt

COINS=["BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT"]
CACHE="/tmp/okx_funding"
os.makedirs(CACHE, exist_ok=True)
def e(): return ccxt.okx({"enableRateLimit":True,"timeout":20000})


def fetch_oi(ex, sym):
    f=f"{CACHE}/oi_{sym.replace('/','_').replace(':','_')}.csv"
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["dt"]).set_index("dt")["oi"]
    rows=[]; cursor=ex.milliseconds()-110*86400000
    while True:
        try: b=ex.fetch_open_interest_history(sym,"1d",since=cursor,limit=100)
        except Exception as ex_: print(f"  {sym} OI err: {str(ex_)[:70]}"); break
        if not b: break
        rows+=b; last=b[-1]["timestamp"]
        if last<=cursor: break
        cursor=last+86400000
        if cursor>ex.milliseconds(): break
        time.sleep(ex.rateLimit/1000)
    if not rows: return None
    df=pd.DataFrame([(r["timestamp"], r.get("openInterestValue")) for r in rows],
                    columns=["ts","oi"]).dropna().drop_duplicates("ts")
    df["dt"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    s=df.set_index("dt")["oi"].sort_index(); s.to_csv(f,header=True); return s


def fetch_price(ex, sym):
    f=f"{CACHE}/px_{sym.replace('/','_').replace(':','_')}.csv"
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["dt"]).set_index("dt")["close"]
    rows=[]; cursor=ex.milliseconds()-110*86400000
    while True:
        try: o=ex.fetch_ohlcv(sym,"1d",since=cursor,limit=200)
        except Exception: break
        if not o: break
        rows+=o; last=o[-1][0]
        if last<=cursor: break
        cursor=last+86400000
        if cursor>ex.milliseconds(): break
        time.sleep(ex.rateLimit/1000)
    if not rows: return None
    df=pd.DataFrame(rows,columns=["ts","o","h","l","close","v"]).drop_duplicates("ts")
    df["dt"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    s=df.set_index("dt")["close"].sort_index(); s.to_csv(f,header=True); return s


def build():
    ex=e(); rows=[]
    for sym in COINS:
        oi=fetch_oi(ex,sym); px=fetch_price(ex,sym)
        if oi is None or px is None: print(f"  {sym}: missing"); continue
        oid=oi.resample("1D").last(); c=px.resample("1D").last()
        df=pd.DataFrame({"oi":oid,"close":c}).dropna()
        if len(df)<60: print(f"  {sym}: only {len(df)}d"); continue
        df["sym"]=sym; df["ret"]=df["close"].pct_change(); df["oichg"]=df["oi"].pct_change()
        rows.append(df); print(f"  {sym}: {len(df)} days")
    return pd.concat(rows) if rows else None


def quadrant(r,o):
    if r>0 and o>0: return "Price↑ OI↑ (new longs)"
    if r>0 and o<0: return "Price↑ OI↓ (short cover)"
    if r<0 and o>0: return "Price↓ OI↑ (new shorts)"
    if r<0 and o<0: return "Price↓ OI↓ (long liq.)"
    return None


def main():
    print("OPEN-INTEREST / PRICE-DIVERGENCE STUDY — OKX perps (real OI)\n")
    D=build()
    if D is None: print("NO DATA."); return
    D["quad"]=[quadrant(r,o) for r,o in zip(D["ret"],D["oichg"])]
    print(f"\nPooled: {len(D)} coin-days, {D.index.min().date()} → {D.index.max().date()}")
    print("(⚠ ~99-day OKX cap — single regime, underpowered. Treat as exploratory.)\n")

    for H in [1,3,7,30]:
        D[f"fwd{H}"]=D.groupby("sym")["close"].transform(lambda c: c.shift(-(H+1))/c.shift(-1)-1)

    quads=["Price↑ OI↑ (new longs)","Price↑ OI↓ (short cover)",
           "Price↓ OI↑ (new shorts)","Price↓ OI↓ (long liq.)"]
    for H in [1,3,7,30]:
        print(f"### Forward {H}d return by quadrant (T+1)\n")
        print(f"| {'Quadrant':26s} | {'n':>4s} | {'mean fwd':>9s} | {'%pos':>5s} | {'t-test p':>8s} | reading")
        print("|"+"-"*28+"|"+"-"*6+"|"+"-"*11+"|"+"-"*7+"|"+"-"*10+"|"+"-"*22)
        for q in quads:
            f=D[D["quad"]==q][f"fwd{H}"].dropna()
            if len(f)<10: print(f"| {q:26s} | {len(f):4d} | (too few) |"); continue
            t,p=stats.ttest_1samp(f,0)
            read = "✅ predicts" if p<0.05 else "noise"
            print(f"| {q:26s} | {len(f):4d} | {f.mean()*100:+8.2f}% | {(f>0).mean()*100:4.0f}% | {p:8.3f} | {read}")
        print()

    # divergence focus: do the 'reversal' quadrants actually reverse?
    print("### DIVERGENCE TEST — do weak-structure quadrants reverse? (3d fwd)\n")
    for q,expect in [("Price↑ OI↓ (short cover)","expect DOWN (reversal)"),
                     ("Price↓ OI↓ (long liq.)","expect UP (bounce)")]:
        f=D[D["quad"]==q]["fwd3"].dropna()
        if len(f)<10: continue
        t,p=stats.ttest_1samp(f,0)
        got = "DOWN" if f.mean()<0 else "UP"
        print(f"  {q}: mean 3d fwd {f.mean()*100:+.2f}% ({got}), {expect}, p={p:.3f}, n={len(f)}")


if __name__=="__main__":
    main()
