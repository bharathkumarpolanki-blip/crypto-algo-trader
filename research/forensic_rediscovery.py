"""
FORENSIC RE-DISCOVERY — were any positive, significant signals discarded only
because of high frequency / short holding / fees? Re-test the qualifying signals
at 30/60/90-day horizons as real T+1 long/cash strategies.

Qualifying signals from the edge inventory (p<0.05, +IC, +MI, +quintile spread):
  • Trend Following     (close/SMA50 - 1)         IC +0.031 p=0.006 net +0.44%
  • Market Structure    (Donchian-50 position-0.5) IC +0.027 p=0.017 net +0.86%
(Mean Reversion was significant but NEGATIVE IC = its direct form fails; its
 inverse is just Trend, already covered.)

Strict T+1, fees 0.6%/side + 0.05% slip, long-only, equal-weight portfolio.
Lower frequency at longer horizons = less fee drag — the hypothesis under test.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from exchange.market_data import fetch_ohlcv

COINS=["BTC/USD","ETH/USD","SOL/USD","LINK/USD","XRP/USD"]
CAP=1000.0; FEE=0.006; SLIP=0.0005
_cache={}
def load(s):
    if s not in _cache:
        d=fetch_ohlcv(s,"1d",limit=2500); _cache[s]=d if (not d.empty and len(d)>260) else None
    return _cache[s]


def sig_trend(df):       return (df["close"]/df["close"].rolling(50).mean()-1)
def sig_structure(df):
    c=df["close"]; rng=(c.rolling(50).max()-c.rolling(50).min())
    return (c-c.rolling(50).min())/rng - 0.5
SIGNALS={"Trend Following":sig_trend, "Market Structure":sig_structure}


def position(df, sigfn, H):
    """Long(1)/cash(0) decided at close[T] (signal>0), re-evaluated every H days,
       held in between. Returned UN-shifted (caller shifts +1 for T+1)."""
    s=sigfn(df); idx=df.index
    pos=pd.Series(np.nan,index=idx)
    decide=idx[::H]
    for t in decide:
        pos[t]=1.0 if (s.get(t,np.nan)>0) else 0.0
    return pos.ffill().fillna(0.0)


def bt(df, sigfn, H, fee=FEE, slip=SLIP):
    ret=df["close"].pct_change().fillna(0)
    p=position(df,sigfn,H).shift(1).fillna(0.0)      # T+1
    turn=p.diff().abs().fillna(p.abs())
    daily=p*ret - turn*(fee+slip)
    return (1+daily).cumprod(), p, daily, ret


def portfolio(sigfn, H, start=None, end=None, fee=FEE, slip=SLIP):
    eqs=[]; idx=None; dailies=[]
    for s in COINS:
        df=load(s)
        if df is None: continue
        if start or end:
            df=df[(df.index>=start) if start else slice(None)]
            if end: df=df[df.index<end]
            if len(df)<60: continue
        _,p,daily,ret=bt(df,sigfn,H,fee,slip)
        dailies.append(daily); idx=daily.index if idx is None else idx.union(daily.index)
    if not dailies: return None
    D=pd.concat(dailies,axis=1).reindex(idx).fillna(0).mean(axis=1)
    eq=CAP*(1+D).cumprod()
    return eq, D


def stats_(eq, D):
    if eq is None or len(eq)<2 or eq.iloc[-1]<=0: return dict(cagr=-100,sharpe=0,pf=0,maxdd=0)
    yrs=(eq.index[-1]-eq.index[0]).days/365.25 or 1
    cagr=((eq.iloc[-1]/CAP)**(1/yrs)-1)*100
    dd=((eq-eq.cummax())/eq.cummax()).min()*100
    r=eq.pct_change().dropna(); sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else 0
    pos=D[D>0].sum(); neg=abs(D[D<0].sum()); pf=pos/neg if neg>0 else 0
    return dict(cagr=cagr,sharpe=sh,pf=pf,maxdd=dd)


def ic_at(sigfn, H):
    sigs=[]; fwds=[]
    for s in COINS:
        df=load(s)
        if df is None: continue
        sg=sigfn(df); c=df["close"]; fw=c.shift(-(H+1))/c.shift(-1)-1
        sigs.append(sg.values); fwds.append(fw.values)
    S=np.concatenate(sigs); F=np.concatenate(fwds); m=~(np.isnan(S)|np.isnan(F))
    ic,p=stats.spearmanr(S[m],F[m])
    g=( pd.Series(F[m])[pd.Series(S[m])>=pd.Series(S[m]).quantile(0.8)].mean()
       -pd.Series(F[m])[pd.Series(S[m])<=pd.Series(S[m]).quantile(0.2)].mean() )*100
    return ic,p,g


def perm_p(eq, D):
    base=stats_(eq,D)["sharpe"]; rng=np.random.default_rng(0); arr=D.values; perms=[]
    for _ in range(2000):
        sh=rng.permutation(arr); e=np.cumprod(1+sh); rr=np.diff(e)/e[:-1]
        perms.append(rr.mean()/rr.std()*np.sqrt(365) if rr.std()>0 else 0)
    return (np.array(perms)>=base).mean()


def main():
    print("FORENSIC RE-DISCOVERY — qualifying signals re-tested at long horizons (T+1)\n")
    # baseline reference: buy & hold portfolio
    bh,bhD=portfolio(lambda df: df["close"]*0+1, 1)   # always long
    bhs=stats_(bh,bhD)
    print(f"Benchmark Buy&Hold portfolio: CAGR {bhs['cagr']:+.0f}%  Sharpe {bhs['sharpe']:+.2f}  MaxDD {bhs['maxdd']:.0f}%\n")

    print(f"| {'Signal':16s} | {'H':>3s} | {'IC':>8s} | {'p(IC)':>6s} | {'Gross%':>6s} | {'Net%':>6s} | "
          f"{'CAGR':>6s} | {'Sharpe':>6s} | {'PF':>5s} | {'MaxDD':>6s} | {'OOS Sh':>6s} | {'perm p':>6s} |")
    print("|"+"-"*18+"|"+"-"*5+"|"+("-"*10+"|")+("-"*8+"|")*1+("-"*8+"|")*2+("-"*8+"|")*4+("-"*8+"|")*2)
    for name,fn in SIGNALS.items():
        for H in [10,30,60,90]:
            ic,p,g=ic_at(fn,H)
            net=g - (365/H)/(365/H)*1.30   # one round trip per holding period
            net=g-1.30
            full=portfolio(fn,H); s=stats_(*full)
            # OOS newest 20%
            idx=full[0].index; cut=idx[int(len(idx)*0.8)]
            oos=portfolio(fn,H,start=str(cut.date())); so=stats_(*oos) if oos else {"sharpe":0,"cagr":0}
            pp=perm_p(*full)
            print(f"| {name:16s} | {H:3d} | {ic:+8.4f} | {p:6.3f} | {g:+6.2f} | {net:+6.2f} | "
                  f"{s['cagr']:+5.0f}% | {s['sharpe']:+6.2f} | {s['pf']:5.2f} | {s['maxdd']:5.0f}% | "
                  f"{so['sharpe']:+6.2f} | {pp:6.3f} |")
    print("\nNet% = gross quintile spread − 1.30% round trip. perm p = P(random timing ≥ strategy Sharpe).")
    print("OOS Sh = Sharpe on newest 20% (unseen). A meaningful edge needs: net%>0, Sharpe>buy&hold,")
    print("positive OOS Sharpe, and perm p<0.05.")


if __name__=="__main__":
    main()
