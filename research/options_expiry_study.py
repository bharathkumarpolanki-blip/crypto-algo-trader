"""
SCHEDULED-FLOW EVENT STUDY #1 (the free first move) — OPTIONS EXPIRY effects.

Forward-visible flow: Deribit BTC/ETH options expire on a FIXED calendar
(weekly = every Friday 08:00 UTC; monthly = last Friday). Dealer gamma hedging
into expiry is documented in TradFi to (a) pin price toward high-OI strikes and
(b) suppress realized vol into expiry, then release it after. Unlike indicators,
this flow is KNOWN IN ADVANCE — you can position before it happens.

This $0 version tests the CALENDAR effect on price/vol around expiries using only
free daily price (ccxt). Strict event-study design, in-sample vs out-of-sample.
(The live max-pain/gamma signal would later use Deribit's free current OI.)

Tests:
  1. Monthly-expiry effect: returns pre/event/post + vol pre vs post.
  2. Weekly-Friday effect.
  3. OOS split (first 70% / last 30%) + per-side t-tests.
Success = a stable, significant, OOS-surviving calendar effect.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats
from exchange.market_data import fetch_ohlcv

COINS = ["BTC/USD", "ETH/USD"]


def monthly_expiries(idx):
    """Last Friday of each month within the price index range."""
    days = pd.Series(idx)
    out = []
    for (y, m), grp in days.groupby([days.dt.year, days.dt.month]):
        fri = grp[grp.dt.dayofweek == 4]          # Fridays
        if len(fri): out.append(fri.iloc[-1])     # last Friday
    return pd.DatetimeIndex(out)


def event_study(close, events, pre=5, post=5):
    """For each event date present in `close`, compute pre/event/post returns + vol."""
    rows = []
    c = close
    idxpos = {t: i for i, t in enumerate(c.index)}
    for e in events:
        if e not in idxpos: continue
        i = idxpos[e]
        if i - pre - 10 < 0 or i + post + 10 >= len(c): continue
        p = c.values
        pre_ret  = p[i-1]/p[i-pre] - 1                     # T-pre → T-1
        day_ret  = p[i]/p[i-1] - 1                         # expiry day
        post_ret = p[i+post]/p[i] - 1                      # T → T+post
        r = np.diff(p)/p[:-1]
        vol_pre  = r[i-11:i-1].std()
        vol_post = r[i:i+10].std()
        rows.append((e, pre_ret, day_ret, post_ret, vol_pre, vol_post))
    return pd.DataFrame(rows, columns=["date","pre","day","post","vol_pre","vol_post"])


def ttest(x):
    x = x[~np.isnan(x)]
    if len(x) < 8: return 0.0, 1.0
    t, p = stats.ttest_1samp(x, 0)
    return x.mean(), p


def report(df, label):
    print(f"\n  {label}  (n={len(df)} events)")
    for col, name in [("pre","pre (T-5→T-1)"),("day","expiry day"),("post","post (T→T+5)")]:
        m, p = ttest(df[col].values)
        flag = "✅" if p < 0.05 else ""
        print(f"    {name:18s}: mean {m*100:+.2f}%   p={p:.3f} {flag}")
    # vol compression into expiry?
    dv = (df["vol_post"] - df["vol_pre"]).values
    m, p = ttest(dv)
    print(f"    vol_post − vol_pre : {m*100:+.3f}%/day  p={p:.3f}  "
          f"{'✅ vol expands after expiry' if (m>0 and p<0.05) else ''}")


def main():
    print("OPTIONS-EXPIRY CALENDAR EVENT STUDY (free daily price; in-sample vs OOS)\n")
    for sym in COINS:
        df = fetch_ohlcv(sym, "1d", limit=2500)
        if df.empty or len(df) < 400:
            print(f"{sym}: no data"); continue
        close = df["close"]
        mexp = monthly_expiries(close.index)
        es = event_study(close, mexp)
        if len(es) < 12:
            print(f"{sym}: too few expiries"); continue
        print(f"{'='*66}\n{sym}  {close.index[0].date()}→{close.index[-1].date()}  "
              f"({len(mexp)} monthly expiries)\n{'='*66}")
        # in-sample / OOS split by event date
        es = es.sort_values("date").reset_index(drop=True)
        cut = int(len(es)*0.7)
        report(es.iloc[:cut], "IN-SAMPLE (first 70%)")
        report(es.iloc[cut:], "OUT-OF-SAMPLE (last 30%)")
        report(es, "FULL SAMPLE")

        # weekly-Friday effect (all Fridays, daily return that day)
        fr = close.index.dayofweek == 4
        rfri = close.pct_change()[fr]
        roth = close.pct_change()[~fr]
        m1,p1 = ttest(rfri.values); m2,_ = ttest(roth.values)
        print(f"\n  Weekly Friday daily return: {m1*100:+.2f}% (p={p1:.3f})  "
              f"vs non-Friday {m2*100:+.2f}%  {'✅' if p1<0.05 else ''}")

    print("\nVERDICT GUIDE: a deployable scheduled-flow edge needs the SAME effect to be")
    print("significant IN-SAMPLE *and* OUT-OF-SAMPLE with a consistent sign. If it only")
    print("shows in-sample, it's noise — same bar as everything else.")


if __name__ == "__main__":
    main()
