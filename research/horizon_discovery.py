"""
EDGE DISCOVERY & TIME-HORIZON ANALYSIS (Phases 1-6).

Tests the central hypothesis: gross edge grows with holding period while
transaction cost is paid once per trade — so net edge should cross zero at some
horizon. Measures IC of each validated signal across horizons 1H..2W, finds the
peak/decay, runs cost-survival, scores 3 redesigned systems, rebuilds the
confidence weights from validated ICs, and splits by market regime.

Read-only. ML off. Costs: 0.6%/side fee + 0.05%/side slippage = 1.30% round trip.
"""
import sys, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from scipy import stats

import config
from exchange.market_data import fetch_ohlcv
from core.indicators import enrich
from core.strategies import (score_ema_trend, score_ichimoku, score_market_regime,
                             score_volume, score_rsi, score_bollinger,
                             score_stochastic, score_macd, score_adx)

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]
DAYS = 180
HORIZONS = {"1H":1, "6H":6, "12H":12, "24H":24, "48H":48, "72H":72, "1W":168, "2W":336}
FEE = config.FEE_RATE_PCT/100.0
SLIP = 0.05/100.0
RT_COST = (FEE + SLIP) * 2           # 1.30% round trip per trade

# Raw component scorers (sign: + = bullish as the engine intends)
RAW = {
    "Volume":     lambda df: score_volume(df, "neutral"),
    "Stochastic": lambda df: score_stochastic(df),
    "Bollinger":  lambda df: score_bollinger(df, "neutral"),
    "RSI":        lambda df: score_rsi(df, "neutral"),
    "EMA":        lambda df: score_ema_trend(df),
    "Ichimoku":   lambda df: score_ichimoku(df),
    "Regime":     lambda df: score_market_regime(df),
    "MACD":       lambda df: score_macd(df),
    "ADX":        lambda df: score_adx(df),
}
# Validated signal set (mean-reversion as-is + inverted trend)
SIGNALS = ["Volume","Stochastic","Bollinger","RSI",
           "InvEMA","InvIchimoku","InvRegime","InvMACD","InvADX"]


def signal_series(D, name):
    if name.startswith("Inv"):
        return -D[name[3:]].values
    return D[name].values


def collect(symbol):
    df = fetch_ohlcv(symbol, "1h", limit=DAYS*24+400)
    if df.empty or len(df) < 600:
        return None
    df = enrich(df)
    close = df["close"].values; n = len(df)
    maxh = max(HORIZONS.values())
    # regime + volatility tags per bar (for Phase 6)
    atr = df["atr"].values; e200 = df["ema200"].values if "ema200" in df else close
    rows = []
    for i in range(250, n - maxh):
        sl = df.iloc[max(0,i-300):i+1]
        rec = {}
        for nm, fn in RAW.items():
            try: rec[nm] = float(fn(sl))
            except Exception: rec[nm] = 0.0
        for hl, hb in HORIZONS.items():
            rec[f"fwd_{hl}"] = (close[i+hb]-close[i])/close[i]
        # regime tags
        rec["_bull"] = close[i] > e200[i]
        rec["_atr_pct"] = atr[i]/close[i] if close[i] else 0.0
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["_hivol"] = out["_atr_pct"] > out["_atr_pct"].median()
    return out


def ic(series, fwd):
    s = pd.Series(series)
    if s.std()==0 or np.all(np.isnan(fwd)): return 0.0, 1.0
    r,p = stats.spearmanr(s, fwd, nan_policy="omit")
    return (0.0 if np.isnan(r) else r), (1.0 if np.isnan(p) else p)


def phase1(D):
    print("\n### PHASE 1 — HOLDING-PERIOD IC TABLE\n")
    hdr = "| Signal      | " + " | ".join(f"{h:>6s}" for h in HORIZONS) + " |"
    print(hdr); print("|" + "-"*13 + "|" + ("-"*9+"|")*len(HORIZONS))
    table = {}
    for nm in SIGNALS:
        s = signal_series(D, nm)
        row = {}
        for hl in HORIZONS:
            row[hl] = ic(s, D[f"fwd_{hl}"].values)[0]
        table[nm] = row
        print(f"| {nm:11s} | " + " | ".join(f"{row[h]:+6.3f}" for h in HORIZONS) + " |")
    return table


def phase2(table):
    print("\n### PHASE 2 — EDGE DECAY (peak / where edge disappears)\n")
    print("| Signal      | Peak IC | Peak Horizon | Edge Gone By |")
    print("|" + "-"*13 + "|" + "-"*9 + "|" + "-"*14 + "|" + "-"*14 + "|")
    for nm in SIGNALS:
        row = table[nm]
        peak_h = max(row, key=lambda h: row[h]); peak = row[peak_h]
        # "gone" = first horizon AFTER peak where IC drops below half of peak (or <=0)
        gone = "—"
        order = list(HORIZONS)
        past_peak = order[order.index(peak_h)+1:]
        for h in past_peak:
            if row[h] <= max(0, peak*0.5):
                gone = h; break
        print(f"| {nm:11s} | {peak:+7.3f} | {peak_h:12s} | {gone:12s} |")


def phase3(D):
    print("\n### PHASE 3 — COST SURVIVAL (System C hybrid as the tradeable proxy)\n")
    # Gross edge per trade = long-top minus short-bottom quintile mean fwd return
    sysC = (D["Volume"].values + D["Stochastic"].values
            - D["Ichimoku"].values - D["EMA"].values)
    ser = pd.Series(sysC); hi = ser.quantile(0.8); lo = ser.quantile(0.2)
    print("| Horizon | Gross Edge | Trades/yr | Fees+Slip | Net Edge |")
    print("|" + "-"*9 + "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|")
    first_pos = None
    for hl, hb in HORIZONS.items():
        fwd = D[f"fwd_{hl}"].values
        top = fwd[ser>=hi].mean(); bot = fwd[ser<=lo].mean()
        gross = (top - bot)*100            # long-top + short-bot combined, %
        tpy = 8760/hb                      # max trades/yr per instrument if held hb hours
        net = gross - RT_COST*100          # cost paid once per trade (round trip)
        if net > 0 and first_pos is None: first_pos = hl
        print(f"| {hl:7s} | {gross:+9.3f}% | {tpy:9.0f} | {RT_COST*100:8.2f}% | {net:+7.3f}% |")
    print(f"\n  First horizon where gross edge per trade > {RT_COST*100:.2f}% cost: "
          f"{first_pos if first_pos else 'NONE (no horizon clears cost)'}")
    return first_pos


def backtest_system(D, weights, horizon_label):
    """Non-overlapping holding-period backtest at one horizon. weights: dict name->w."""
    hb = HORIZONS[horizon_label]
    score = np.zeros(len(D))
    for nm,w in weights.items():
        score = score + w*signal_series(D, nm)
    ser = pd.Series(score); hi = ser.quantile(0.7); lo = ser.quantile(0.3)
    fwd = D[f"fwd_{horizon_label}"].values
    # step by hb → non-overlapping trades
    idx = np.arange(0, len(D), hb)
    rets = []
    for i in idx:
        sc = score[i]; f = fwd[i]
        if np.isnan(f): continue
        if sc >= hi:   rets.append(f - RT_COST)        # long
        elif sc <= lo: rets.append(-f - RT_COST)       # short
    rets = np.array(rets)
    if len(rets)==0: return None
    wins = rets[rets>0]; losses = rets[rets<=0]
    wr = len(wins)/len(rets)*100
    pf = wins.sum()/abs(losses.sum()) if len(losses) else float("inf")
    yrs = (len(D)/8760.0)                              # approx years of 1h data per symbol-pool
    tpy = len(rets)/max(yrs,1e-9)
    sharpe = rets.mean()/rets.std()*np.sqrt(tpy) if rets.std()>0 else 0
    eq = np.cumprod(1+rets);
    cagr = (eq[-1]**(1/max(yrs,1e-9))-1)*100 if eq[-1]>0 else -100
    peak=np.maximum.accumulate(eq); maxdd=((eq-peak)/peak).min()*100
    ic_h = ic(score, fwd)[0]
    return dict(ic=ic_h, wr=wr, pf=pf, sharpe=sharpe, cagr=cagr, maxdd=maxdd, n=len(rets))


def phase4(D):
    print("\n### PHASE 4 — REDESIGNED SYSTEMS (each at its peak horizon)\n")
    systems = {
        "A PureMeanRev": {"Volume":1,"Stochastic":1,"Bollinger":1,"RSI":1},
        "B InvTrend":    {"InvEMA":1,"InvIchimoku":1,"InvRegime":1,"InvMACD":1,"InvADX":1},
        "C Hybrid":      {"Volume":1,"Stochastic":1,"InvIchimoku":1,"InvEMA":1},
    }
    print("| System         | Horizon | IC | WinRate | PF | Sharpe | CAGR | MaxDD |")
    print("|" + "-"*16 + "|" + "-"*9 + "|" + "-"*8 + "|" + "-"*9 + "|" + "-"*6 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|")
    for nm, w in systems.items():
        # find peak horizon by IC
        best_h, best_ic = None, -9
        for hl in HORIZONS:
            sc = np.zeros(len(D))
            for k,wt in w.items(): sc = sc + wt*signal_series(D,k)
            i_ = ic(sc, D[f"fwd_{hl}"].values)[0]
            if i_>best_ic: best_ic, best_h = i_, hl
        r = backtest_system(D, w, best_h)
        print(f"| {nm:14s} | {best_h:7s} | {r['ic']:+.3f} | {r['wr']:5.1f}% | "
              f"{r['pf']:4.2f} | {r['sharpe']:+5.2f} | {r['cagr']:+6.1f}% | {r['maxdd']:6.1f}% |")


def phase5(D):
    print("\n### PHASE 5 — CONFIDENCE-SCORE REDESIGN (weights from validated IC)\n")
    # use the horizon where signals are strongest collectively (24H..1W); pick 1W
    H = "1W"; fwd = D[f"fwd_{H}"].values
    rows = []
    for nm in SIGNALS:
        r,p = ic(signal_series(D,nm), fwd)
        rows.append((nm, r, p))
    # keep only positive & significant
    keep = [(nm,r) for nm,r,p in rows if r>0 and p<0.05]
    tot = sum(r for _,r in keep) or 1
    print(f"(weights = IC share among positive & significant signals at {H})\n")
    print("| Component   | IC@1W | Weight |")
    print("|" + "-"*13 + "|" + "-"*8 + "|" + "-"*8 + "|")
    for nm,r,p in sorted(rows, key=lambda x:-x[1]):
        if r>0 and p<0.05:
            print(f"| {nm:11s} | {r:+.3f} | {r/tot*100:5.1f}% |")
    print("\n  Dropped (negative or insignificant at 1W):")
    for nm,r,p in rows:
        if not (r>0 and p<0.05):
            print(f"    {nm:11s} IC={r:+.3f} p={p:.3f}")


def phase6(D):
    print("\n### PHASE 6 — REGIME-CONDITIONAL IC (System C hybrid)\n")
    sysC = (D["Volume"].values + D["Stochastic"].values
            - D["Ichimoku"].values - D["EMA"].values)
    H = "24H"; fwd = D[f"fwd_{H}"].values
    masks = {
        "Bull":     D["_bull"].values,
        "Bear":     ~D["_bull"].values,
        "HighVol":  D["_hivol"].values,
        "LowVol":   ~D["_hivol"].values,
    }
    print(f"(IC of System C vs {H} forward return, by regime)\n")
    print("| Regime   | IC | n |")
    print("|" + "-"*10 + "|" + "-"*9 + "|" + "-"*8 + "|")
    for nm, m in masks.items():
        r = ic(sysC[m], fwd[m])[0]
        print(f"| {nm:8s} | {r:+.3f} | {int(m.sum()):6d} |")


def main():
    print(f"HORIZON DISCOVERY | {len(SYMBOLS)} symbols | {DAYS}d | cost {RT_COST*100:.2f}% round trip")
    dfs=[]
    for s in SYMBOLS:
        d=collect(s)
        if d is not None: dfs.append(d); print(f"  {s}: {len(d)} samples")
    D=pd.concat(dfs, ignore_index=True)
    print(f"  POOLED: {len(D)} samples")
    table=phase1(D)
    phase2(table)
    phase3(D)
    phase4(D)
    phase5(D)
    phase6(D)


if __name__=="__main__":
    main()
