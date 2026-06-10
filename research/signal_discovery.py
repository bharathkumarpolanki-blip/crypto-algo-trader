"""
SIGNAL-DISCOVERY investigation (NOT strategy optimization).

Measures the raw predictive power of each indicator component against FORWARD
returns — independent of the existing (anti-predictive) composite. Computes IC,
p-value, mutual information, win rate, and avg forward return; tests long/short/
inverted variants; sweeps all 2- and 3-factor combinations; tests trend-signal
inversion; and reports whether any positive, significant, cost-surviving edge
exists.

Read-only. ML off. Fees+slippage applied where trade-level PF is computed.
"""
import sys, warnings, itertools
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
from core.strategies import (score_ema_trend, score_ichimoku, score_market_regime,
                             score_volume, score_rsi, score_bollinger,
                             score_stochastic, score_macd, score_adx)

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]
DAYS = 180
HORIZON = 6                       # forward bars for the IC target
FEE = config.FEE_RATE_PCT / 100.0
SLIP = 0.05 / 100.0
COST = (FEE + SLIP) * 2           # round-trip cost fraction
WIN_THR = 0.0                     # forward return > 0 counts as a win (direction-adjusted)

# Component scorers. Note: each returns a signed score; sign convention =
# "positive ⇒ the component is bullish". trend4h arg defaulted to 'neutral'.
COMPONENTS = {
    "EMA Trend":  lambda df: score_ema_trend(df),
    "Ichimoku":   lambda df: score_ichimoku(df),
    "Regime":     lambda df: score_market_regime(df),
    "Volume":     lambda df: score_volume(df, "neutral"),
    "RSI":        lambda df: score_rsi(df, "neutral"),
    "Bollinger":  lambda df: score_bollinger(df, "neutral"),
    "Stochastic": lambda df: score_stochastic(df),
    "MACD":       lambda df: score_macd(df),
    "ADX":        lambda df: score_adx(df),
}
TREND_SIGNALS = ["EMA Trend", "Ichimoku", "Regime", "MACD", "ADX"]


def collect(symbol):
    """Per-bar component scores + forward return. Causal (scores use only past)."""
    df = fetch_ohlcv(symbol, "1h", limit=DAYS*24+300)
    if df.empty or len(df) < 400:
        return None
    df = enrich(df)
    close = df["close"].values
    n = len(df)
    rows = []
    for i in range(200, n - HORIZON):
        sl = df.iloc[max(0, i-300):i+1]   # include current bar; scorers read .iloc[-1]
        fwd = (close[i+HORIZON] - close[i]) / close[i]
        rec = {"fwd": fwd}
        for name, fn in COMPONENTS.items():
            try:
                rec[name] = float(fn(sl))
            except Exception:
                rec[name] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def pooled(dfs):
    return pd.concat(dfs, ignore_index=True)


def ic_of(series, fwd):
    if series.std() == 0:
        return 0.0, 1.0
    ic, p = stats.spearmanr(series, fwd)
    return (0.0 if np.isnan(ic) else ic), (1.0 if np.isnan(p) else p)


def section1(D):
    fwd = D["fwd"].values
    print("\n### 1. RAW COMPONENT PREDICTIVE POWER (pooled across symbols)\n")
    print(f"| {'Component':11s} | {'IC':>7s} | {'p-value':>7s} | {'MutInfo':>7s} | {'WinRate':>7s} | {'AvgFwdRet':>9s} |")
    print("|" + "-"*13 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*11 + "|")
    out = {}
    for name in COMPONENTS:
        col = D[name].values
        ic, p = ic_of(D[name], fwd)
        try:
            mi = float(mutual_info_regression(col.reshape(-1,1), fwd, random_state=0)[0])
        except Exception:
            mi = 0.0
        # direction-adjusted win rate: when component is bullish (>0) expect fwd>0,
        # when bearish (<0) expect fwd<0. Neutral (==0) excluded.
        mask = col != 0
        signed = np.sign(col[mask]) * fwd[mask]
        wr = (signed > WIN_THR).mean()*100 if mask.sum() else 0.0
        afr = (np.sign(col[mask]) * fwd[mask]).mean()*100 if mask.sum() else 0.0
        out[name] = (ic, p, mi)
        print(f"| {name:11s} | {ic:+7.4f} | {p:7.3f} | {mi:7.4f} | {wr:6.1f}% | {afr:+8.3f}% |")
    return out


def section2(D):
    fwd = D["fwd"].values
    print("\n### 2. LONG / SHORT / INVERTED IC PER COMPONENT\n")
    print(f"| {'Component':11s} | {'Long IC':>8s} | {'Short IC':>8s} | {'Inverted IC':>11s} | {'Best Variant':12s} |")
    print("|" + "-"*13 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*13 + "|" + "-"*14 + "|")
    for name in COMPONENTS:
        col = D[name].values
        # Long-only: keep bullish signals (>0), IC of magnitude vs fwd
        lmask = col > 0
        long_ic = ic_of(pd.Series(col[lmask]), fwd[lmask])[0] if lmask.sum() > 30 else 0.0
        # Short-only: bearish signals (<0); expect negative fwd → IC of (-col) vs fwd on subset
        smask = col < 0
        short_ic = ic_of(pd.Series(-col[smask]), -fwd[smask])[0] if smask.sum() > 30 else 0.0
        # Inverted: full series, flipped sign
        inv_ic = ic_of(pd.Series(-col), fwd)[0]
        base_ic = ic_of(pd.Series(col), fwd)[0]
        cands = {"Long": long_ic, "Short": short_ic, "Inverted": inv_ic, "Original": base_ic}
        best = max(cands, key=lambda k: cands[k])
        print(f"| {name:11s} | {long_ic:+8.4f} | {short_ic:+8.4f} | {inv_ic:+11.4f} | {best:12s} |")


def combo_ic(D, names):
    """Equal-weight sum of (optionally inverted) component scores → IC vs fwd."""
    fwd = D["fwd"].values
    s = np.zeros(len(D))
    for nm in names:
        s = s + D[nm].values
    return ic_of(pd.Series(s), fwd), s


def section3(D):
    fwd = D["fwd"].values
    print("\n### 3. ALL 2-FACTOR COMBINATIONS (equal-weight sum)\n")
    print(f"| {'Combination':24s} | {'IC':>7s} | {'p-value':>7s} | {'TradeCt':>7s} |")
    print("|" + "-"*26 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|")
    res = []
    for a, b in itertools.combinations(COMPONENTS, 2):
        (ic, p), s = combo_ic(D, [a, b])
        tc = int((s != 0).sum())
        res.append((f"{a}+{b}", ic, p, tc))
    for name, ic, p, tc in sorted(res, key=lambda x: -x[1]):
        print(f"| {name:24s} | {ic:+7.4f} | {p:7.3f} | {tc:7d} |")
    return res


def section4(D):
    print("\n### 4. ALL 3-FACTOR COMBINATIONS — TOP 20 by IC\n")
    res = []
    for a, b, c in itertools.combinations(COMPONENTS, 3):
        (ic, p), s = combo_ic(D, [a, b, c])
        res.append((f"{a}+{b}+{c}", ic, p, int((s != 0).sum())))
    print(f"| {'#':>2s} | {'Combination':34s} | {'IC':>7s} | {'p-value':>7s} | {'TradeCt':>7s} |")
    print("|" + "-"*4 + "|" + "-"*36 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|")
    for rank, (name, ic, p, tc) in enumerate(sorted(res, key=lambda x: -x[1])[:20], 1):
        print(f"| {rank:2d} | {name:34s} | {ic:+7.4f} | {p:7.3f} | {tc:7d} |")
    return res


def section5(D):
    fwd = D["fwd"].values
    print("\n### 5. TREND-SIGNAL INVERSION TEST\n")
    print(f"| {'Signal':11s} | {'Original IC':>11s} | {'Inverted IC':>11s} |")
    print("|" + "-"*13 + "|" + "-"*13 + "|" + "-"*13 + "|")
    for name in TREND_SIGNALS:
        base = ic_of(pd.Series(D[name].values), fwd)[0]
        inv = ic_of(pd.Series(-D[name].values), fwd)[0]
        print(f"| {name:11s} | {base:+11.4f} | {inv:+11.4f} |")


def section6(D, twos, threes):
    fwd = D["fwd"].values
    print("\n### 6. POSITIVE, SIGNIFICANT, COST-SURVIVING EDGE?\n")
    print(f"  Round-trip cost = {COST*100:.2f}%. Edge per signal must beat this to be tradeable.")
    # best single, 2-, 3-factor by IC
    singles = [(n, *ic_of(pd.Series(D[n].values), fwd)) for n in COMPONENTS]
    best_single = max(singles, key=lambda x: x[1])
    best_two = max(twos, key=lambda x: x[1])
    best_three = max(threes, key=lambda x: x[1])
    print(f"  Best single : {best_single[0]:18s} IC={best_single[1]:+.4f} p={best_single[2]:.3f}")
    print(f"  Best 2-factor: {best_two[0]:18s} IC={best_two[1]:+.4f} p={best_two[2]:.3f}")
    print(f"  Best 3-factor: {best_three[0]:18s} IC={best_three[1]:+.4f} p={best_three[2]:.3f}")
    # Translate best IC to an approximate edge: avg |fwd| in top-signal decile
    # Estimate captured edge for the best 3-factor: mean fwd in its top quintile
    (ic, p), s = combo_ic(D, best_three[0].split("+"))
    ser = pd.Series(s)
    top = D["fwd"].values[ser >= ser.quantile(0.8)]
    bot = D["fwd"].values[ser <= ser.quantile(0.2)]
    spread = (top.mean() - bot.mean())*100
    print(f"\n  Best 3-factor top-quintile mean fwd: {top.mean()*100:+.3f}%")
    print(f"  Best 3-factor bot-quintile mean fwd: {bot.mean()*100:+.3f}%")
    print(f"  Long-top / short-bottom gross edge per signal: {spread:+.3f}%  vs cost {COST*100:.2f}%")
    survives = (best_three[1] > 0 and best_three[2] < 0.05 and spread > COST*100)
    print(f"  Survives cost? {'YES' if survives else 'NO'}")
    return best_single, best_two, best_three, spread, survives


def main():
    print(f"SIGNAL DISCOVERY | {len(SYMBOLS)} symbols | {DAYS}d | horizon {HORIZON} bars | "
          f"cost {COST*100:.2f}% round trip")
    dfs = []
    for s in SYMBOLS:
        d = collect(s)
        if d is not None:
            dfs.append(d)
            print(f"  {s}: {len(d)} samples")
    D = pooled(dfs)
    print(f"  POOLED: {len(D)} samples")
    section1(D)
    section2(D)
    twos = section3(D)
    threes = section4(D)
    section5(D)
    bs, bt, b3, spread, survives = section6(D, twos, threes)

    print("\n### 7. FINAL CONCLUSION\n")
    fwd = D["fwd"].values
    trend_orig = np.mean([ic_of(pd.Series(D[n].values), fwd)[0] for n in TREND_SIGNALS])
    trend_inv  = np.mean([ic_of(pd.Series(-D[n].values), fwd)[0] for n in TREND_SIGNALS])
    mr_comps = ["RSI","Bollinger","Stochastic","Volume"]
    mr_ic = np.mean([ic_of(pd.Series(D[n].values), fwd)[0] for n in mr_comps])
    print(f"  Mean trend IC (original):  {trend_orig:+.4f}")
    print(f"  Mean trend IC (inverted):  {trend_inv:+.4f}")
    print(f"  Mean mean-reversion IC:    {mr_ic:+.4f}")
    print(f"  Best edge survives cost:   {'YES' if survives else 'NO'}  "
          f"(best 3-factor spread {spread:+.3f}% vs {COST*100:.2f}% cost)")


if __name__ == "__main__":
    main()
