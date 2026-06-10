"""
Breakout / Retest / Floor strategy backtest — daily, long-only, fees on.

Tests the price-action strategies the user asked about:
  1. Donchian breakout   — buy N-day high breakout, exit on M-day low (Turtle)
  2. Breakout + retest    — buy only after a breakout pulls back & holds the level
  3. Support floor bounce — buy near established support when it bounces

Each on BTC/ETH/SOL over full daily history, vs buy & hold. No look-ahead.

Usage: python3 breakout_backtest.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

import config
from exchange.market_data import fetch_ohlcv

colorama_init(autoreset=True)
FEE = getattr(config, "FEE_RATE_PCT", 0.6) / 100.0
CAP = 1000.0
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]


# ── Strategy signal generators → boolean "in position" series ─────────────────

def donchian(df, enter_n=50, exit_n=20):
    """Turtle breakout: enter on N-day high, exit on M-day low."""
    hi = df["high"].rolling(enter_n).max().shift(1).values   # prior N-day high
    lo = df["low"].rolling(exit_n).min().shift(1).values
    inpos = np.zeros(len(df), dtype=bool)
    holding = False
    c = df["close"].values
    for i in range(len(df)):
        if not holding and not np.isnan(hi[i]) and c[i] > hi[i]:
            holding = True
        elif holding and not np.isnan(lo[i]) and c[i] < lo[i]:
            holding = False
        inpos[i] = holding
    return pd.Series(inpos, index=df.index)


def breakout_retest(df, level_n=50, retest_pct=2.0, stop_pct=5.0, max_wait=10):
    """
    Breakout then retest: price breaks the N-day high, then we wait for it to
    pull back toward that level (within retest_pct%) and hold — only then enter.
    Exit on a stop_pct% drop from entry.
    """
    hi = df["high"].rolling(level_n).max().shift(1).values
    c  = df["close"].values
    inpos = np.zeros(len(df), dtype=bool)
    state = "wait"      # wait → armed (broke out, awaiting retest) → in
    level = entry = 0.0
    wait_ct = 0
    for i in range(len(df)):
        if state == "wait":
            if not np.isnan(hi[i]) and c[i] > hi[i]:
                level = hi[i]; state = "armed"; wait_ct = 0
        elif state == "armed":
            wait_ct += 1
            # retest: price pulls back to within retest_pct% of the level and holds above it
            if level*(1) <= c[i] <= level*(1 + retest_pct/100):
                state = "in"; entry = c[i]
            elif c[i] < level*(1 - retest_pct/100) or wait_ct > max_wait:
                state = "wait"   # failed retest (broke back down or timed out)
        elif state == "in":
            if c[i] < entry*(1 - stop_pct/100):
                state = "wait"
        inpos[i] = (state == "in")
    return pd.Series(inpos, index=df.index)


def support_bounce(df, support_n=30, near_pct=3.0, stop_pct=5.0, target_pct=12.0):
    """
    Buy near established support (N-day low) when price bounces (closes up),
    exit on stop_pct% loss or target_pct% gain.
    """
    lo = df["low"].rolling(support_n).min().shift(1).values
    c  = df["close"].values
    o  = df["open"].values
    inpos = np.zeros(len(df), dtype=bool)
    holding = False; entry = 0.0
    for i in range(len(df)):
        if not holding:
            near_support = not np.isnan(lo[i]) and c[i] <= lo[i]*(1 + near_pct/100) and c[i] >= lo[i]
            bounced = c[i] > o[i]
            if near_support and bounced:
                holding = True; entry = c[i]
        else:
            if c[i] <= entry*(1 - stop_pct/100) or c[i] >= entry*(1 + target_pct/100):
                holding = False
        inpos[i] = holding
    return pd.Series(inpos, index=df.index)


STRATEGIES = {
    "Donchian breakout": donchian,
    "Breakout + retest": breakout_retest,
    "Support bounce":    support_bounce,
}


# ── Backtest engine: long-only, full allocation when in position, fees ────────

def run(df, signal):
    inpos = signal.values
    c = df["close"].values
    cash, units, entry = CAP, 0.0, 0.0
    eq, trades = [], []
    for i in range(len(df)):
        if inpos[i] and units == 0.0:
            units = (cash*(1-FEE))/c[i]; entry = c[i]; cash = 0.0
        elif not inpos[i] and units > 0.0:
            cash = units*c[i]*(1-FEE); trades.append((c[i]-entry)/entry); units = 0.0
        eq.append(cash + units*c[i])
    if units > 0:
        cash = units*c[-1]*(1-FEE); trades.append((c[-1]-entry)/entry); eq[-1] = cash
    return pd.Series(eq, index=df.index), trades


def metrics(eq, trades, df):
    end = eq.iloc[-1]; yrs = (df.index[-1]-df.index[0]).days/365.25
    cagr = (end/CAP)**(1/yrs)-1 if yrs>0 and end>0 else -1
    maxdd = ((eq-eq.cummax())/eq.cummax()).min()
    wins = [t for t in trades if t>0]
    wr = len(wins)/len(trades)*100 if trades else 0
    calmar = cagr/abs(maxdd) if maxdd<0 else 0
    return (end/CAP-1)*100, cagr*100, maxdd*100, len(trades), wr, calmar


def main():
    print(f"\nBreakout / Retest / Floor — daily, long-only, fees {FEE*100:.1f}%/side\n" + "="*78)
    for sym in SYMBOLS:
        df = fetch_ohlcv(sym, "1d", limit=2000)
        if df.empty or len(df) < 300:
            print(f"\n{sym}: not enough daily data"); continue
        yrs = (df.index[-1]-df.index[0]).days/365.25
        print(f"\n{Fore.CYAN}{sym}{Style.RESET_ALL}  {df.index[0].date()} → {df.index[-1].date()} ({yrs:.1f}y)")

        # Buy & hold benchmark
        bh_units=(CAP*(1-FEE))/df['close'].iloc[0]; bh=bh_units*df['close']
        bret=(bh.iloc[-1]/CAP-1)*100; bdd=((bh-bh.cummax())/bh.cummax()).min()*100
        rows=[["BUY & HOLD", f"{bret:+.0f}%", f"{bdd:.0f}%", "1", "—", f"{(bret/100)/(abs(bdd)/100) if bdd else 0:.2f}"]]

        for name, fn in STRATEGIES.items():
            eq, trades = run(df, fn(df))
            ret, cagr, dd, nt, wr, cal = metrics(eq, trades, df)
            c = Fore.GREEN if ret > 0 else Fore.RED
            rows.append([name, c+f"{ret:+.0f}%"+Style.RESET_ALL, f"{dd:.0f}%",
                         str(nt), f"{wr:.0f}%", f"{cal:.2f}"])
        print(tabulate(rows, headers=["Strategy","TotalRet","MaxDD","Trades","Win%","Calmar"],
                       tablefmt="rounded_outline"))
    print(f"\n{Fore.YELLOW}A strategy must beat BUY & HOLD on Calmar (return per unit of drawdown) "
          f"to be worth it.{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
