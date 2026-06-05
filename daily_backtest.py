"""
Daily trend-following backtest — honest multi-year experiment.

Tests several classic, documented trend-following rules on DAILY candles over the
full available history (~5.5 years for BTC/ETH), long-only, fees included.

Crucially compares against BUY & HOLD — a trend strategy must justify itself.
In crypto bull markets, holding is hard to beat on raw return; a trend system's
real value is usually MUCH smaller drawdowns (you sidestep the crashes), i.e.
better risk-adjusted return. We report both.

Strategies:
  1. SMA200       — long while close > 200-day SMA, flat otherwise
  2. GoldenCross  — long while 50-day SMA > 200-day SMA
  3. Donchian     — Turtle-style: enter on 50-day high, exit on 20-day low
  4. MA+ATRtrail  — long above 100-day SMA, exit on a chandelier ATR trailing stop

Usage: python3 daily_backtest.py
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

FEE = getattr(config, "FEE_RATE_PCT", 0.6) / 100.0   # per side


# ── Indicators ────────────────────────────────────────────────────────────────

def sma(s, n):  return s.rolling(n).mean()

def atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ── Strategy signal generators: return a boolean "in-market" Series ───────────

def sig_sma200(df):
    return df["close"] > sma(df["close"], 200)

def sig_golden(df):
    return sma(df["close"], 50) > sma(df["close"], 200)

def sig_donchian(df, enter_n=50, exit_n=20):
    hi = df["high"].rolling(enter_n).max()
    lo = df["low"].rolling(exit_n).min()
    inpos = pd.Series(False, index=df.index)
    holding = False
    for i in range(len(df)):
        if not holding and df["close"].iloc[i] >= hi.iloc[i-1] if i > 0 else False:
            holding = True
        elif holding and df["close"].iloc[i] <= lo.iloc[i-1] if i > 0 else False:
            holding = False
        inpos.iloc[i] = holding
    return inpos

def sig_ma_atr(df, ma_n=100, atr_mult=4.0):
    above = df["close"] > sma(df["close"], ma_n)
    a = atr(df, 22)
    inpos = pd.Series(False, index=df.index)
    holding = False
    peak = 0.0
    for i in range(len(df)):
        c = df["close"].iloc[i]
        if not holding:
            if bool(above.iloc[i]):
                holding = True; peak = c
        else:
            peak = max(peak, c)
            trail = peak - atr_mult * (a.iloc[i] if not np.isnan(a.iloc[i]) else 0)
            if c < trail or not bool(above.iloc[i]):
                holding = False
        inpos.iloc[i] = holding
    return inpos


STRATEGIES = {
    "SMA200":      sig_sma200,
    "GoldenCross": sig_golden,
    "Donchian":    sig_donchian,
    "MA+ATRtrail": sig_ma_atr,
}


# ── Backtest engine (long-only, daily, position = full allocation when in) ────

def run(df, signal_fn, capital=1000.0):
    inpos = signal_fn(df).fillna(False).values
    close = df["close"].values
    n = len(df)

    cash = capital
    units = 0.0
    equity_curve = []
    trades = []
    entry_price = 0.0

    for i in range(n):
        # Act on yesterday's signal at today's open≈close (daily, use close)
        want_in = inpos[i]
        price = close[i]

        if want_in and units == 0.0:           # ENTER
            units = (cash * (1 - FEE)) / price
            entry_price = price
            cash = 0.0
        elif not want_in and units > 0.0:       # EXIT
            cash = units * price * (1 - FEE)
            trades.append((price - entry_price) / entry_price)
            units = 0.0

        equity_curve.append(cash + units * price)

    # Close any open position at the end
    if units > 0.0:
        cash = units * close[-1] * (1 - FEE)
        trades.append((close[-1] - entry_price) / entry_price)
        equity_curve[-1] = cash

    eq = pd.Series(equity_curve, index=df.index)
    return eq, trades


def metrics(eq, trades, df, start_cap=1000.0):
    end = eq.iloc[-1]
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (end / start_cap) ** (1 / yrs) - 1 if yrs > 0 and end > 0 else -1
    peak = eq.cummax()
    maxdd = ((eq - peak) / peak).min()
    wins = [t for t in trades if t > 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    # Calmar = CAGR / |maxDD| — risk-adjusted return
    calmar = cagr / abs(maxdd) if maxdd < 0 else float("inf")
    return {
        "ret_pct": (end / start_cap - 1) * 100,
        "cagr_pct": cagr * 100,
        "maxdd_pct": maxdd * 100,
        "trades": len(trades),
        "win_pct": wr,
        "calmar": calmar,
        "end": end,
    }


def buy_hold(df, start_cap=1000.0):
    units = (start_cap * (1 - FEE)) / df["close"].iloc[0]
    eq = units * df["close"]
    return eq


def main():
    symbols = ["BTC/USD", "ETH/USD"]
    print(f"\nDaily trend-following — honest multi-year test (fees {FEE*100:.1f}%/side)\n"
          + "=" * 78)

    for sym in symbols:
        df = fetch_ohlcv(sym, "1d", limit=2000)
        if df.empty or len(df) < 250:
            print(f"\n{sym}: not enough daily data"); continue
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        print(f"\n{Fore.CYAN}{sym}{Style.RESET_ALL}  "
              f"{df.index[0].date()} → {df.index[-1].date()}  ({yrs:.1f} years, {len(df)} days)")

        # Benchmark: buy & hold
        bh = buy_hold(df)
        bh_m = metrics(bh, [(df['close'].iloc[-1]-df['close'].iloc[0])/df['close'].iloc[0]], df)
        rows = [["BUY & HOLD", f"{bh_m['ret_pct']:+.0f}%", f"{bh_m['cagr_pct']:+.0f}%",
                 f"{bh_m['maxdd_pct']:.0f}%", "1", "—", f"{bh_m['calmar']:.2f}"]]

        for name, fn in STRATEGIES.items():
            eq, trades = run(df, fn)
            m = metrics(eq, trades, df)
            ret_c   = Fore.GREEN if m["ret_pct"] >= bh_m["ret_pct"] else Fore.WHITE
            calmar_c= Fore.GREEN if m["calmar"] > bh_m["calmar"] else Fore.RED
            rows.append([
                name,
                ret_c + f"{m['ret_pct']:+.0f}%" + Style.RESET_ALL,
                f"{m['cagr_pct']:+.0f}%",
                f"{m['maxdd_pct']:.0f}%",
                str(m["trades"]),
                f"{m['win_pct']:.0f}%",
                calmar_c + f"{m['calmar']:.2f}" + Style.RESET_ALL,
            ])

        print(tabulate(rows,
              headers=["Strategy", "TotalRet", "CAGR", "MaxDD", "Trades", "Win%", "Calmar"],
              tablefmt="rounded_outline"))

    print(f"\n{Fore.YELLOW}Calmar = CAGR / MaxDrawdown (higher = better risk-adjusted). "
          f"A trend system earns its keep by cutting MaxDD vs buy & hold,{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}even if total return is lower — survivable drawdowns matter for real money.{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
