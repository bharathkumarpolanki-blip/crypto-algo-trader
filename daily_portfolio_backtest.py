"""
Daily trend-following PORTFOLIO backtest — the honest proof.

Each day:
  1. Look at every coin in the basket.
  2. Keep only those in a confirmed uptrend (close > 200d SMA, 50d > 200d).
  3. Rank them by risk-adjusted momentum; hold the top N, equal-weight.
  4. Sit in cash for any slot with no qualifying coin.
Rebalances daily with fees. Compares against benchmarks:
  - Buy & hold BTC (the usual yardstick)
  - Equal-weight buy & hold the whole basket

Usage: python3 daily_portfolio_backtest.py
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
from core.daily_strategy import in_uptrend, trend_strength

colorama_init(autoreset=True)

FEE      = getattr(config, "FEE_RATE_PCT", 0.6) / 100.0
TOP_N    = 5          # how many trending coins to hold at once
CAPITAL  = 1000.0

# Basket — liquid majors/alts with multi-year daily history on Coinbase
BASKET = ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD",
          "LINK/USD","DOT/USD","AVAX/USD","LTC/USD","BCH/USD","ATOM/USD",
          "UNI/USD","XLM/USD","ETC/USD","AAVE/USD"]


def load_basket():
    """Fetch daily candles, align on common dates, return dict of clean frames."""
    data = {}
    for sym in BASKET:
        df = fetch_ohlcv(sym, "1d", limit=1500)
        if not df.empty and len(df) > 250:
            data[sym] = df
    return data


def build_panels(data):
    """Aligned wide panels of close + signal + strength across all coins."""
    closes, ups, strengths = {}, {}, {}
    for sym, df in data.items():
        closes[sym]    = df["close"]
        ups[sym]       = in_uptrend(df)
        strengths[sym] = trend_strength(df)
    close_p = pd.DataFrame(closes).sort_index()
    up_p    = pd.DataFrame(ups).reindex(close_p.index)
    str_p   = pd.DataFrame(strengths).reindex(close_p.index)
    return close_p, up_p, str_p


def run_portfolio(close_p, up_p, str_p, top_n=TOP_N, start_cap=CAPITAL,
                  rebalance_every=7):
    """
    Top-N trend portfolio that ONLY trades the delta (sells leavers, buys
    entrants) and rebalances every `rebalance_every` days. This avoids the
    catastrophic churn of liquidating the whole book each time.
    Returns (equity Series, number of individual buy/sell trades).
    """
    dates = close_p.index
    equity = []
    holdings: dict[str, float] = {}   # symbol → units held
    cash = start_cap
    n_trades = 0
    start_i = 205   # 200d SMA warm-up

    for i in range(start_i, len(dates)):
        px = close_p.iloc[i]

        def mark():
            return cash + sum(u * px[s] for s, u in holdings.items()
                              if not np.isnan(px.get(s, np.nan)))

        # Only act on rebalance days
        if (i - start_i) % rebalance_every == 0:
            up_today, str_today = up_p.iloc[i], str_p.iloc[i]
            cands = [s for s in close_p.columns
                     if bool(up_today.get(s, False))
                     and not np.isnan(str_today.get(s, np.nan))
                     and not np.isnan(px.get(s, np.nan))]
            cands.sort(key=lambda s: str_today[s], reverse=True)
            target = set(cands[:top_n])
            current = set(holdings.keys())

            # SELL only coins leaving the target (or that fell out of uptrend)
            for s in list(current - target):
                if not np.isnan(px.get(s, np.nan)):
                    cash += holdings[s] * px[s] * (1 - FEE)
                    n_trades += 1
                del holdings[s]

            # BUY only coins entering the target, sized to an equal slice
            entrants = list(target - set(holdings.keys()))
            if entrants:
                port_val = mark()
                # Target equal weight across the full target set
                slice_val = port_val / max(len(target), 1)
                for s in entrants:
                    if slice_val > 1 and cash >= slice_val * 0.5 and not np.isnan(px.get(s, np.nan)):
                        spend = min(slice_val, cash)
                        holdings[s] = (spend * (1 - FEE)) / px[s]
                        cash -= spend
                        n_trades += 1

        equity.append(mark())

    return pd.Series(equity, index=dates[start_i:]), n_trades


def metrics(eq, start_cap=CAPITAL):
    end  = eq.iloc[-1]
    yrs  = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (end / start_cap) ** (1 / yrs) - 1 if yrs > 0 and end > 0 else -1
    peak = eq.cummax()
    maxdd= ((eq - peak) / peak).min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else float("inf")
    # Sharpe on daily returns
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    return {"ret": (end/start_cap-1)*100, "cagr": cagr*100, "maxdd": maxdd*100,
            "calmar": calmar, "sharpe": sharpe, "end": end, "yrs": yrs}


def bench_hold(close_p, syms, start_i=205, start_cap=CAPITAL):
    """Equal-weight buy & hold of `syms` from start_i to end."""
    px0 = close_p.iloc[start_i]
    valid = [s for s in syms if not np.isnan(px0.get(s, np.nan))]
    alloc = start_cap / len(valid)
    units = {s: (alloc*(1-FEE))/px0[s] for s in valid}
    eq = []
    for i in range(start_i, len(close_p)):
        px = close_p.iloc[i]
        eq.append(sum(u*px.get(s, px0[s]) for s, u in units.items()))
    return pd.Series(eq, index=close_p.index[start_i:])


def main():
    print(f"\nLoading daily history for {len(BASKET)} coins…")
    data = load_basket()
    print(f"  Got usable history for {len(data)} coins: {', '.join(data.keys())}")
    close_p, up_p, str_p = build_panels(data)

    eq, rebals = run_portfolio(close_p, up_p, str_p)
    m = metrics(eq)
    yrs = m["yrs"]
    print(f"\nTest window: {eq.index[0].date()} → {eq.index[-1].date()}  ({yrs:.1f} years)")
    print(f"Strategy: hold top {TOP_N} trending coins, rebalanced daily, fees {FEE*100:.1f}%/side\n")

    # Benchmarks
    btc_eq = bench_hold(close_p, ["BTC/USD"])
    bm     = metrics(btc_eq)
    basket_eq = bench_hold(close_p, list(data.keys()))
    bk     = metrics(basket_eq)

    rows = [
        ["TREND PORTFOLIO (top 5)",
         f"{m['ret']:+.0f}%", f"{m['cagr']:+.0f}%", f"{m['maxdd']:.0f}%",
         f"{m['calmar']:.2f}", f"{m['sharpe']:.2f}"],
        ["Buy & Hold BTC",
         f"{bm['ret']:+.0f}%", f"{bm['cagr']:+.0f}%", f"{bm['maxdd']:.0f}%",
         f"{bm['calmar']:.2f}", f"{bm['sharpe']:.2f}"],
        ["Buy & Hold basket (equal-wt)",
         f"{bk['ret']:+.0f}%", f"{bk['cagr']:+.0f}%", f"{bk['maxdd']:.0f}%",
         f"{bk['calmar']:.2f}", f"{bk['sharpe']:.2f}"],
    ]
    print(tabulate(rows, headers=["", "TotalRet", "CAGR", "MaxDD", "Calmar", "Sharpe"],
                   tablefmt="rounded_outline"))
    print(f"\nIndividual buy/sell trades: {rebals}")
    print(f"{Fore.YELLOW}Calmar & Sharpe = risk-adjusted return (higher is better). The trend")
    print(f"portfolio's job: capture most of the upside with far smaller drawdowns.{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
