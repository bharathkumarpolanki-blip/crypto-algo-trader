"""
Walk-forward backtest using historical OHLCV data.

Usage:
    python3 backtest.py --symbol BTC/USD --days 90
    python3 backtest.py --symbol ETH/USD --days 180
    python3 backtest.py --symbol BTC/USD --days 90 --multi   # run all watchlist symbols

Fetches live history via CCXT public endpoints (no API key needed).
Runs the full multi-strategy + 4h trend-gate signal engine on each 1h candle
in a rolling window, and reports P&L, win rate, Sharpe ratio, and max drawdown.
"""

import argparse
import sys
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

import config
from market_data import fetch_ohlcv
from indicators import enrich
from strategies import analyse, trend_direction_4h

colorama_init(autoreset=True)
logging.basicConfig(level=logging.WARNING)

MIN_RR = 2.0   # only enter if R:R >= 2.0 (matches ATR_TARGET/ATR_STOP = 4/1.5 ≈ 2.67)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days",      type=int, default=90)
    p.add_argument("--capital",   type=float, default=config.TOTAL_CAPITAL_USDT)
    p.add_argument("--risk-pct",  type=float, default=config.RISK_PER_TRADE_PCT)
    p.add_argument("--multi",     action="store_true", help="Run all WATCHLIST symbols")
    return p.parse_args()


def run_backtest(symbol: str, timeframe: str, days: int,
                 capital: float, risk_pct: float, verbose: bool = True) -> dict:

    # Fetch 1h data — need more candles than days*24 so indicators warm up
    limit_1h = min(days * 24 + 300, 1000)
    # Fetch 4h data for the trend gate
    limit_4h = min(days * 6 + 100, 500)

    if verbose:
        print(f"\nFetching {symbol} — {limit_1h} × 1h candles + {limit_4h} × 4h candles...")

    df1h = fetch_ohlcv(symbol, "1h",            limit=limit_1h)
    df4h = fetch_ohlcv(symbol, config.TF_TREND, limit=limit_4h)

    if df1h.empty or len(df1h) < 150:
        if verbose: print("Not enough 1h data.")
        return {}

    df1h = enrich(df1h)
    df4h = enrich(df4h) if not df4h.empty else pd.DataFrame()

    if verbose:
        print(f"Running strategy engine on {len(df1h)} 1h candles (warm-up window = 100)...\n")

    window = 100
    trades: list[dict] = []
    capital_curve: list[float] = [capital]
    curr_capital = capital
    in_trade = False
    entry = stop = target = qty = 0.0
    trade_side = ""
    consecutive_losses = 0

    for i in range(window, len(df1h)):
        slice_1h = df1h.iloc[:i]
        current  = df1h.iloc[i]
        price    = current["close"]

        # Map current 1h timestamp → most recent 4h slice
        ts_1h = df1h.index[i]
        if not df4h.empty:
            slice_4h = df4h[df4h.index <= ts_1h]
            slice_4h = enrich(slice_4h) if len(slice_4h) >= 60 else pd.DataFrame()
        else:
            slice_4h = pd.DataFrame()

        # ── Check exit ────────────────────────────────────────────────────────
        if in_trade:
            # Use high/low of the candle for realistic fill
            candle_high = current["high"]
            candle_low  = current["low"]

            hit_stop = (trade_side == "long"  and candle_low  <= stop)   or \
                       (trade_side == "short" and candle_high >= stop)
            hit_tp   = (trade_side == "long"  and candle_high >= target) or \
                       (trade_side == "short" and candle_low  <= target)

            if hit_tp or hit_stop:
                exit_price = target if hit_tp else stop
                pnl = (exit_price - entry) * qty if trade_side == "long" \
                      else (entry - exit_price) * qty
                curr_capital += pnl
                reason = "TP" if hit_tp else "SL"
                trades.append({
                    "exit_time": str(current.name)[:19],   # Timestamp → plain string
                    "side": trade_side, "entry": entry,
                    "exit": exit_price, "pnl": pnl,
                    "reason": reason, "capital": curr_capital,
                })
                capital_curve.append(curr_capital)
                in_trade = False
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

        # ── Check entry ───────────────────────────────────────────────────────
        if not in_trade:
            # Cooldown: skip 6 candles after 2+ consecutive losses
            if consecutive_losses >= 2:
                consecutive_losses -= 1
                continue

            signal = analyse(
                symbol, slice_1h,
                df_trend=slice_4h if len(slice_4h) >= 60 else None,
                include_sentiment=False,
            )

            if (signal.direction in ("long", "short")
                    and signal.score >= config.MIN_SIGNAL_SCORE
                    and signal.atr > 0):

                # Always compute fresh TP/SL from ACTUAL entry price (not signal's stale price)
                atr = signal.atr
                if signal.direction == "long":
                    fresh_stop   = price - config.ATR_STOP_MULTIPLIER   * atr
                    fresh_target = price + config.ATR_TARGET_MULTIPLIER * atr
                else:
                    fresh_stop   = price + config.ATR_STOP_MULTIPLIER   * atr
                    fresh_target = price - config.ATR_TARGET_MULTIPLIER * atr

                fresh_rr = abs(fresh_target - price) / abs(fresh_stop - price)

                if fresh_rr >= MIN_RR:
                    risk_usdt     = curr_capital * (risk_pct / 100)
                    risk_per_unit = abs(price - fresh_stop)
                    if risk_per_unit > 0:
                        entry      = price
                        stop       = fresh_stop
                        target     = fresh_target
                        qty        = risk_usdt / risk_per_unit
                        trade_side = signal.direction
                        in_trade   = True

    # Close any open trade at last price
    if in_trade:
        last_price = df1h.iloc[-1]["close"]
        pnl = (last_price - entry) * qty if trade_side == "long" else (entry - last_price) * qty
        curr_capital += pnl
        trades.append({
            "exit_time": str(df1h.index[-1])[:19],   # Timestamp → plain string
            "side": trade_side, "entry": entry,
            "exit": last_price, "pnl": pnl,
            "reason": "EOD", "capital": curr_capital,
        })
        capital_curve.append(curr_capital)

    # ── Summary stats ─────────────────────────────────────────────────────────
    if not trades:
        if verbose: print("No trades generated in this period.")
        return {"symbol": symbol, "trades": 0, "return_pct": 0}

    total_pnl = sum(t["pnl"] for t in trades)
    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    win_rate  = len(wins) / len(trades) * 100
    avg_win   = np.mean([t["pnl"] for t in wins])   if wins   else 0
    avg_loss  = np.mean([t["pnl"] for t in losses]) if losses else 0
    pf_denom  = abs(sum(t["pnl"] for t in losses))
    profit_factor = sum(t["pnl"] for t in wins) / pf_denom if pf_denom > 0 else float("inf")

    pnl_series    = pd.Series([t["pnl"] for t in trades])
    sharpe        = (pnl_series.mean() / pnl_series.std() * np.sqrt(252)) \
                    if pnl_series.std() > 0 else 0

    caps       = pd.Series(capital_curve)
    peak       = caps.cummax()
    drawdown   = (caps - peak) / peak * 100
    max_dd     = drawdown.min()
    total_ret  = (curr_capital - capital) / capital * 100

    result = {
        "symbol": symbol, "days": days,
        "start_capital": capital, "end_capital": round(curr_capital, 2),
        "return_pct": round(total_ret, 2), "total_pnl": round(total_pnl, 2),
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(win_rate, 1), "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2), "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 2),
        "trade_list": trades,
    }

    if verbose:
        _print_result(result)

    return result


def _print_result(r: dict) -> None:
    ret_color = Fore.GREEN if r["return_pct"] >= 0 else Fore.RED
    print("=" * 62)
    print(f"  BACKTEST RESULTS — {r['symbol']}  ({r['days']} days, 1h)")
    print("=" * 62)
    stats = [
        ["Starting Capital",  f"${r['start_capital']:.2f}"],
        ["Ending Capital",    f"${r['end_capital']:.2f}"],
        ["Total Return",      ret_color + f"{r['return_pct']:+.2f}%" + Style.RESET_ALL],
        ["Total P&L",         ret_color + f"${r['total_pnl']:+.2f}" + Style.RESET_ALL],
        ["Total Trades",      r["trades"]],
        ["Win Rate",          f"{r['win_rate']:.1f}%"],
        ["Wins / Losses",     f"{r['wins']} / {r['losses']}"],
        ["Avg Win",           f"${r['avg_win']:.2f}"],
        ["Avg Loss",          f"${r['avg_loss']:.2f}"],
        ["Profit Factor",     f"{r['profit_factor']:.2f}"],
        ["Sharpe Ratio",      f"{r['sharpe']:.2f}"],
        ["Max Drawdown",      f"{r['max_dd']:.2f}%"],
    ]
    print(tabulate(stats, tablefmt="rounded_outline"))

    last_n = r["trade_list"][-10:]
    if last_n:
        print("\nLast 10 Trades:")
        rows = []
        for t in last_n:
            pnl_str = (Fore.GREEN if t["pnl"] > 0 else Fore.RED) + f"{t['pnl']:+.2f}" + Style.RESET_ALL
            rows.append([str(t["exit_time"])[:16], t["side"].upper(),
                         f"{t['entry']:.2f}", f"{t['exit']:.2f}", t["reason"], pnl_str])
        print(tabulate(rows, headers=["Time","Side","Entry","Exit","Reason","PnL"],
                       tablefmt="rounded_outline"))
    print()


if __name__ == "__main__":
    args = parse_args()

    if args.multi:
        # Use dynamic universe if bot has been run at least once, else fall back to config.WATCHLIST
        try:
            import universe
            symbols_to_test = universe.get_watchlist()
            print(f"Using dynamic universe ({len(symbols_to_test)} symbols): {', '.join(symbols_to_test)}")
        except Exception:
            symbols_to_test = config.WATCHLIST
            print(f"Using static watchlist ({len(symbols_to_test)} symbols)")

        results = []
        for sym in symbols_to_test:
            print(f"Backtesting {sym}...", end=" ", flush=True)
            r = run_backtest(sym, args.timeframe, args.days,
                             args.capital, args.risk_pct, verbose=False)
            if r:
                results.append(r)
                ret_color = Fore.GREEN if r["return_pct"] >= 0 else Fore.RED
                print(ret_color + f"{r['return_pct']:+.1f}%" + Style.RESET_ALL +
                      f"  ({r['trades']} trades, PF={r['profit_factor']})")
            else:
                print("no data")

        if results:
            print("\n── Portfolio Summary ──────────────────────────────────────")
            summary_rows = sorted(
                [[r["symbol"], f"{r['return_pct']:+.1f}%", r["trades"],
                  f"{r['win_rate']:.0f}%", r["profit_factor"], f"{r['max_dd']:.1f}%"]
                 for r in results],
                key=lambda x: float(x[1].replace("%", "").replace("+", "")), reverse=True
            )
            print(tabulate(summary_rows,
                           headers=["Symbol","Return","Trades","WinRate","PF","MaxDD"],
                           tablefmt="rounded_outline"))
    else:
        run_backtest(args.symbol, args.timeframe, args.days,
                     args.capital, args.risk_pct, verbose=True)