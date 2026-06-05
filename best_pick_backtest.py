"""
"Best single pick per run" backtest — tests the intuition directly:
  "With many coins scanned, surely the SINGLE highest-scoring one each run
   is profitable."

Each hour, when flat, scan EVERY coin in the basket, compute the rule-based
signal score (no ML, no look-ahead), and enter the single best actionable
coin. Manage it with the same ATR stop/target. One position at a time —
always the best opportunity available across the whole universe. Fees on.

If the intuition is right, this is reliably profitable. Let's see.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

import config
from exchange.market_data import fetch_ohlcv
from core.indicators import enrich
from core.strategies import analyse

FEE   = getattr(config, "FEE_RATE_PCT", 0.6) / 100.0
DAYS  = 90
CAP   = 1000.0
RISK  = config.RISK_PER_TRADE_PCT / 100.0
WARMUP = 205

BASKET = ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","ADA/USD","LINK/USD","AVAX/USD"]


def round_trip_fee(entry, exit_, qty):
    return (entry*qty*FEE) + (exit_*qty*FEE)


def main():
    print(f"\nLoading 1h history for {len(BASKET)} coins (~{DAYS} days each)…")
    data = {}
    for s in BASKET:
        df = fetch_ohlcv(s, "1h", limit=DAYS*24+300)
        if not df.empty and len(df) > 400:
            data[s] = enrich(df)
    syms = list(data.keys())
    # Align on common length (use the shortest)
    n = min(len(df) for df in data.values())
    for s in syms:
        data[s] = data[s].iloc[-n:]
    print(f"  Usable: {len(syms)} coins, {n} candles "
          f"({(data[syms[0]].index[-1]-data[syms[0]].index[0]).days} days)")

    cap = CAP
    in_trade = False
    held = None
    entry = stop = target = qty = 0.0
    side = ""
    trades = []
    t0 = time.time()

    for i in range(WARMUP, n):
        # ── Exit check (cheap — only the held coin) ───────────────────────────
        if in_trade:
            cur = data[held].iloc[i]
            hi, lo = cur["high"], cur["low"]
            hit_sl = (side=="long" and lo<=stop) or (side=="short" and hi>=stop)
            hit_tp = (side=="long" and hi>=target) or (side=="short" and lo<=target)
            if hit_sl or hit_tp:
                ep = target if hit_tp else stop
                gross = (ep-entry)*qty if side=="long" else (entry-ep)*qty
                cap += gross - round_trip_fee(entry, ep, qty)
                trades.append(gross - round_trip_fee(entry, ep, qty))
                in_trade = False; held = None

        # ── Entry: when flat, scan ALL coins, pick the single best ────────────
        if not in_trade:
            best = None   # (score, sym, signal)
            for s in syms:
                sl = data[s].iloc[max(0,i-300):i]
                if len(sl) < 100:
                    continue
                try:
                    sig = analyse(s, sl, include_sentiment=False, include_ml=False)
                except Exception:
                    continue
                if sig.direction in ("long","short") and sig.score >= config.MIN_SIGNAL_SCORE and sig.atr>0:
                    if best is None or sig.score > best[0]:
                        best = (sig.score, s, sig)
            if best is not None:
                _, s, sig = best
                price = data[s].iloc[i]["close"]
                atr = sig.atr
                if sig.direction=="long":
                    st = price - config.ATR_STOP_MULTIPLIER*atr
                    tg = price + config.ATR_TARGET_MULTIPLIER*atr
                else:
                    st = price + config.ATR_STOP_MULTIPLIER*atr
                    tg = price - config.ATR_TARGET_MULTIPLIER*atr
                rpu = abs(price-st)
                if rpu>0:
                    q = (cap*RISK)/rpu
                    gross_tp = abs(tg-price)*q
                    fee_tp = round_trip_fee(price, tg, q)
                    # profit-floor parity
                    if gross_tp - fee_tp >= getattr(config,"MIN_NET_PROFIT_USD",1.0) \
                       and gross_tp >= fee_tp*getattr(config,"MIN_WIN_FEE_MULTIPLE",3.0):
                        held, entry, stop, target, qty, side = s, price, st, tg, q, sig.direction
                        in_trade = True

    # Stats
    wins=[t for t in trades if t>0]; losses=[t for t in trades if t<=0]
    wr = len(wins)/len(trades)*100 if trades else 0
    pf = sum(wins)/abs(sum(losses)) if losses else float("inf")
    ret = (cap-CAP)/CAP*100

    print(f"\n{'='*60}")
    print(f"  BEST SINGLE PICK PER RUN — {len(syms)} coins scanned each hour")
    print(f"{'='*60}")
    print(f"  Start capital : ${CAP:.2f}")
    print(f"  End capital   : ${cap:.2f}")
    print(f"  Total return  : {ret:+.1f}%")
    print(f"  Trades        : {len(trades)}")
    print(f"  Win rate      : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  (ran in {time.time()-t0:.0f}s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
