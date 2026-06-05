"""
Strategy tuning harness — tries parameter combinations against the honest,
fee-accurate, leak-free backtest and reports which configuration is profitable.

Usage:  python3 tune.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging
logging.disable(logging.CRITICAL)   # silence training/info logs during tuning

import config
from backtest import run_backtest

# Symbols to evaluate each config on (representative spread).
# BTC = trending major, SOL = high-volatility alt — the two extremes.
SYMBOLS = ["BTC/USD", "SOL/USD"]
DAYS    = 120   # shorter window for fast iteration; winner re-validated on 175

# Candidate configurations to try. Each dict overrides config values.
CANDIDATES = [
    {"name": "baseline (current)",
     "MIN_SIGNAL_SCORE": 5.5, "ATR_STOP_MULTIPLIER": 1.5, "ATR_TARGET_MULTIPLIER": 3.0,
     "MIN_WIN_FEE_MULTIPLE": 3.0},

    {"name": "stricter score 6.5",
     "MIN_SIGNAL_SCORE": 6.5, "ATR_STOP_MULTIPLIER": 1.5, "ATR_TARGET_MULTIPLIER": 3.0,
     "MIN_WIN_FEE_MULTIPLE": 3.0},

    {"name": "wide stop 2.5 / target 6 (RR2.4)",
     "MIN_SIGNAL_SCORE": 6.0, "ATR_STOP_MULTIPLIER": 2.5, "ATR_TARGET_MULTIPLIER": 6.0,
     "MIN_WIN_FEE_MULTIPLE": 3.0},

    {"name": "strict 7 + wide stop 2.5 / tgt 6",
     "MIN_SIGNAL_SCORE": 7.0, "ATR_STOP_MULTIPLIER": 2.5, "ATR_TARGET_MULTIPLIER": 6.0,
     "MIN_WIN_FEE_MULTIPLE": 4.0},

    {"name": "very strict 7.5 + big RR (stop3/tgt9)",
     "MIN_SIGNAL_SCORE": 7.5, "ATR_STOP_MULTIPLIER": 3.0, "ATR_TARGET_MULTIPLIER": 9.0,
     "MIN_WIN_FEE_MULTIPLE": 5.0},
]


def apply(cfg: dict):
    for k, v in cfg.items():
        if k != "name":
            setattr(config, k, v)
    # Keep MIN_RR in backtest aligned with target/stop ratio
    import backtest
    backtest.MIN_RR = min(2.0, cfg["ATR_TARGET_MULTIPLIER"] / cfg["ATR_STOP_MULTIPLIER"] - 0.01)


def main():
    print(f"\nTuning over {len(SYMBOLS)} symbols × {DAYS} days "
          f"(honest: no ML, fees on)\n" + "=" * 70)
    for cfg in CANDIDATES:
        apply(cfg)
        rows, total_pnl, total_trades = [], 0.0, 0
        for sym in SYMBOLS:
            r = run_backtest(sym, "1h", DAYS, 1000, config.RISK_PER_TRADE_PCT, verbose=False)
            if not r or r.get("trades", 0) == 0:
                rows.append((sym, 0, 0.0, 0.0)); continue
            rows.append((sym, r["trades"], r["return_pct"], r["profit_factor"]))
            total_pnl += r["total_pnl"]; total_trades += r["trades"]
        print(f"\n▶ {cfg['name']}")
        for sym, n, ret, pf in rows:
            flag = "✅" if ret > 0 else "❌"
            print(f"   {flag} {sym:9s}  {n:3d} trades  return {ret:+7.1f}%  PF {pf:.2f}")
        avg_ret = sum(r[2] for r in rows) / len(rows)
        print(f"   ── avg return {avg_ret:+.1f}%  |  total {total_trades} trades  |  net $ {total_pnl:+.0f}")


if __name__ == "__main__":
    main()
