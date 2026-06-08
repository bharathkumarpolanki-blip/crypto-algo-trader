"""
UNIFIED DEPLOYABILITY GATE.

Any strategy must pass this before paper trading. Runs the robustness battery
distilled from the 15-phase audit and prints a PASS/FAIL scorecard against fixed,
non-negotiable criteria. No optimization — this is a judge, not a tuner.

Pass criteria (all must hold):
  • Out-of-sample PF      > 1.20
  • Out-of-sample CAGR    > 0
  • Sharpe (full sample)  > 1.00
  • Cost-robust           (still profitable at 2x fees)
  • Walk-forward stable    (>= 60% of test years positive)

Usage: python3 validate.py                # validates the rotation strategy
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)

import numpy as np
import core.rotation_strategy as R

GATES = {
    "oos_pf":      ("Out-of-sample PF > 1.20",        lambda s: s["oos_pf"]   > 1.20),
    "oos_cagr":    ("Out-of-sample CAGR > 0",          lambda s: s["oos_cagr"] > 0),
    "sharpe":      ("Full-sample Sharpe > 1.00",       lambda s: s["sharpe"]   > 1.00),
    "cost":        ("Profitable at 2x fees",           lambda s: s["cost2x_cagr"] > 0),
    "walkfwd":     ("≥60% test years positive",        lambda s: s["wf_pos_frac"] >= 0.60),
}


def evaluate(label, runner):
    """runner(start,end,fee_mult)->metrics dict. Builds the scorecard."""
    print(f"\n{'='*72}\n  VALIDATING: {label}\n{'='*72}")
    full = runner(None, None, 1.0)
    if "error" in full:
        print("  ERROR:", full["error"]); return
    print(f"  Full sample ({full['years']}y): ret {full['ret']:+.0f}%  CAGR {full['cagr']:+.1f}%  "
          f"Sharpe {full['sharpe']:+.2f}  Sortino {full['sortino']:+.2f}  PF {full['pf']:.2f}  "
          f"MaxDD {full['maxdd']:.0f}%  trades {full['trades_n']}")

    def safe(r):
        return r if (isinstance(r, dict) and "error" not in r and "cagr" in r) else \
            {"cagr": -100, "sharpe": 0, "pf": 0, "maxdd": 0, "ret": 0, "trades_n": 0}

    # Out-of-sample: newest 20% (no tuning — same fixed params)
    eq = full["equity"]; cut = eq.index[int(len(eq)*0.8)]
    oos = safe(runner(str(cut.date()), None, 1.0))
    print(f"  Out-of-sample (from {cut.date()}): CAGR {oos['cagr']:+.1f}%  Sharpe {oos['sharpe']:+.2f}  "
          f"PF {oos['pf']:.2f}  MaxDD {oos['maxdd']:.0f}%")

    # Cost stress 2x
    c2 = safe(runner(None, None, 2.0))

    # Walk-forward by calendar year
    years = sorted(set(eq.index.year))[1:]   # skip first (warmup-heavy)
    wf = []
    for y in years:
        r = runner(f"{y}-01-01", f"{y+1}-01-01", 1.0)
        if "error" in r or r["trades_n"] == 0:
            continue
        wf.append((y, r["ret"]))
    pos_frac = (sum(1 for _, rr in wf if rr > 0) / len(wf)) if wf else 0
    print(f"  Walk-forward years: " + "  ".join(f"{y}:{rr:+.0f}%" for y, rr in wf))

    s = {"sharpe": full["sharpe"], "oos_pf": oos["pf"], "oos_cagr": oos["cagr"],
         "cost2x_cagr": c2["cagr"], "wf_pos_frac": pos_frac}

    print(f"\n  {'CRITERION':34s} {'RESULT':>12s}   VERDICT")
    print("  " + "-"*60)
    allpass = True
    detail = {"oos_pf": f"{oos['pf']:.2f}", "oos_cagr": f"{oos['cagr']:+.1f}%",
              "sharpe": f"{full['sharpe']:+.2f}", "cost2x_cagr": f"{c2['cagr']:+.1f}%",
              "walkfwd": f"{pos_frac*100:.0f}%"}
    dmap = {"oos_pf":"oos_pf","oos_cagr":"oos_cagr","sharpe":"sharpe","cost":"cost2x_cagr","walkfwd":"walkfwd"}
    for key, (desc, fn) in GATES.items():
        ok = fn(s); allpass &= ok
        print(f"  {desc:34s} {detail[dmap[key]]:>12s}   {'✅ PASS' if ok else '❌ FAIL'}")
    print("  " + "-"*60)
    print(f"  OVERALL: {'✅ ALL GATES PASSED → paper-trading eligible' if allpass else '❌ FAILED → NOT deployable'}")
    return allpass


def main():
    # Variant runners — same strategy, different breadth (no param tuning, just structure)
    def make(top_n, regime):
        return lambda start, end, fee_mult: R.backtest(
            top_n=top_n, use_regime_gate=regime, start=start, end=end, fee_mult=fee_mult)

    bh = R.buy_hold("BTC/USD")
    print(f"Benchmark — BTC Buy&Hold: CAGR {bh.get('cagr',0):+.1f}%  Sharpe {bh.get('sharpe',0):+.2f}  "
          f"MaxDD {bh.get('maxdd',0):.0f}%")

    evaluate("Rotation Top-2 + Trend + Regime gate", make(2, True))
    evaluate("Rotation Top-2 + Trend (NO regime gate)", make(2, False))
    evaluate("Rotation Top-3 + Trend + Regime gate", make(3, True))


if __name__ == "__main__":
    main()
