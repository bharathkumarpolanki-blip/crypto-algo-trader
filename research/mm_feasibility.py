"""
MARKET-MAKING — PHASE 0 FEASIBILITY (build-or-kill).

The make-or-break for retail MM: is the SPREAD you can capture bigger than
FEES + ADVERSE SELECTION? You quote a bid below mid and an ask above mid; when
both fill you bank the spread minus fees. But you get "adversely selected" — your
bid fills right as price drops, your ask fills right as price rises — so realized
edge = spread − 2×maker_fee − adverse_selection.

This study measures the two things we CAN measure from public data:
  1. Top-of-book SPREAD (bps) per pair  → the gross room.
  2. Short-horizon VOLATILITY (mid moves over ~60s) → an adverse-selection proxy:
     if the mid routinely jumps more than half the spread before you can offload,
     you're picked off and MM loses.

Gate 1 (necessary): spread > 2×maker_fee. If fees alone eat the spread → DEAD.
Gate 2 (the hard one): half-spread > short-horizon move. If price moves more than
your edge before you can flatten, adverse selection eats you → DEAD.

Venue: Hyperliquid (lowest maker fee we can reach; Coinbase retail maker ~0.4-0.6%
is fatal for tight-spread MM before we even start). Read-only, keyless.
"""
import sys, time, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, ccxt

# A spread of pairs: liquid majors → mid-caps → smaller alts (wider spreads).
TOKENS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK",
          "ARB", "OP", "SUI", "LTC", "WLD", "TIA", "SEI", "JUP"]
SAMPLES = 10          # mid-price sampling rounds
INTERVAL = 3          # seconds between rounds (round-robin over all pairs ≈ 1 window)


def main():
    hl = ccxt.hyperliquid({"enableRateLimit": True})
    mk = hl.load_markets()

    # real maker/taker fees from the venue (not guessed)
    sample_mkt = next((mk[s] for s in mk if s == "BTC/USDC:USDC"), None)
    maker = (sample_mkt or {}).get("maker", 0.00015)
    taker = (sample_mkt or {}).get("taker", 0.00045)
    rt_fee_bps = 2 * maker * 1e4
    print(f"Hyperliquid fees: maker {maker*1e4:.2f}bps  taker {taker*1e4:.2f}bps  "
          f"→ round-trip maker cost {rt_fee_bps:.2f}bps\n")

    # round-robin sample: every round, hit ALL pairs → total wall time ≈ 1 window,
    # not (pairs × window). mids/spreads collected per token.
    syms = [f"{t}/USDC:USDC" for t in TOKENS if f"{t}/USDC:USDC" in mk]
    mids = {s: [] for s in syms}
    spr = {s: [] for s in syms}
    for r in range(SAMPLES):
        for s in syms:
            try:
                ob = hl.fetch_order_book(s, limit=5)
                if ob["bids"] and ob["asks"]:
                    bid, ask = ob["bids"][0][0], ob["asks"][0][0]
                    m = (bid + ask) / 2
                    mids[s].append(m)
                    spr[s].append((ask - bid) / m * 1e4)
            except Exception:
                pass
        print(f"  ...sampled round {r+1}/{SAMPLES}", flush=True)
        time.sleep(INTERVAL)

    print(f"\n{'PAIR':14} {'spread':>8} {'rt_fee':>7} {'gross':>8} | "
          f"{'~step move':>10} {'½spread':>8} {'verdict':>22}")
    print("-" * 92)
    rows = []
    for s in syms:
        if len(mids[s]) < SAMPLES // 2:
            continue
        tok = s.split("/")[0]
        m = np.array(mids[s])
        spread_bps = float(np.median(spr[s]))
        half_spread = spread_bps / 2
        rets = np.diff(m) / m[:-1] * 1e4
        move_bps = float(np.std(rets))                 # typical mid move per step (bps)
        gross = spread_bps - rt_fee_bps
        if gross <= 0:
            verdict = "DEAD (fees eat spread)"
        elif half_spread <= move_bps:
            verdict = "DEAD (adverse selection)"
        elif half_spread < 2 * move_bps:
            verdict = "marginal"
        else:
            verdict = "room — worth a sim"
        rows.append((tok, spread_bps, gross, move_bps, half_spread, verdict))
        print(f"{s:14} {spread_bps:7.1f}b {rt_fee_bps:6.1f}b {gross:7.1f}b | "
              f"{move_bps:9.1f}b {half_spread:7.1f}b {verdict:>22}")

    print("\n" + "=" * 90)
    survivors = [r for r in rows if "room" in r[5] or "marginal" in r[5]]
    dead = [r for r in rows if "DEAD" in r[5]]
    print(f"  {len(dead)}/{len(rows)} pairs DEAD on fees or adverse selection.")
    if survivors:
        print(f"  Possible room ({len(survivors)}): "
              + ", ".join(f"{r[0]}({r[1]:.0f}b spread, {r[3]:.0f}b move)" for r in survivors))
        print("  → these have GROSS room, but adverse selection is only proxied here.")
        print("     A live paper-quoting sim is needed to know REALIZED edge.")
    else:
        print("  No pair has room net of fees + short-horizon move → retail MM looks DEAD here.")
    print("\n  Honest caveats this snapshot can't capture: queue position (you wait behind")
    print("  faster quotes), inventory risk on trends, and pro-HFT competition. All make")
    print("  the REAL edge LOWER than the gross room above, never higher.")


if __name__ == "__main__":
    main()
