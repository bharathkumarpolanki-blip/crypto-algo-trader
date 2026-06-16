"""
TOKEN-UNLOCK REAL-DATE CONFIRMATION HARNESS.

The free study found a significant unlock-DAY effect (−1.6% abnormal vs BTC,
placebo p=0.0000) — but using unlock dates DETECTED from CoinGecko supply jumps,
which could be a measurement-timing artifact. This harness re-runs the decisive
day-effect placebo using REAL scheduled unlock dates (from DefiLlama).

USAGE once you have DefiLlama data:
  Export the unlock schedule and save it as:
      research/data/unlock_dates.csv
  with columns:  cg_id,date          (one row per unlock event)
      cg_id = the CoinGecko id (e.g. "arbitrum", "optimism", "aptos")
      date  = the unlock date, YYYY-MM-DD
  Then run:  python3 research/token_unlock_confirm.py

It reuses the cached free CoinGecko prices in /tmp/cg_unlocks (last ~365d) and
ccxt BTC, and reports the day-effect placebo on REAL dates. If real dates also
give ~−1.6% (p<0.05) → the effect is REAL, not a supply-detection artifact.
If real dates give ~0 → it was an artifact, and the lead is dead.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from exchange.market_data import fetch_ohlcv

CACHE = "/tmp/cg_unlocks"
CSV = os.path.join(os.path.dirname(__file__), "data", "unlock_dates.csv")
rng = np.random.default_rng(0)


def load_prices():
    px = {}
    if not os.path.isdir(CACHE): return px
    for fn in os.listdir(CACHE):
        if not fn.endswith(".csv"): continue
        cid = fn[:-4]
        try:
            df = pd.read_csv(f"{CACHE}/{fn}", parse_dates=["date"]).set_index("date")
            if len(df) > 60: px[cid] = df
        except Exception: pass
    return px


def day_placebo(real, rand, label):
    real, rand = np.array(real), np.array(rand)
    if len(real) < 10:
        print(f"  {label}: only {len(real)} events — too few."); return
    diff = real.mean() - rand.mean(); allv = np.concatenate([real, rand])
    null = [(lambda s: s[:len(real)].mean()-s[len(real):].mean())(rng.permutation(allv)) for _ in range(5000)]
    p = (np.array(null) <= diff).mean()
    print(f"  {label}")
    print(f"    real unlock days {real.mean()*100:+.2f}% (n={len(real)})  vs random {rand.mean()*100:+.2f}% (n={len(rand)})")
    print(f"    difference {diff*100:+.2f}%   permutation p={p:.4f}  "
          f"{'✅ REAL (unlock-specific)' if p<0.05 else '❌ artifact / not unlock-specific'}")


def main():
    px = load_prices()
    if not px:
        print("No cached CoinGecko prices in /tmp/cg_unlocks — run token_unlock_full.py first.")
        return
    btc = fetch_ohlcv("BTC/USD","1d",limit=400)["close"]; btc.index = btc.index.tz_localize(None).normalize()
    bret = btc.pct_change()

    if not os.path.exists(CSV):
        print(f"REAL-DATE confirmation: drop your DefiLlama unlock dates at:\n  {CSV}")
        print("  columns: cg_id,date   (e.g.  arbitrum,2024-03-16)\n")
        print("Until then, here is the SUPPLY-JUMP-DETECTED baseline (what we have for free):")
        real, rand = [], []
        for cid, df in px.items():
            df = df.copy(); df["d"] = df["supply"].pct_change()
            abn = df["price"].pct_change() - bret.reindex(df.index)
            uds = df.index[df["d"] > 0.015]
            for d in uds:
                v = abn.get(d);  real.append(v) if (v is not None and not np.isnan(v)) else None
            bad = set()
            for d in uds:
                i = df.index.get_loc(d)
                for k in (-1,0,1):
                    if 0 <= i+k < len(df): bad.add(df.index[i+k])
            pool = [d for d in df.index if d not in bad]
            for _ in range(len(uds)*20):
                v = abn.get(pool[int(rng.integers(len(pool)))])
                if v is not None and not np.isnan(v): rand.append(v)
        day_placebo(real, rand, "SUPPLY-JUMP-DETECTED dates (free baseline)")
        return

    # ── REAL DATES present → the decisive confirmation ────────────────────────
    ud = pd.read_csv(CSV); ud["date"] = pd.to_datetime(ud["date"]).dt.normalize()
    print(f"REAL-DATE CONFIRMATION  ({len(ud)} unlock events from {CSV})\n")
    real, rand = [], []
    matched = 0
    for cid, grp in ud.groupby("cg_id"):
        if cid not in px: continue
        df = px[cid]; abn = df["price"].pct_change() - bret.reindex(df.index)
        evset = set(d for d in grp["date"] if d in df.index)
        matched += len(evset)
        for d in evset:
            v = abn.get(d)
            if v is not None and not np.isnan(v): real.append(v)
        bad = set()
        for d in evset:
            i = df.index.get_loc(d)
            for k in (-1,0,1):
                if 0 <= i+k < len(df): bad.add(df.index[i+k])
        pool = [d for d in df.index if d not in bad]
        if not pool: continue
        for _ in range(max(len(evset),1)*20):
            v = abn.get(pool[int(rng.integers(len(pool)))])
            if v is not None and not np.isnan(v): rand.append(v)
    print(f"  matched {matched} unlock events to cached price history (last ~365d window)\n")
    day_placebo(real, rand, "REAL scheduled unlock dates")
    print("\n→ If REAL dates give ~−1.6% with p<0.05, the effect is confirmed (not a")
    print("  supply-detection artifact) and worth a tradability study. If ~0, it was an artifact.")


if __name__ == "__main__":
    main()
