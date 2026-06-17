"""
CARRY HARVESTER — PHASE 0 FEASIBILITY (build-or-kill), MULTI-YEAR.

Question: is a DELTA-NEUTRAL carry (long spot + short perp) actually profitable
NET of cost, and does it survive across regimes (2021 bull → 2022 bear → 2023-25)?
We do NOT build anything until this says yes.

Mechanics: long spot + short perp of equal notional is ~price-neutral. The short
perp EARNS funding each period when funding>0 (longs pay shorts), PAYS when <0.
Realized carry for a year = SUM of funding rates over that year (each rate is the
per-period return on notional while the position is held). Summing the realized
rates is interval-agnostic (Binance shifted 8h→4h), so it's the true carry.

Data: Binance funding-rate monthly dumps from data.binance.vision (the public CDN,
NOT the geo-blocked API) — full history back to 2020. Fees: maker round-trip ALL
legs ~0.20% one-time (amortized over a long hold = negligible). Honest risks
(counterparty, funding-flip, liquidation) flagged in the writeup, not modeled away.
"""
import sys, io, os, zipfile, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, requests

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
CACHE = "/tmp/binance_funding"; os.makedirs(CACHE, exist_ok=True)
RT_FEE = 0.0020          # one-time round-trip, all legs, maker (~0.20%)
BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"


def fetch_sym(sym):
    frames = []
    for yr in range(2020, 2027):
        for mo in range(1, 13):
            if yr == 2026 and mo > 6:
                break
            tag = f"{sym}-fundingRate-{yr}-{mo:02d}"
            fp = f"{CACHE}/{tag}.csv"
            if os.path.exists(fp):
                if os.path.getsize(fp) > 0:
                    frames.append(pd.read_csv(fp))
                continue
            url = f"{BASE}/{sym}/{tag}.zip"
            try:
                r = requests.get(url, timeout=30)
                if r.status_code != 200:
                    open(fp, "w").close(); continue       # cache the miss
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]))
                df.to_csv(fp, index=False); frames.append(df)
            except Exception:
                open(fp, "w").close()
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["calc_time"], unit="ms")
    return df.set_index("date")["last_funding_rate"].astype(float).sort_index()


def main():
    print("CARRY FEASIBILITY — delta-neutral funding harvest (Binance dumps, full history)\n")
    summary = {}
    for sym in SYMS:
        tok = sym.replace("USDT", "")
        f = fetch_sym(sym)
        if f is None or len(f) < 100:
            print(f"{tok}: no data"); continue
        print(f"━━━ {tok}  ({f.index.min().date()} → {f.index.max().date()}, "
              f"{len(f)} periods)")
        rows = []
        for yr, g in f.groupby(f.index.year):
            days = max((g.index.max() - g.index.min()).days, 1)
            gross = g.sum()
            ann = gross * 365.0 / days                  # annualize partial years
            smart = g[g > 0].sum() * 365.0 / days       # harvest-positive-only
            pos = (g > 0).mean()
            neg = (g < 0).astype(int).values
            run = maxrun = 0
            for v in neg:
                run = run + 1 if v else 0; maxrun = max(maxrun, run)
            rows.append((yr, gross*100, ann*100, smart*100, pos*100, maxrun))
        R = pd.DataFrame(rows, columns=["yr","gross%","ann%","smart%","%pos","negRun"])
        for _, r in R.iterrows():
            tag = " (partial)" if int(r["yr"]) in (f.index.min().year, 2026) else ""
            print(f"   {int(r['yr'])}: realized {r['gross%']:+6.2f}%  annualized {r['ann%']:+6.2f}%  "
                  f"| smart(+only) {r['smart%']:+6.2f}%  | {r['%pos']:4.0f}% pos  "
                  f"| worst neg streak {int(r['negRun'])}p{tag}")
        full_days = max((f.index.max() - f.index.min()).days, 1)
        ann_all = f.sum() * 365.0 / full_days
        smart_all = f[f > 0].sum() * 365.0 / full_days
        print(f"   FULL: annualized always-on {ann_all*100:+.2f}%  net(−fees) {(ann_all-RT_FEE)*100:+.2f}%  "
              f"| smart {smart_all*100:+.2f}%  | {(f>0).mean()*100:.0f}% periods positive\n")
        summary[tok] = (ann_all*100, smart_all*100, (f>0).mean()*100,
                        R.set_index("yr")["ann%"].to_dict(),
                        R.set_index("yr")["smart%"].to_dict())

    print("════════ VERDICT INPUTS ════════")
    for tok, (g, s, p, byyr, smartyr) in summary.items():
        worst = min(byyr.values()); best = max(byyr.values())
        worst_smart = min(smartyr.values())
        print(f"  {tok}: always-on ann {g:+.1f}%  smart {s:+.1f}%  | {p:.0f}% pos  | "
              f"worst yr {worst:+.1f}% (smart {worst_smart:+.1f}%)  best yr {best:+.1f}%")
    print("\nRead: carry is REAL if always-on annualized is solidly positive across MOST")
    print("regimes, and the 'smart' (sit-out-negative-funding) variant keeps the worst")
    print("years from going deeply negative. Watch 2022 (bear) — that's the real test.")


if __name__ == "__main__":
    main()
