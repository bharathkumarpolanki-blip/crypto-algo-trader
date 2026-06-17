"""
TOKEN-UNLOCK CROSS-VALIDATION — DefiLlama-sourced, multi-year (2023-2026).

The free CoinGecko study (365d cap) found an unlock-DAY effect: tokens drop
~-1.5% vs BTC specifically on supply-unlock days (placebo p=0.0000, held-out
p=0.015). Two open worries:
  (1) single-data-source  (CoinGecko only)
  (2) single-regime       (last 365 days only)
  (3) supply-detection-timing artifact

This harness re-runs the DECISIVE day-effect placebo on a SECOND, INDEPENDENT
data source (DefiLlama mcap+price exports) spanning 3+ years. Circulating supply
= mcap / price; an unlock = a step-up in supply. Abnormal return = token ret −
BTC ret. If DefiLlama data (different source, longer/multi-regime span) also
gives a significant negative unlock-day effect → the finding is CONFIRMED,
cross-source and cross-regime. If it washes out → it was an artifact.

Data: research/data/defillama/<Token>.csv, each exported via Google Sheets
=DEFILLAMA_HISTORICAL("mcap"/"price", ...). Format: Date,Market Cap,,Date,Price
(DD/MM/YYYY). BTC daily from Coinbase (ccxt, paginated, keyless public client).
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "data", "defillama")
JUMP = 0.015          # supply step-up that counts as an unlock (same as CG study)
PRE, POST = 10, 10
rng = np.random.default_rng(0)


# ── BTC daily benchmark (paginated, multi-year) ───────────────────────────────
def btc_daily():
    from exchange.market_data import get_public_exchange
    ex = get_public_exchange()
    now_ms = ex.milliseconds()
    since = ex.parse8601("2022-12-01T00:00:00Z")
    rows, last = [], None
    while since < now_ms:
        try:
            batch = ex.fetch_ohlcv("BTC/USD", "1d", since=since, limit=300)
        except Exception:
            break
        if not batch:
            break
        rows += batch
        newest = max(r[0] for r in batch)
        if newest == last:
            break
        last = newest
        since = last + 86_400_000
        time.sleep(0.25)
        if len(rows) > 3000:
            break
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t")
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize(None).dt.normalize()
    s = df.set_index("date")["c"].sort_index()
    return s.pct_change()


# ── load one DefiLlama export ─────────────────────────────────────────────────
def load_token(path):
    raw = pd.read_csv(path, header=None, skiprows=1,
                      names=["date", "mcap", "_", "date2", "price"])
    df = pd.DataFrame({
        "date":  pd.to_datetime(raw["date"], format="%d/%m/%Y", errors="coerce"),
        "mcap":  pd.to_numeric(raw["mcap"], errors="coerce"),
        "price": pd.to_numeric(raw["price"], errors="coerce"),
    }).dropna(subset=["date", "mcap", "price"])
    df = df[(df["mcap"] > 0) & (df["price"] > 0)].set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["supply"] = df["mcap"] / df["price"]
    return df if len(df) > 60 else None


def window(df, bret, i):
    """pre-sum, day, post-sum abnormal (token−BTC) around index i, or None."""
    if i - PRE < 0 or i + POST >= len(df):
        return None
    w = df.index[i - PRE:i + POST + 1]
    tr = df["price"].pct_change().reindex(w)
    br = bret.reindex(w)
    if tr.isna().mean() > 0.3 or br.isna().mean() > 0.3:
        return None
    abn = (tr - br).fillna(0.0)
    return abn.iloc[:PRE].sum(), abn.iloc[PRE], abn.iloc[PRE + 1:].sum()


def main():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".csv"))
    print(f"DefiLlama cross-validation — {len(files)} tokens, source-independent of CoinGecko\n")
    bret = btc_daily()
    print(f"BTC benchmark: {bret.index.min().date()} → {bret.index.max().date()} "
          f"({len(bret)} days)\n")

    ev = []                       # (token, pre, day, post)
    real_day, rand_day = [], []   # day-effect placebo pools
    spans = []
    for fn in files:
        tok = fn[:-4]
        df = load_token(os.path.join(DATA, fn))
        if df is None:
            print(f"  {tok:12s} — unusable"); continue
        spans.append((tok, df.index.min().date(), df.index.max().date(), len(df)))
        df["dsupply"] = df["supply"].pct_change()
        abn_day = (df["price"].pct_change() - bret.reindex(df.index))
        unlock_pos = [df.index.get_loc(e) for e in df.index[df["dsupply"] > JUMP]]
        valid = [i for i in unlock_pos if PRE <= i < len(df) - POST]

        for i in valid:
            r = window(df, bret, i)
            if r:
                ev.append((tok, *r))
        # day-effect pools (decisive test): unlock-day abnormal vs random-day
        uds = [df.index[i] for i in valid]
        for d in uds:
            v = abn_day.get(d)
            if v is not None and not np.isnan(v):
                real_day.append(v)
        bad = set()
        for i in valid:
            for k in (-1, 0, 1):
                if 0 <= i + k < len(df):
                    bad.add(df.index[i + k])
        pool = [d for d in df.index if d not in bad and not np.isnan(abn_day.get(d, np.nan))]
        for _ in range(max(len(valid), 1) * 20):
            if pool:
                rand_day.append(abn_day.get(pool[int(rng.integers(len(pool)))]))

    print("Token coverage:")
    for tok, a, b, n in spans:
        print(f"  {tok:12s} {a} → {b}  ({n}d)")
    print()

    if len(ev) < 20:
        print(f"Only {len(ev)} events — too few."); return
    E = pd.DataFrame(ev, columns=["token", "pre", "day", "post"])
    print(f"### {len(E)} unlock events across {E['token'].nunique()} tokens "
          f"(DefiLlama, multi-year)\n")

    def rep(d, label):
        print(f"  {label} (n={len(d)})")
        for col, nm in [("pre", "pre (T-10→T-1)"), ("day", "unlock day"),
                        ("post", "post (T+1→T+10)")]:
            x = d[col].dropna().values
            if len(x) < 8:
                continue
            t, p = stats.ttest_1samp(x, 0)
            flag = "✅" if p < 0.05 else ""
            print(f"    {nm:18s}: abnormal {x.mean()*100:+.2f}%  p={p:.4f} {flag}")

    rep(E, "FULL SAMPLE")
    toks = sorted(E["token"].unique())
    print(); rep(E[E["token"].isin(set(toks[::2]))], "TRAIN tokens")
    rep(E[E["token"].isin(set(toks[1::2]))], "TEST tokens (held out)")

    # ── DECISIVE: day-effect placebo (unlock day vs random day) ───────────────
    real, rand = np.array(real_day), np.array(rand_day)
    diff = real.mean() - rand.mean()
    allv = np.concatenate([real, rand])
    null = [(lambda s: s[:len(real)].mean() - s[len(real):].mean())(rng.permutation(allv))
            for _ in range(5000)]
    pperm = (np.array(null) <= diff).mean()
    print(f"\n### DAY-EFFECT PLACEBO (decisive: unlock-day vs random-day abnormal)")
    print(f"  Real unlock days: {real.mean()*100:+.2f}% (n={len(real)})  |  "
          f"Random days: {rand.mean()*100:+.2f}% (n={len(rand)})")
    print(f"  Difference: {diff*100:+.2f}%   permutation p={pperm:.4f}  "
          f"{'✅ UNLOCK-SPECIFIC (cross-source confirmed)' if pperm < 0.05 else '❌ not distinguishable from drift'}")

    print("\n→ CoinGecko (365d) found unlock-day −1.5% (placebo p≈0.000, holdout p=0.015).")
    print("  If DefiLlama (independent source, 3+yr) agrees → CONFIRMED cross-source & cross-regime.")


if __name__ == "__main__":
    main()
