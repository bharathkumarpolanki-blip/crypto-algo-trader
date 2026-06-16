"""
TOKEN-UNLOCK EVENT STUDY (the $0 forward-visible-flow test).

Thesis (IPO-lockup-expiry analog): large token unlocks = forced SUPPLY on a
schedule known in advance → tokens should DRIFT DOWN into the unlock as the market
front-runs insider/VC selling, possibly bouncing after. This is the strongest
forward-visible flow — you can position BEFORE the supply hits.

DATA ($0): CoinGecko free gives price + market cap (365d cap). Circulating supply =
market_cap / price; an UNLOCK appears as a step-up in supply — so we DETECT unlocks
from the data (no reliance on memorised dates). Returns are measured BTC-RELATIVE
(abnormal) to strip out the market regime (all events are in the last ~year).

Tests: abnormal return pre / event / post unlock; aggregate t-tests; token holdout
(train tokens vs test tokens) for OOS generalisation.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, requests
from scipy import stats
from exchange.market_data import fetch_ohlcv

H = {"User-Agent": "Mozilla/5.0"}
# VC-heavy tokens with scheduled vesting unlocks (CoinGecko ids)
TOKENS = ["arbitrum","optimism","aptos","sui","celestia","sei-network","immutable-x",
          "starknet","worldcoin-wld","pyth-network","jupiter-exchange-solana",
          "wormhole","ethena","jito-governance-token","dydx-chain","altlayer",
          "manta-network","ondo-finance","ethereum-name-service"]
JUMP = 0.015      # >1.5% daily circulating-supply increase = an unlock cliff
PRE, POST = 10, 10


def cg(coin):
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                         params={"vs_currency":"usd","days":365,"interval":"daily"},
                         timeout=25, headers=H)
        if r.status_code != 200: return None
        j = r.json()
        p = pd.DataFrame(j["prices"], columns=["t","price"])
        m = pd.DataFrame(j["market_caps"], columns=["t","mc"])
        df = p.merge(m, on="t"); df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
        df = df.groupby("date").last()
        df["supply"] = df["mc"] / df["price"]
        return df
    except Exception:
        return None


def main():
    print("TOKEN-UNLOCK EVENT STUDY  (CoinGecko free, 365d, BTC-relative abnormal returns)\n")
    # BTC benchmark (free via ccxt)
    btc = fetch_ohlcv("BTC/USD","1d",limit=400)["close"]
    btc.index = btc.index.tz_localize(None).normalize()
    btc_ret = btc.pct_change()

    events = []   # (token, date, pre_abn, day_abn, post_abn, supply_jump)
    used = []
    for coin in TOKENS:
        df = cg(coin)
        time.sleep(2.5)                      # be gentle with the free rate limit
        if df is None or len(df) < 60: continue
        df["dsupply"] = df["supply"].pct_change()
        tret = df["price"].pct_change()
        jumps = df.index[df["dsupply"] > JUMP]
        n_ev = 0
        for e in jumps:
            i = df.index.get_loc(e)
            if i - PRE < 0 or i + POST >= len(df): continue
            w = df.index[i-PRE:i+POST+1]
            br = btc_ret.reindex(w)
            tr = tret.reindex(w)
            if br.isna().mean() > 0.3 or tr.isna().mean() > 0.3: continue
            abn = (tr - br)                  # abnormal (token minus BTC) daily
            pre_abn  = abn.iloc[:PRE].sum()
            day_abn  = abn.iloc[PRE]
            post_abn = abn.iloc[PRE+1:].sum()
            events.append((coin, e, pre_abn, day_abn, post_abn, df["dsupply"].iloc[i]))
            n_ev += 1
        if n_ev: used.append((coin, n_ev))

    if len(events) < 15:
        print(f"Only {len(events)} unlock events detected — too few (CoinGecko 365d limit).")
        for c,n in used: print(f"   {c}: {n}")
        return
    E = pd.DataFrame(events, columns=["token","date","pre","day","post","jump"])
    print(f"Detected {len(E)} unlock events across {E['token'].nunique()} tokens "
          f"({E['date'].min().date()} → {E['date'].max().date()})\n")

    def rep(df, label):
        print(f"  {label} (n={len(df)})")
        for col,name in [("pre","pre-unlock (T-10→T-1)"),("day","unlock day"),("post","post-unlock (T+1→T+10)")]:
            x = df[col].dropna().values
            if len(x) < 8: continue
            t,p = stats.ttest_1samp(x,0)
            flag = "✅" if p<0.05 else ""
            print(f"    {name:24s}: mean abnormal {x.mean()*100:+.2f}%   p={p:.3f} {flag}")

    rep(E, "FULL SAMPLE")

    # token holdout: split tokens in half (does the effect generalise to unseen tokens?)
    toks = sorted(E["token"].unique())
    train_tok = set(toks[::2]); test_tok = set(toks[1::2])
    print()
    rep(E[E["token"].isin(train_tok)], "TRAIN tokens")
    rep(E[E["token"].isin(test_tok)],  "TEST tokens (held out)")

    print("\nVERDICT: a real, deployable unlock edge needs the pre-unlock drift to be")
    print("significant with a CONSISTENT sign on the held-out tokens too. If it only")
    print("appears in-sample / full-sample, it's noise.")


if __name__ == "__main__":
    main()
