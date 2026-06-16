"""
TOKEN-UNLOCK STUDY — EXPANDED ($0). ~70 VC-heavy alts, CoinGecko free (365d).
Detect unlocks from circulating-supply step-ups (market_cap/price), measure
BTC-relative abnormal returns around them, + placebo control (unlock vs random),
+ token holdout. Fetch-once with disk cache. One pass = study + placebo.

Goal: enough unlock events (target 80-150) to reach significance WITHOUT paying.
"""
import sys, warnings, time, os
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, requests
from scipy import stats
from exchange.market_data import fetch_ohlcv

H={"User-Agent":"Mozilla/5.0"}
CACHE="/tmp/cg_unlocks"; os.makedirs(CACHE, exist_ok=True)
JUMP=0.015; PRE,POST=10,10
rng=np.random.default_rng(0)

TOKENS=[
 "arbitrum","optimism","aptos","sui","celestia","sei-network","immutable-x","starknet",
 "worldcoin-wld","pyth-network","jupiter-exchange-solana","wormhole","ethena",
 "jito-governance-token","dydx-chain","altlayer","manta-network","ondo-finance",
 "ethereum-name-service","blur","apecoin","the-sandbox","decentraland","gmx",
 "axie-infinity","gala","render-token","fetch-ai","injective-protocol","kava",
 "mina-protocol","flow","near","oasis-network","celo","1inch","ribbon-finance",
 "radiant-capital","magic","illuvium","stepn","space-id","aevo","ether-fi","renzo",
 "layerzero","dymension","portal","pixels","tensor","parcl","kamino","drift-protocol",
 "io-net","grass","saga-2","omni-network","dogwifcoin","jasmycoin","mantra-dao",
 "ondo-us-dollar-yield","zircuit","eigenlayer","starknet-token","cyberconnect",
 "hashflow","gains-network","pendle","gmt-token","stargate-finance","sonic-3",
 "ether-fi-staked-eth","aerodrome-finance","velodrome-finance","morpho",
]


def fetch(coin):
    f=f"{CACHE}/{coin}.csv"
    if os.path.exists(f):
        try:
            df=pd.read_csv(f, parse_dates=["date"]).set_index("date")
            return df if len(df)>60 else None
        except Exception: pass
    for attempt in range(2):
        try:
            r=requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                           params={"vs_currency":"usd","days":365,"interval":"daily"},timeout=25,headers=H)
            if r.status_code==429:
                time.sleep(30); continue
            if r.status_code!=200: return None
            j=r.json()
            p=pd.DataFrame(j["prices"],columns=["t","price"]); m=pd.DataFrame(j["market_caps"],columns=["t","mc"])
            df=p.merge(m,on="t"); df["date"]=pd.to_datetime(df["t"],unit="ms").dt.normalize()
            df=df.groupby("date").last(); df["supply"]=df["mc"]/df["price"]
            df[["price","mc","supply"]].to_csv(f)
            return df if len(df)>60 else None
        except Exception:
            time.sleep(5)
    return None


def post_abn(df, bret, i):
    if i-PRE<0 or i+POST>=len(df): return None
    w=df.index[i-PRE:i+POST+1]; tr=df["price"].pct_change().reindex(w); br=bret.reindex(w)
    if tr.isna().mean()>0.3 or br.isna().mean()>0.3: return None
    abn=tr-br
    return abn.iloc[:PRE].sum(), abn.iloc[PRE], abn.iloc[PRE+1:].sum()


def main():
    print(f"Fetching {len(TOKENS)} tokens (cached)…")
    btc=fetch_ohlcv("BTC/USD","1d",limit=400)["close"]; btc.index=btc.index.tz_localize(None).normalize()
    bret=btc.pct_change()
    real=[]; placebo=[]; ev=[]   # ev=(token,pre,day,post)
    got=0
    for c in TOKENS:
        df=fetch(c); time.sleep(2.0 if not os.path.exists(f"{CACHE}/{c}.csv") else 0)
        if df is None: continue
        got+=1
        df["dsupply"]=df["supply"].pct_change()
        pos=[df.index.get_loc(e) for e in df.index[df["dsupply"]>JUMP]]
        valid=[i for i in pos if PRE<=i<len(df)-POST]
        for i in valid:
            r=post_abn(df,bret,i)
            if r: ev.append((c,*r)); real.append(r[2])
        if valid:
            bad=set(j for i in valid for j in range(i-3,i+4))
            pool=[i for i in range(PRE,len(df)-POST) if i not in bad]
            for _ in range(len(valid)*20):
                r=post_abn(df,bret,int(rng.choice(pool)))
                if r: placebo.append(r[2])
    print(f"  usable tokens: {got}/{len(TOKENS)}\n")

    if len(ev)<20:
        print(f"Only {len(ev)} events — still too few."); return
    E=pd.DataFrame(ev,columns=["token","pre","day","post"])
    print(f"### {len(E)} unlock events across {E['token'].nunique()} tokens\n")

    def rep(df,label):
        print(f"  {label} (n={len(df)})")
        for col,nm in [("pre","pre (T-10→T-1)"),("day","unlock day"),("post","post (T+1→T+10)")]:
            x=df[col].dropna().values
            if len(x)<8: continue
            t,p=stats.ttest_1samp(x,0)
            print(f"    {nm:18s}: abnormal {x.mean()*100:+.2f}%  p={p:.4f} {'✅' if p<0.05 else ''}")
    rep(E,"FULL SAMPLE")
    toks=sorted(E["token"].unique())
    print(); rep(E[E["token"].isin(set(toks[::2]))],"TRAIN tokens")
    rep(E[E["token"].isin(set(toks[1::2]))],"TEST tokens (held out)")

    # placebo permutation
    real=np.array(real); plc=np.array(placebo)
    diff=real.mean()-plc.mean(); allv=np.concatenate([real,plc])
    null=[(lambda s:s[:len(real)].mean()-s[len(real):].mean())(rng.permutation(allv)) for _ in range(5000)]
    pperm=(np.array(null)<=diff).mean()
    print(f"\n### PLACEBO CONTROL (post-unlock abnormal: real vs random)")
    print(f"  Real unlocks: {real.mean()*100:+.2f}% (n={len(real)})  |  Random: {plc.mean()*100:+.2f}% (n={len(plc)})")
    print(f"  Difference: {diff*100:+.2f}%   permutation p={pperm:.4f}  "
          f"{'✅ UNLOCK-SPECIFIC' if pperm<0.05 else '❌ not distinguishable from drift'}")


if __name__=="__main__":
    main()
