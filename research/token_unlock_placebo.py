"""
TOKEN-UNLOCK PLACEBO CONTROL — is the ~-2% abnormal drift UNLOCK-SPECIFIC, or just
these alts underperforming BTC all year?

Compare BTC-relative returns around REAL unlocks vs around RANDOM (placebo) dates
for the SAME tokens. If unlock windows are significantly MORE negative than random
windows → real unlock effect. If equal → it's general alt-drift, not unlocks.
Permutation p-value. $0 (CoinGecko free + ccxt BTC).
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, requests
from exchange.market_data import fetch_ohlcv

H={"User-Agent":"Mozilla/5.0"}
TOKENS=["arbitrum","optimism","aptos","sui","celestia","sei-network","immutable-x",
        "starknet","worldcoin-wld","pyth-network","jupiter-exchange-solana","wormhole",
        "ethena","jito-governance-token","dydx-chain","altlayer","manta-network",
        "ondo-finance","ethereum-name-service"]
JUMP=0.015; PRE,POST=10,10
rng=np.random.default_rng(0)


def cg(coin):
    try:
        r=requests.get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
                       params={"vs_currency":"usd","days":365,"interval":"daily"},timeout=25,headers=H)
        if r.status_code!=200: return None
        j=r.json(); p=pd.DataFrame(j["prices"],columns=["t","price"]); m=pd.DataFrame(j["market_caps"],columns=["t","mc"])
        df=p.merge(m,on="t"); df["date"]=pd.to_datetime(df["t"],unit="ms").dt.normalize()
        df=df.groupby("date").last(); df["supply"]=df["mc"]/df["price"]; return df
    except Exception: return None


def window_abn(df, btc_ret, i):
    if i-PRE<0 or i+POST>=len(df): return None
    w=df.index[i-PRE:i+POST+1]; tr=df["price"].pct_change().reindex(w); br=btc_ret.reindex(w)
    if tr.isna().mean()>0.3 or br.isna().mean()>0.3: return None
    abn=(tr-br)
    return abn.iloc[PRE+1:].sum()        # post-unlock cumulative abnormal (T+1..T+10)


def main():
    print("TOKEN-UNLOCK PLACEBO CONTROL (real unlocks vs random dates, BTC-relative)\n")
    btc=fetch_ohlcv("BTC/USD","1d",limit=400)["close"]; btc.index=btc.index.tz_localize(None).normalize()
    bret=btc.pct_change()
    real=[]; placebo=[]
    for coin in TOKENS:
        df=cg(coin); time.sleep(2.5)
        if df is None or len(df)<60: continue
        df["dsupply"]=df["supply"].pct_change()
        unlock_pos=[df.index.get_loc(e) for e in df.index[df["dsupply"]>JUMP]]
        valid=[i for i in unlock_pos if PRE<=i<len(df)-POST]
        if not valid: continue
        for i in valid:
            v=window_abn(df,bret,i)
            if v is not None: real.append(v)
        # placebo: same number of random positions (not within ±3d of a real unlock)
        bad=set()
        for i in valid:
            for k in range(-3,4): bad.add(i+k)
        pool=[i for i in range(PRE,len(df)-POST) if i not in bad]
        for _ in range(len(valid)*20):       # 20x oversample for a stable null
            i=int(rng.choice(pool)); v=window_abn(df,bret,i)
            if v is not None: placebo.append(v)

    real=np.array(real); placebo=np.array(placebo)
    print(f"Real unlock windows : n={len(real)}  mean post-abnormal {real.mean()*100:+.2f}%")
    print(f"Placebo (random)    : n={len(placebo)} mean post-abnormal {placebo.mean()*100:+.2f}%")
    diff=real.mean()-placebo.mean()
    print(f"Difference (unlock − random): {diff*100:+.2f}%")
    # permutation p: is real-window mean more negative than random by chance?
    null=[]
    allv=np.concatenate([real,placebo])
    for _ in range(5000):
        s=rng.permutation(allv); null.append(s[:len(real)].mean()-s[len(real):].mean())
    null=np.array(null)
    p=(null<=diff).mean()            # one-sided: real more negative than random
    print(f"Permutation p (unlock more negative than random): {p:.4f}  "
          f"{'✅ unlock-specific effect' if p<0.05 else '❌ NOT distinguishable from alt-drift'}")
    print(f"\nReading: if p<0.05 and difference is clearly negative → unlocks add real")
    print(f"selling pressure beyond the tokens' general drift → worth proper (paid) data.")


if __name__=="__main__":
    main()
