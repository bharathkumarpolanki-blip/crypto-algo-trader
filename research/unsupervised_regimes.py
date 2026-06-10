"""
UNSUPERVISED REGIME DISCOVERY — let the data find its own market states, then
test whether those hidden states predict FORWARD returns out-of-sample.

Methods: K-Means, Gaussian Mixture, Bayesian Gaussian Mixture, and a hand-rolled
Gaussian HMM (Baum-Welch + Viterbi; hmmlearn unavailable in sandbox).

Honest protocol (no lookahead / no in-sample cheating):
  • Features causal (computed at close[T]); standardized on TRAIN only.
  • Models FIT on TRAIN (older 70%); states then assigned to TEST (newest 30%).
  • Per state we measure FORWARD (T+1) return / vol / Sharpe / persistence.
  • The verdict is the TEST set: do hidden states show significantly different
    forward returns on UNSEEN data? (Kruskal-Wallis p). In-sample dispersion is
    expected and meaningless; only OOS dispersion = predictive information.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, requests
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

CM="https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
np.random.seed(0)


def cm(asset, metrics):
    p={"assets":asset,"metrics":",".join(metrics),"frequency":"1d","page_size":10000,"start_time":"2014-01-01"}
    j=requests.get(CM,params=p,timeout=60).json()
    df=pd.DataFrame(j["data"]); df["time"]=pd.to_datetime(df["time"]).dt.tz_localize(None)
    for m in metrics: df[m]=pd.to_numeric(df[m],errors="coerce")
    return df.set_index("time")


def features():
    df=cm("btc",["CapMVRVCur","CapMrktCurUSD","AdrActCnt"])
    price=df["CapMrktCurUSD"]; ret=price.pct_change()
    F=pd.DataFrame({
        "ret1":  ret,
        "ret7":  price.pct_change(7),
        "vol14": ret.rolling(14).std(),
        "mvrv_z":(df["CapMVRVCur"]-df["CapMVRVCur"].rolling(180).mean())/df["CapMVRVCur"].rolling(180).std(),
    }, index=df.index)
    F["fwd1"]=price.shift(-2)/price.shift(-1)-1          # T+1 forward 1d return
    F=F.dropna()
    return F, price


# ── minimal diagonal-Gaussian HMM (Baum-Welch + Viterbi) ──────────────────────
class GaussHMM:
    def __init__(self,K,iters=25): self.K=K; self.iters=iters
    def _emit(self,X):
        # log N(x|mu,var) per state -> (n,K)
        n=len(X); lp=np.zeros((n,self.K))
        for k in range(self.K):
            d=X-self.mu[k]; lp[:,k]=-0.5*np.sum(d*d/self.var[k]+np.log(2*np.pi*self.var[k]),axis=1)
        return lp
    def fit(self,X):
        n,d=X.shape; K=self.K
        km=KMeans(K,n_init=5,random_state=0).fit(X); lab=km.labels_
        self.mu=km.cluster_centers_.copy()
        self.var=np.array([X[lab==k].var(0)+1e-3 for k in range(K)])
        self.pi=np.full(K,1/K); self.A=np.full((K,K),1/K)
        for _ in range(self.iters):
            logB=self._emit(X); B=np.exp(logB-logB.max(1,keepdims=True))+1e-12
            # forward-backward with scaling
            al=np.zeros((n,K)); be=np.zeros((n,K)); c=np.zeros(n)
            al[0]=self.pi*B[0]; c[0]=al[0].sum(); al[0]/=c[0]
            for t in range(1,n):
                al[t]=(al[t-1]@self.A)*B[t]; c[t]=al[t].sum(); al[t]/=c[t]+1e-300
            be[-1]=1
            for t in range(n-2,-1,-1):
                be[t]=(self.A@(B[t+1]*be[t+1]))/(c[t+1]+1e-300)
            g=al*be; g/=g.sum(1,keepdims=True)+1e-300
            xi=np.zeros((K,K))
            for t in range(n-1):
                x=(al[t][:,None]*self.A*(B[t+1]*be[t+1])[None,:]); xi+=x/(x.sum()+1e-300)
            self.pi=g[0]+1e-6; self.pi/=self.pi.sum()
            self.A=xi/(xi.sum(1,keepdims=True)+1e-300)
            for k in range(K):
                w=g[:,k]; sw=w.sum()+1e-300
                self.mu[k]=(w[:,None]*X).sum(0)/sw
                self.var[k]=(w[:,None]*(X-self.mu[k])**2).sum(0)/sw+1e-3
        return self
    def predict(self,X):           # Viterbi
        n=len(X); logB=self._emit(X); logA=np.log(self.A+1e-300); d=np.zeros((n,self.K)); psi=np.zeros((n,self.K),int)
        d[0]=np.log(self.pi+1e-300)+logB[0]
        for t in range(1,n):
            m=d[t-1][:,None]+logA; psi[t]=m.argmax(0); d[t]=m.max(0)+logB[t]
        s=np.zeros(n,int); s[-1]=d[-1].argmax()
        for t in range(n-2,-1,-1): s[t]=psi[t+1][s[t+1]]
        return s


def state_stats(states, fwd, K):
    rows=[]
    for k in range(K):
        m=states==k; f=fwd[m]
        if len(f)<5: rows.append((k,len(f),0,0,0,0)); continue
        ann=f.mean()*365*100; vol=f.std()*np.sqrt(365)*100
        sh=f.mean()/f.std()*np.sqrt(365) if f.std()>0 else 0
        # persistence = avg consecutive run length
        runs=[]; r=0
        for v in m:
            if v: r+=1
            elif r>0: runs.append(r); r=0
        if r>0: runs.append(r)
        rows.append((k,len(f),ann,vol,sh,np.mean(runs) if runs else 0))
    return rows


def kruskal(states, fwd, K):
    groups=[fwd[states==k] for k in range(K) if (states==k).sum()>=5]
    if len(groups)<2: return 1.0
    try: return stats.kruskal(*groups)[1]
    except Exception: return 1.0


def run_method(name, model, Xtr, Xte, ftr, fte, K):
    if name=="HMM":
        model.fit(Xtr); str_tr=model.predict(Xtr); str_te=model.predict(Xte)
    elif name=="KMeans":
        model.fit(Xtr); str_tr=model.predict(Xtr); str_te=model.predict(Xte)
    else:
        model.fit(Xtr); str_tr=model.predict(Xtr); str_te=model.predict(Xte)
    p_tr=kruskal(str_tr,ftr,K); p_te=kruskal(str_te,fte,K)
    print(f"\n── {name} ({K} states) ──  forward-return dispersion: train p={p_tr:.4f}  TEST p={p_te:.4f}  "
          f"{'✅ predictive OOS' if p_te<0.05 else '❌ not OOS-significant'}")
    print(f"   {'state':5s} {'n(te)':>6s} {'fwdRet(ann)':>11s} {'vol(ann)':>9s} {'Sharpe':>7s} {'persist(d)':>10s}")
    rows=state_stats(str_te, fte, K)
    for k,n,ann,vol,sh,per in sorted(rows,key=lambda r:-r[4]):
        print(f"   {k:5d} {n:6d} {ann:+10.0f}% {vol:8.0f}% {sh:+7.2f} {per:10.1f}")
    # OOS Sharpe spread between best and worst state = economic significance
    shs=[r[4] for r in rows if r[1]>=5]
    return p_te, (max(shs)-min(shs)) if shs else 0


def main():
    print("UNSUPERVISED REGIME DISCOVERY — hidden states from price+on-chain (BTC, 12y)\n")
    F,price=features()
    feats=["ret1","ret7","vol14","mvrv_z"]
    cut=int(len(F)*0.7)
    tr=F.iloc[:cut]; te=F.iloc[cut:]
    sc=StandardScaler().fit(tr[feats].values)
    Xtr=sc.transform(tr[feats].values); Xte=sc.transform(te[feats].values)
    ftr=tr["fwd1"].values; fte=te["fwd1"].values
    print(f"Features: {feats}  | train {tr.index.min().date()}→{tr.index.max().date()}  "
          f"test {te.index.min().date()}→{te.index.max().date()}")
    K=4
    results={}
    for name,model in [("KMeans",KMeans(K,n_init=10,random_state=0)),
                       ("GaussianMixture",GaussianMixture(K,n_init=5,random_state=0)),
                       ("BayesianGMM",BayesianGaussianMixture(n_components=K,n_init=5,random_state=0)),
                       ("HMM",GaussHMM(K,iters=25))]:
        try:
            p_te,spread=run_method(name,model,Xtr,Xte,ftr,fte,K)
            results[name]=(p_te,spread)
        except Exception as ex:
            print(f"\n── {name}: error {str(ex)[:80]}")

    print("\n"+"="*70,"\nVERDICT — do hidden regimes contain OOS predictive information?\n"+"="*70)
    print(f"| {'Method':16s} | {'TEST p (fwd-ret differs?)':>26s} | {'OOS Sharpe spread':>17s} |")
    print("|"+"-"*18+"|"+"-"*28+"|"+"-"*19+"|")
    for nm,(p,sp) in results.items():
        flag="✅" if p<0.05 else "❌"
        print(f"| {nm:16s} | {p:25.4f}{flag} | {sp:16.2f} |")
    nsig=sum(1 for p,_ in results.values() if p<0.05)
    print(f"\n  Methods with OOS-significant state dispersion: {nsig}/{len(results)}")


if __name__=="__main__":
    main()
