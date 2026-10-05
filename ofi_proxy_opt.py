import json, math, os, glob
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from huggingface_hub import snapshot_download

REPO="ibrahimdaud/btcusdt-futures-features"
TICK=0.01
FEE=0.0005
SLIP_BPS=1.0
TFS={"5m":1,"15m":3,"30m":6,"1h":12,"2h":24,"4h":48,"6h":72,"12h":144,"1d":288}
Z_WINS=[3,5,8,13,21]
PR_WINS=[3,5,10,20]
Z_THS=[0.10,0.20,0.35,0.50,0.75]
P_THS=[0.0,100.0,500.0,1000.0,2500.0]

def ema_alpha(n,b=1.0): return 2*b/(n+1)

def load():
    root=snapshot_download(REPO,repo_type="dataset",ignore_patterns=["raw/*"])
    files=glob.glob(os.path.join(root,"features","BTCUSDT","*.parquet"))
    if not files:
        raise RuntimeError("No feature parquet files found")
    parts=[]
    for f in files:
        x=pd.read_parquet(f,columns=["bar_time_ms","close","depth_imbalance_1pct"])
        parts.append(x)
    df=pd.concat(parts,ignore_index=True)
    df["ts"]=pd.to_datetime(df.bar_time_ms,unit="ms",utc=True)
    df=df.sort_values("ts").drop_duplicates("ts")
    df=df[(df.ts>=pd.Timestamp("2023-01-01",tz="UTC"))&(df.ts<pd.Timestamp("2026-06-01",tz="UTC"))]
    df=df.dropna(subset=["close","depth_imbalance_1pct"])
    return df[["ts","close","depth_imbalance_1pct"]].reset_index(drop=True)

def sample_tf(df,k):
    # use last observation in each k*5m block, matching a sampled state rather than averaging books
    x=df.iloc[k-1::k].copy().reset_index(drop=True)
    return x

@njit(cache=True)
def make_features(ofi,price,zw,pw):
    n=len(ofi)
    ez=np.empty(n); epr=np.empty(n)
    am=2.0/(zw+1.0); ap=2.0/(pw+1.0)
    mean=ofi[0]; msq=ofi[0]*ofi[0]; zema=0.0; prem=0.0
    lastp=price[0]
    for i in range(n):
        v=ofi[i]
        if i==0:
            mean=v; msq=v*v
        else:
            mean=am*v+(1-am)*mean
            msq=am*(v*v)+(1-am)*msq
        var=msq-mean*mean
        std=math.sqrt(var) if var>1e-12 else 1e-6
        z=(v-mean)/std
        if i==0: zema=z
        else: zema=ap*z+(1-ap)*zema
        pr=(price[i]-lastp)/TICK if i>0 else 0.0
        lastp=price[i]
        if i==0: prem=pr
        else: prem=ap*pr+(1-ap)*prem
        ez[i]=zema; epr[i]=prem
    return ez,epr

@njit(cache=True)
def simulate(price,ez,epr,zth,pth,start_i,end_i,long_only,fee,slip_bps):
    eq=1.0; pos=0 # -1 short,0 flat,1 long
    peak=1.0; mdd=0.0; trades=0; wins=0; gp=0.0; gl=0.0
    entry_eq=1.0
    prevp=price[start_i]
    for i in range(start_i+1,end_i):
        p=price[i]
        # mark existing position over sampled interval
        r=(p/prevp-1.0)*pos
        eq*=max(0.0,1.0+r)
        prevp=p
        if eq<=0: return 0.0,-1.0,0.0,trades,wins
        if eq>peak: peak=eq
        dd=eq/peak-1.0
        if dd<mdd: mdd=dd

        target=pos
        if long_only:
            if pos==0 and ez[i]>=zth and epr[i]>=pth: target=1
            elif pos==1 and (ez[i]<0 or epr[i]<0): target=0
        else:
            if ez[i]>=zth and epr[i]>=pth: target=1
            elif ez[i]<=-zth and epr[i]<=-pth: target=-1
            elif pos==1 and (ez[i]<0 or epr[i]<0): target=0
            elif pos==-1 and (ez[i]>0 or epr[i]>0): target=0

        if target!=pos:
            # close old leg if any; then open new leg if any
            nlegs=(1 if pos!=0 else 0)+(1 if target!=0 else 0)
            cost=nlegs*(fee+slip_bps/10000.0)
            before=eq
            eq*=max(0.0,1.0-cost)
            if pos!=0:
                pnl=eq-entry_eq
                trades+=1
                if pnl>0: wins+=1; gp+=pnl
                else: gl+=-pnl
            pos=target
            if pos!=0: entry_eq=eq
            if eq>peak: peak=eq
            dd=eq/peak-1.0
            if dd<mdd: mdd=dd
    pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
    return eq-1.0,mdd,pf,trades,wins

def split_idx(ts):
    arr=np.array(ts.values.astype("datetime64[ns]"))
    def idx(d): return int(np.searchsorted(arr,np.datetime64(d)))
    # train 2023-2024, validation 2025, test Jan-May 2026
    return idx("2023-01-01"),idx("2025-01-01"),idx("2026-01-01"),len(arr)

def main():
    df=load()
    print("DATA",len(df),df.ts.iloc[0],df.ts.iloc[-1],flush=True)
    allres={}
    for tf,k in TFS.items():
        x=sample_tf(df,k)
        price=x.close.to_numpy(np.float64); ofi=x.depth_imbalance_1pct.to_numpy(np.float64)
        a,b,c,d=split_idx(x.ts)
        rows=[]
        feature_cache={}
        for zw in Z_WINS:
            for pw in PR_WINS:
                feature_cache[(zw,pw)]=make_features(ofi,price,zw,pw)
        for (zw,pw),(ez,epr) in feature_cache.items():
            for zth in Z_THS:
                for pth in P_THS:
                    for long_only in (False,True):
                        tr=simulate(price,ez,epr,zth,pth,a,b,long_only,FEE,SLIP_BPS)
                        # robust train score; penalize drawdown and tiny trade counts
                        ret,dd,pf,n,w=tr
                        score=ret-0.8*abs(dd)
                        if n<20: score-=0.5
                        rows.append((score,zw,pw,zth,pth,long_only,ret,dd,pf,n,w))
        rows.sort(reverse=True,key=lambda r:r[0])
        # plateau-aware: among top 50 train scores, prefer configs with nearby settings also positive
        top=rows[:50]
        candidates=[]
        for r in top:
            _,zw,pw,zth,pth,lo,*_=r
            neigh=[q for q in rows if abs(q[1]-zw)<=3 and abs(q[2]-pw)<=5 and abs(q[3]-zth)<=0.2 and abs(q[4]-pth)<=500 and q[5]==lo]
            good=sum(1 for q in neigh if q[6]>0 and q[7]>-0.5 and q[9]>=10)
            share=good/max(len(neigh),1)
            candidates.append((share,r))
        candidates.sort(key=lambda x:(x[0],x[1][0]),reverse=True)
        share,r=candidates[0]
        score,zw,pw,zth,pth,lo,train_ret,train_dd,train_pf,train_n,train_w=r
        ez,epr=feature_cache[(zw,pw)]
        val=simulate(price,ez,epr,zth,pth,b,c,lo,FEE,SLIP_BPS)
        test=simulate(price,ez,epr,zth,pth,c,d,lo,FEE,SLIP_BPS)
        full=simulate(price,ez,epr,zth,pth,a,d,lo,FEE,SLIP_BPS)
        allres[tf]={
          "selected":{"z_window":zw,"pr_window":pw,"z_thresh":zth,"price_thresh_ticks":pth,"long_only":bool(lo),"neighbor_good_share":share},
          "train_2023_2024":{"return":train_ret,"maxdd":train_dd,"pf":train_pf,"trades":train_n,"wins":train_w},
          "validation_2025":{"return":val[0],"maxdd":val[1],"pf":val[2],"trades":val[3],"wins":val[4]},
          "test_2026_Jan_May":{"return":test[0],"maxdd":test[1],"pf":test[2],"trades":test[3],"wins":test[4]},
          "full_2023_May2026":{"return":full[0],"maxdd":full[1],"pf":full[2],"trades":full[3],"wins":full[4]},
          "bars":len(x)
        }
        print(tf,allres[tf],flush=True)
    # Rank timeframes ONLY by train score then report OOS; don't silently re-rank on OOS
    def trainscore(v):
        q=v["train_2023_2024"]
        return q["return"]-0.8*abs(q["maxdd"])
    ranked=sorted(allres.items(),key=lambda kv:trainscore(kv[1]),reverse=True)
    out={
      "warning":"PROXY test: uses Binance futures 5-minute ±1% depth imbalance, not exact top-N L2 tick replay. Exact reconstructable public L2 found only for Jun-Jul 2026.",
      "source":"ibrahimdaud/binance-btcusdt features; raw source Binance public archive",
      "execution":{"fee_per_leg":FEE,"slippage_bps_per_leg":SLIP_BPS,"position_size":"100% equity","tick_size_assumed":TICK},
      "protocol":{"train":"2023-2024","validation":"2025","test":"2026-01 through 2026-05","timeframes":list(TFS.keys())},
      "ranked_by_train":[{"timeframe":k,**v} for k,v in ranked]
    }
    Path("ofi_proxy_results.json").write_text(json.dumps(out,indent=2))
    print("RESULT_JSON_BEGIN"); print(json.dumps(out)); print("RESULT_JSON_END")

if __name__=="__main__": main()
