import json, math
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from ofi_proxy_opt import load, sample_tf, simulate, TFS, FEE, SLIP_BPS, TICK

ZW=[3,5,8,13,21]
PW=[3,5,10,20]
ZT=[.10,.20,.35,.50,.75]
PT=[.2,100.,500.,1000.,2500.]

@njit(cache=True)
def feats(ofi,price,zw,pw,bias):
    n=len(ofi); ez=np.empty(n); ep=np.empty(n)
    az=2.0/(zw+1.0); ap=min(1.0,2.0*bias/(pw+1.0))
    m=ofi[0]; s=ofi[0]*ofi[0]; ze=0.0; pe=0.0; last=price[0]
    for i in range(n):
        v=ofi[i]
        if i:
            m=az*v+(1-az)*m; s=az*v*v+(1-az)*s
        var=s-m*m; sd=math.sqrt(var) if var>1e-12 else 1e-6
        z=(v-m)/sd
        ze=z if i==0 else ap*z+(1-ap)*ze
        pdlt=0.0 if i==0 else (price[i]-last)/TICK
        last=price[i]
        pe=pdlt if i==0 else ap*pdlt+(1-ap)*pe
        ez[i]=ze; ep[i]=pe
    return ez,ep

def bounds(ts):
    arr=ts.astype("int64").to_numpy()
    def ix(d): return int(np.searchsorted(arr,pd.Timestamp(d,tz="UTC").value))
    return ix("2023-01-01"),ix("2024-01-01"),ix("2025-01-01"),ix("2026-01-01"),len(arr)

def train_score(y23,y24,tr):
    if tr[3]<20: return -99.
    if y23[0]<-.25 or y24[0]<-.25: return -98.
    return .35*(y23[0]+y24[0])+.30*min(y23[0],y24[0])-.75*abs(tr[1])+.03*min(tr[2],3)

def val_score(ts,v):
    if v[3]<5: return -99.
    return ts+.8*(v[0]-.8*abs(v[1]))+.02*min(v[2],3)

def md(x):
    return {"return":float(x[0]),"maxdd":float(x[1]),"pf":float(x[2]),"trades":int(x[3]),"wins":int(x[4])}

def evaluate(price,ez,ep,zth,pth,lo,A,B,C,D,E):
    y23=simulate(price,ez,ep,zth,pth,A,B,lo,FEE,SLIP_BPS)
    y24=simulate(price,ez,ep,zth,pth,B,C,lo,FEE,SLIP_BPS)
    tr=simulate(price,ez,ep,zth,pth,A,C,lo,FEE,SLIP_BPS)
    va=simulate(price,ez,ep,zth,pth,C,D,lo,FEE,SLIP_BPS)
    te=simulate(price,ez,ep,zth,pth,D,E,lo,FEE,SLIP_BPS)
    fu=simulate(price,ez,ep,zth,pth,A,E,lo,FEE,SLIP_BPS)
    return y23,y24,tr,va,te,fu

def main():
    df=load(); print("DATA",len(df),df.ts.iloc[0],df.ts.iloc[-1],flush=True)
    coarse={}
    context={}
    for tf,k in TFS.items():
        x=sample_tf(df,k); price=x.close.to_numpy(float); ofi=x.depth_imbalance_1pct.to_numpy(float)
        A,B,C,D,E=bounds(x.ts); rows=[]; cache={}
        for zw in ZW:
            for pw in PW:
                ez,ep=feats(ofi,price,zw,pw,1.0); cache[(zw,pw)]=(ez,ep)
                for zt in ZT:
                    for pt in PT:
                        for lo in (False,True):
                            y23=simulate(price,ez,ep,zt,pt,A,B,lo,FEE,SLIP_BPS)
                            y24=simulate(price,ez,ep,zt,pt,B,C,lo,FEE,SLIP_BPS)
                            tr=simulate(price,ez,ep,zt,pt,A,C,lo,FEE,SLIP_BPS)
                            s=train_score(y23,y24,tr)
                            rows.append((s,zw,pw,zt,pt,lo,y23,y24,tr))
        rows.sort(key=lambda q:q[0],reverse=True)
        best=None
        for r in rows[:150]:
            s,zw,pw,zt,pt,lo,y23,y24,tr=r; ez,ep=cache[(zw,pw)]
            va=simulate(price,ez,ep,zt,pt,C,D,lo,FEE,SLIP_BPS)
            vs=val_score(s,va)
            if best is None or vs>best[0]: best=(vs,r,va)
        vs,r,va=best
        coarse[tf]={"combined":float(vs),"seed":{"zw":r[1],"pw":r[2],"bias":1.0,"zt":r[3],"pt":r[4],"lo":bool(r[5])},
                    "train_score":float(r[0]),"train":md(r[8]),"validation":md(va)}
        context[tf]=(price,ofi,A,B,C,D,E)
        print("COARSE",tf,json.dumps(coarse[tf]),flush=True)

    top_tfs=sorted(coarse,key=lambda t:coarse[t]["combined"],reverse=True)[:4]
    finalists=[]
    zall=[3,4,5,6,8,10,13,16,21,26]
    pall=[3,4,5,7,10,13,16,20,26]
    for tf in top_tfs:
        price,ofi,A,B,C,D,E=context[tf]; seed=coarse[tf]["seed"]
        zc=[z for z in zall if abs(z-seed["zw"])<=5] or [seed["zw"]]
        pc=[p for p in pall if abs(p-seed["pw"])<=6] or [seed["pw"]]
        ztc=sorted(set(max(.02,min(1.2,round(seed["zt"]+d,2))) for d in (-.15,-.05,0,.05,.15)))
        ptc=sorted(set(max(.2,min(6000.,seed["pt"]+d)) for d in (-750,-250,0,250,750)))
        best=None
        for zw in zc:
          for pw in pc:
            for bias in (.5,.75,1.,1.25,1.5):
              ez,ep=feats(ofi,price,zw,pw,bias)
              for zt in ztc:
                for pt in ptc:
                  for lo in (seed["lo"],not seed["lo"]):
                    y23=simulate(price,ez,ep,zt,pt,A,B,lo,FEE,SLIP_BPS)
                    y24=simulate(price,ez,ep,zt,pt,B,C,lo,FEE,SLIP_BPS)
                    tr=simulate(price,ez,ep,zt,pt,A,C,lo,FEE,SLIP_BPS)
                    s=train_score(y23,y24,tr)
                    if s<-90: continue
                    va=simulate(price,ez,ep,zt,pt,C,D,lo,FEE,SLIP_BPS)
                    vs=val_score(s,va)
                    if best is None or vs>best[0]:
                        best=(vs,(zw,pw,bias,zt,pt,lo),y23,y24,tr,va,ez,ep)
        vs,p,y23,y24,tr,va,ez,ep=best; zw,pw,bias,zt,pt,lo=p
        te=simulate(price,ez,ep,zt,pt,D,E,lo,FEE,SLIP_BPS)
        fu=simulate(price,ez,ep,zt,pt,A,E,lo,FEE,SLIP_BPS)
        harsh=simulate(price,ez,ep,zt,pt,A,E,lo,.0010,2.0)
        rec={"timeframe":tf,"settings":{"z_window":zw,"pr_window":pw,"pr_bias":bias,"z_thresh":zt,
             "price_thresh_ticks":pt,"direction":"long_only" if lo else "long_short"},
             "combined_selection_score":float(vs),"2023":md(y23),"2024":md(y24),"train_2023_24":md(tr),
             "validation_2025":md(va),"untouched_test_2026_Jan_May":md(te),"full_2023_May2026":md(fu),"harsh_full":md(harsh)}
        finalists.append(rec); print("FINALIST",json.dumps(rec),flush=True)
    finalists.sort(key=lambda r:r["combined_selection_score"],reverse=True)
    out={"warning":"This is the closest multi-year proxy available, not exact top-35 tick-book replay. It uses Binance BTCUSDT 5m depth_imbalance_1pct (±1% of mid). 2026 is untouched by parameter/timeframe selection.",
         "execution":{"fee_per_leg":FEE,"slippage_bps_per_leg":SLIP_BPS,"size":"100% equity","leverage":"1x"},
         "protocol":{"train":"2023-24","validation_used_for_selection":"2025","untouched_test":"2026 Jan-May",
                     "timeframes":list(TFS.keys()),"searched":"Z/price EMA windows, PR bias, Z threshold, price threshold, direction, timeframe"},
         "coarse_by_timeframe":coarse,"winner":finalists[0],"finalists":finalists}
    Path("ofi_validation_results.json").write_text(json.dumps(out,indent=2))
    print("RESULT_JSON_BEGIN"); print(json.dumps(out)); print("RESULT_JSON_END")

if __name__=="__main__": main()
