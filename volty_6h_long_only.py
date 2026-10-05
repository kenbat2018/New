import io, json, math, zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
from numba import njit

CAP=100000.0
SYMBOL="BTCUSDT"
TF="6h"
START_MONTH="2017-08"
END_MONTH="2026-09"
BASE_FEE=0.0005
BASE_SLIP=10.0
COSTS={
    "baseline":{"fee":0.0005,"slip":10.0},
    "harsh":{"fee":0.0010,"slip":25.0},
    "extreme":{"fee":0.0015,"slip":50.0},
}

def month_range(start,end):
    p=pd.Period(start,freq="M"); q=pd.Period(end,freq="M")
    while p<=q:
        yield str(p); p+=1

def download_binance_6h():
    frames=[]; ledger=[]
    for month in month_range(START_MONTH,END_MONTH):
        url=f"https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{TF}/{SYMBOL}-{TF}-{month}.zip"
        try:
            req=Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urlopen(req,timeout=45) as r: payload=r.read()
            z=zipfile.ZipFile(io.BytesIO(payload)); raw=z.read(z.namelist()[0])
            x=pd.read_csv(io.BytesIO(raw),header=None)
            for col in range(min(6,x.shape[1])): x[col]=pd.to_numeric(x[col],errors="coerce")
            x=x.dropna(subset=[0,1,2,3,4])
            ts=x[0].astype("int64"); unit="us" if ts.median()>10**14 else "ms"
            dt=pd.to_datetime(ts,unit=unit,utc=True)
            frames.append(pd.DataFrame({"datetime":dt,"open":x[1].astype(float),"high":x[2].astype(float),
                                        "low":x[3].astype(float),"close":x[4].astype(float)}))
            ledger.append({"month":month,"rows":len(x),"status":"ok"})
        except HTTPError as e:
            ledger.append({"month":month,"rows":0,"status":f"http_{e.code}"})
        except Exception as e:
            ledger.append({"month":month,"rows":0,"status":"error","error":str(e)[:160]})
    if not frames: raise RuntimeError("No data")
    df=pd.concat(frames,ignore_index=True).drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    gaps=int((df.index.to_series().diff().dropna()>pd.Timedelta(hours=6)).sum())
    return df[["open","high","low","close"]],ledger,gaps

@njit(cache=True)
def sim_long_only(o,h,l,c,tr,year_idx,L,M,fee,slip):
    n=len(c); ny=int(year_idx.max())+1
    cash=CAP; qty=0.0; inpos=False; ep=0.0; ecomm=0.0; entry_eq=CAP
    peak=CAP; maxdd=0.0
    ys=np.full(ny,np.nan); ye=np.full(ny,np.nan); yp=np.full(ny,np.nan); ymdd=np.zeros(ny)
    gp=np.zeros(ny); gl=np.zeros(ny); ty=np.zeros(ny,np.int64); wy=np.zeros(ny,np.int64)
    tpnl=np.empty(n); tret=np.empty(n); tn=0
    csum=np.empty(n+1); csum[0]=0.0
    for i in range(n): csum[i+1]=csum[i]+tr[i]

    for j in range(L+1,n):
        y=year_idx[j]
        eqo=cash+qty*o[j]
        if math.isnan(ys[y]):
            ys[y]=eqo; yp[y]=eqo
        i=j-1
        atr=(csum[i+1]-csum[i+1-L])/L
        up=c[i]+atr*M
        dn=c[i]-atr*M

        # TradingView default synthetic path: O -> nearest extreme -> other extreme -> C.
        if abs(o[j]-h[j]) <= abs(o[j]-l[j]):
            p1=h[j]; p2=l[j]
        else:
            p1=l[j]; p2=h[j]

        # State-dependent orders:
        # flat -> upper stop enters long
        # long -> lower stop exits to cash (never opens short)
        # A new exit can become live after an intrabar entry; a new entry can become live after an intrabar exit.
        a=o[j]

        # Gap action at open based on state entering the bar.
        if (not inpos) and o[j]>=up:
            fill=o[j]+slip
            eqpre=cash
            qty=eqpre/fill
            comm=fee*qty*fill
            cash-=qty*fill+comm
            inpos=True; ep=fill; ecomm=comm
            entry_eq=max(cash+qty*fill,1e-12)
        elif inpos and o[j]<=dn:
            fill=max(o[j]-slip,0.01)
            close_comm=fee*qty*fill
            pnl=qty*(fill-ep)-ecomm-close_comm
            tpnl[tn]=pnl; tret[tn]=pnl/max(entry_eq,1e-12); tn+=1
            ty[y]+=1
            if pnl>0: gp[y]+=pnl; wy[y]+=1
            elif pnl<0: gl[y]+=-pnl
            cash+=qty*fill-close_comm
            qty=0.0; inpos=False; ep=0.0; ecomm=0.0

        eq=cash+qty*o[j]
        if eq>peak: peak=eq
        dd=eq/peak-1.0
        if dd<maxdd: maxdd=dd
        if math.isnan(yp[y]) or eq>yp[y]: yp[y]=eq
        ddy=eq/yp[y]-1.0
        if ddy<ymdd[y]: ymdd[y]=ddy

        pts=(p1,p2,c[j])
        for kk in range(3):
            b=pts[kk]
            if b>a:
                # only a flat account can trigger long entry on an upward crossing
                if (not inpos) and a<up and up<=b:
                    fill=up+slip
                    eqpre=cash
                    qty=eqpre/fill
                    comm=fee*qty*fill
                    cash-=qty*fill+comm
                    inpos=True; ep=fill; ecomm=comm
                    entry_eq=max(cash+qty*fill,1e-12)
            elif b<a:
                # only a long account can trigger exit on a downward crossing
                if inpos and b<=dn and dn<a:
                    fill=max(dn-slip,0.01)
                    close_comm=fee*qty*fill
                    pnl=qty*(fill-ep)-ecomm-close_comm
                    tpnl[tn]=pnl; tret[tn]=pnl/max(entry_eq,1e-12); tn+=1
                    ty[y]+=1
                    if pnl>0: gp[y]+=pnl; wy[y]+=1
                    elif pnl<0: gl[y]+=-pnl
                    cash+=qty*fill-close_comm
                    qty=0.0; inpos=False; ep=0.0; ecomm=0.0

            eq=cash+qty*b
            if eq>peak: peak=eq
            dd=eq/peak-1.0
            if dd<maxdd: maxdd=dd
            if math.isnan(yp[y]) or eq>yp[y]: yp[y]=eq
            ddy=eq/yp[y]-1.0
            if ddy<ymdd[y]: ymdd[y]=ddy
            a=b

        ye[y]=cash+qty*c[j]
        if ye[y]<=0:
            cash=0.0; qty=0.0; inpos=False
            for yy in range(y,ny):
                if math.isnan(ys[yy]): ys[yy]=0.0
                ye[yy]=0.0
                if math.isnan(yp[yy]): yp[yy]=0.0
                ymdd[yy]=-1.0
            break

    endeq=max(cash+qty*c[-1],0.0)
    gpt=gp.sum(); glt=gl.sum()
    pf=gpt/glt if glt>0 else (999.0 if gpt>0 else 0.0)
    return endeq,endeq/CAP-1,maxdd,pf,tn,ys,ye,ymdd,gp,gl,ty,wy,tpnl[:tn],tret[:tn]

def run(df,L,M,fee=BASE_FEE,slip=BASE_SLIP):
    o=df.open.to_numpy(float); h=df.high.to_numpy(float); lo=df.low.to_numpy(float); c=df.close.to_numpy(float)
    p=np.empty_like(c); p[0]=c[0]; p[1:]=c[:-1]
    tr=np.maximum(h-lo,np.maximum(np.abs(h-p),np.abs(lo-p)))
    years=df.index.year.to_numpy(); uniq=np.unique(years); mp={int(y):i for i,y in enumerate(uniq)}
    yi=np.array([mp[int(y)] for y in years],dtype=np.int64)
    out=sim_long_only(o,h,lo,c,tr,yi,int(L),float(M),float(fee),float(slip))
    endeq,ret,mdd,pf,tn,ys,ye,ymdd,gp,gl,ty,wy,tpnl,tret=out
    annual={}
    for k,y in enumerate(uniq):
        rr=(ye[k]/ys[k]-1) if np.isfinite(ys[k]) and ys[k]>0 and np.isfinite(ye[k]) else np.nan
        annual[int(y)]={"return":float(rr),"maxdd":float(ymdd[k]),"pf":float(gp[k]/gl[k]) if gl[k]>0 else (999.0 if gp[k]>0 else 0.0),
                        "trades":int(ty[k]),"wins":int(wy[k])}
    years_n=max((df.index[-1]-df.index[0]).total_seconds()/86400/365.25,1e-9)
    cagr=float((endeq/CAP)**(1/years_n)-1) if endeq>0 else -1.0
    return {"length":int(L),"mult":float(M),"return":float(ret),"cagr":cagr,"maxdd":float(mdd),"pf":float(pf),
            "trades":int(tn),"win_rate":float((tpnl>0).mean()) if tn else np.nan,
            "median_trade":float(np.median(tret)) if tn else np.nan,"annual":annual}

def stats(r,years):
    vals=[r["annual"].get(y) for y in years]
    vals=[x for x in vals if x and np.isfinite(x["return"])]
    if not vals: return None
    re=np.array([x["return"] for x in vals]); dd=np.array([x["maxdd"] for x in vals])
    return {"median":float(np.median(re)),"mean":float(np.mean(re)),"std":float(np.std(re)),
            "worst":float(np.min(re)),"positive_years":int((re>0).sum()),
            "worst_dd":float(np.min(dd)),"trades":int(sum(x["trades"] for x in vals))}

def main():
    df,ledger,gaps=download_binance_6h()
    print("DATA",len(df),df.index[0],df.index[-1],"gaps",gaps,flush=True)
    run(df.iloc[:min(3000,len(df))],20,3.0)

    rows=[]
    # Broad but simple two-parameter search. Selection years are 2018-2024 only.
    for L in range(2,81):
        for M in np.arange(0.5,8.0001,0.25):
            r=run(df,L,float(M))
            s=stats(r,[2018,2019,2020,2021,2022,2023,2024])
            if s is None: continue
            capped=np.clip(np.array([r["annual"][y]["return"] for y in [2018,2019,2020,2021,2022,2023,2024]]),-.75,2.0)
            # Balanced robustness objective: annual consistency + return, with a substantial DD penalty.
            score=float(np.median(capped)-0.25*np.std(capped)-0.55*abs(r["maxdd"])+0.10*np.min(capped))
            rows.append({"length":L,"mult":round(float(M),2),"score":score,
                         "dev_pos":s["positive_years"],"dev_median":s["median"],"dev_mean":s["mean"],"dev_std":s["std"],
                         "dev_worst":s["worst"],"dev_worst_dd":s["worst_dd"],"dev_trades":s["trades"],
                         "full_cagr":r["cagr"],"full_dd":r["maxdd"],"full_pf":r["pf"],"full_trades":r["trades"]})
    g=pd.DataFrame(rows)
    plats=[]; shares=[]
    for _,x in g.iterrows():
        n=g[(g.length.between(x.length-2,x.length+2))&(g.mult.between(x.mult-.5,x.mult+.5))]
        plats.append(float(n.score.median()))
        shares.append(float(((n.dev_pos>=5)&(n.full_dd>-.60)&(n.full_cagr>0)).mean()))
    g["plateau"]=plats; g["neighbor_good_share"]=shares

    # Do not simply choose the max CAGR. Require a broad region, positive majority of years,
    # and meaningful trade count. Prefer <=50% lifetime DD when available.
    cand=g[(g.dev_pos>=5)&(g.dev_trades>=20)&(g.full_dd>-.50)&(g.neighbor_good_share>=.40)]
    if len(cand)==0:
        cand=g[(g.dev_pos>=5)&(g.dev_trades>=20)&(g.full_dd>-.60)&(g.neighbor_good_share>=.30)]
    if len(cand)==0:
        cand=g[(g.dev_pos>=4)&(g.dev_trades>=15)&(g.full_dd>-.70)]
    cand=cand.sort_values(["plateau","score"],ascending=False)

    top=[]
    for rank,(_,x) in enumerate(cand.head(15).iterrows(),1):
        L=int(x.length); M=float(x.mult)
        base=run(df,L,M)
        cost_tests={name:run(df,L,M,cfg["fee"],cfg["slip"]) for name,cfg in COSTS.items()}
        top.append({"rank":rank,"length":L,"mult":M,"plateau":float(x.plateau),"neighbor_good_share":float(x.neighbor_good_share),
                    "development_2018_2024":stats(base,[2018,2019,2020,2021,2022,2023,2024]),
                    "y2025":base["annual"].get(2025,{}),"y2026":base["annual"].get(2026,{}),
                    "full":{k:base[k] for k in ["return","cagr","maxdd","pf","trades","win_rate","median_trade"]},
                    "annual":base["annual"],
                    "cost_tests":{name:{k:r[k] for k in ["return","cagr","maxdd","pf","trades"]} for name,r in cost_tests.items()}})

    # Also report the pure max-CAGR under drawdown caps, for transparency.
    caps={}
    for cap in (0.30,0.40,0.50,0.60):
        z=g[(g.full_dd>=-cap)&(g.dev_trades>=20)].sort_values("full_cagr",ascending=False)
        if len(z):
            x=z.iloc[0]; r=run(df,int(x.length),float(x.mult))
            caps[str(cap)]={"length":int(x.length),"mult":float(x.mult),"cagr":r["cagr"],"maxdd":r["maxdd"],"pf":r["pf"],"trades":r["trades"],
                            "dev":stats(r,[2018,2019,2020,2021,2022,2023,2024]),"y2025":r["annual"].get(2025,{}),"y2026":r["annual"].get(2026,{})}

    out={"data":{"rows":len(df),"start":str(df.index[0]),"end":str(df.index[-1]),"gaps_gt_6h":gaps,
                 "source":"Official Binance spot monthly BTCUSDT 6h kline archives"},
         "variant":{"direction":"LONG ONLY","flat_rule":"upper ATR stop enters long","exit_rule":"lower ATR stop exits to cash; never short",
                    "capital":CAP,"size":"100% equity","leverage":"1x","baseline_fee_pct":BASE_FEE*100,"baseline_slippage_usd":BASE_SLIP},
         "protocol":{"hypothesis_note":"Long-only was proposed after inspecting prior long/short history, so this is exploratory rather than pristine OOS proof.",
                     "selection":"2018-2024 only; 2017 partial excluded; 2025 and 2026 reported after selection",
                     "grid":"length 2..80; ATR multiplier .5..8.0 step .25",
                     "objective":"robust plateau, majority-positive years, drawdown penalty; not max CAGR"},
         "top15":top,"max_cagr_under_dd_caps":caps}
    Path("volty_6h_long_only.json").write_text(json.dumps(out,indent=2))
    g.to_csv("volty_6h_long_only_grid.csv",index=False)
    Path("volty_6h_long_only_ledger.json").write_text(json.dumps(ledger,indent=2))
    print("LONG_JSON_BEGIN"); print(json.dumps(out)); print("LONG_JSON_END",flush=True)

if __name__=="__main__":
    main()
