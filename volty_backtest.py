import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

CAP=100000.0
FEE=0.0005
SLIP=10.0
TIMEFRAMES=["15min","30min","1h","2h","4h","6h","12h","1d"]

@njit(cache=True)
def sim(o,h,l,c,tr,yi,L,M,fee,slip):
    n=len(c); ny=int(yi.max())+1
    cash=CAP; pos=0.0; direction=0; ep=0.0; ecomm=0.0
    peak=CAP; maxdd=0.0
    ys=np.full(ny,np.nan); ye=np.full(ny,np.nan); yp=np.full(ny,np.nan); ymdd=np.zeros(ny)
    gp=np.zeros(ny); gl=np.zeros(ny); ty=np.zeros(ny,np.int64); wy=np.zeros(ny,np.int64)
    csum=np.empty(n+1); csum[0]=0.0
    for i in range(n): csum[i+1]=csum[i]+tr[i]

    for j in range(L+1,n):
        y=yi[j]
        eqo=cash+pos*o[j]
        if math.isnan(ys[y]):
            ys[y]=eqo; yp[y]=eqo
        i=j-1
        atr=(csum[i+1]-csum[i+1-L])/L
        up=c[i]+atr*M; dn=c[i]-atr*M
        buy_active=True; sell_active=True

        if abs(o[j]-h[j]) <= abs(o[j]-l[j]):
            p1=h[j]; p2=l[j]
        else:
            p1=l[j]; p2=h[j]

        # open gap fills
        if buy_active and o[j]>=up:
            buy_active=False
            if direction!=1:
                fill=o[j]+slip
                eqpre=cash+pos*fill
                target=eqpre/fill
                delta=target-pos
                comm=fee*abs(delta)*fill
                if direction!=0:
                    close_comm=fee*abs(pos)*fill
                    pnl=pos*(fill-ep)-ecomm-close_comm
                    ty[y]+=1
                    if pnl>0: gp[y]+=pnl; wy[y]+=1
                    elif pnl<0: gl[y]+=-pnl
                cash-=delta*fill+comm; pos=target; direction=1; ep=fill; ecomm=fee*abs(target)*fill
        if sell_active and o[j]<=dn:
            sell_active=False
            if direction!=-1:
                fill=o[j]-slip
                if fill<=0: fill=0.01
                eqpre=cash+pos*fill
                target=-eqpre/fill
                delta=target-pos
                comm=fee*abs(delta)*fill
                if direction!=0:
                    close_comm=fee*abs(pos)*fill
                    pnl=pos*(fill-ep)-ecomm-close_comm
                    ty[y]+=1
                    if pnl>0: gp[y]+=pnl; wy[y]+=1
                    elif pnl<0: gl[y]+=-pnl
                cash-=delta*fill+comm; pos=target; direction=-1; ep=fill; ecomm=fee*abs(target)*fill

        # mark at open
        eq=cash+pos*o[j]
        if eq>peak: peak=eq
        dd=eq/peak-1.0 if peak>0 else -1.0
        if dd<maxdd: maxdd=dd
        if math.isnan(yp[y]) or eq>yp[y]: yp[y]=eq
        ddy=eq/yp[y]-1.0 if yp[y]>0 else -1.0
        if ddy<ymdd[y]: ymdd[y]=ddy

        a=o[j]
        pts=(p1,p2,c[j])
        for kk in range(3):
            b=pts[kk]
            if b>a:
                if buy_active and a<up and up<=b:
                    buy_active=False
                    if direction!=1:
                        fill=up+slip
                        eqpre=cash+pos*fill
                        target=eqpre/fill
                        delta=target-pos
                        comm=fee*abs(delta)*fill
                        if direction!=0:
                            close_comm=fee*abs(pos)*fill
                            pnl=pos*(fill-ep)-ecomm-close_comm
                            ty[y]+=1
                            if pnl>0: gp[y]+=pnl; wy[y]+=1
                            elif pnl<0: gl[y]+=-pnl
                        cash-=delta*fill+comm; pos=target; direction=1; ep=fill; ecomm=fee*abs(target)*fill
            elif b<a:
                if sell_active and b<=dn and dn<a:
                    sell_active=False
                    if direction!=-1:
                        fill=dn-slip
                        if fill<=0: fill=0.01
                        eqpre=cash+pos*fill
                        target=-eqpre/fill
                        delta=target-pos
                        comm=fee*abs(delta)*fill
                        if direction!=0:
                            close_comm=fee*abs(pos)*fill
                            pnl=pos*(fill-ep)-ecomm-close_comm
                            ty[y]+=1
                            if pnl>0: gp[y]+=pnl; wy[y]+=1
                            elif pnl<0: gl[y]+=-pnl
                        cash-=delta*fill+comm; pos=target; direction=-1; ep=fill; ecomm=fee*abs(target)*fill

            eq=cash+pos*b
            if eq>peak: peak=eq
            dd=eq/peak-1.0 if peak>0 else -1.0
            if dd<maxdd: maxdd=dd
            if math.isnan(yp[y]) or eq>yp[y]: yp[y]=eq
            ddy=eq/yp[y]-1.0 if yp[y]>0 else -1.0
            if ddy<ymdd[y]: ymdd[y]=ddy
            a=b

        ye[y]=cash+pos*c[j]
        if ye[y]<=0:
            for yy in range(y,ny):
                if math.isnan(ys[yy]): ys[yy]=0.0
                ye[yy]=0.0
                if math.isnan(yp[yy]): yp[yy]=0.0
                ymdd[yy]=-1.0
            cash=0.0; pos=0.0; direction=0
            break

    endeq=cash+pos*c[-1]
    if endeq<0: endeq=0.0
    gpt=gp.sum(); glt=gl.sum()
    pf=gpt/glt if glt>0 else (999.0 if gpt>0 else 0.0)
    return endeq,endeq/CAP-1,maxdd,pf,ty.sum(),ys,ye,ymdd,gp,gl,ty,wy

def run(df,L,M,fee=FEE,slip=SLIP):
    o=df.open.to_numpy(float); h=df.high.to_numpy(float); l=df.low.to_numpy(float); c=df.close.to_numpy(float)
    p=np.empty_like(c); p[0]=c[0]; p[1:]=c[:-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-p),np.abs(l-p)))
    yrs=df.index.year.to_numpy(); uniq=np.unique(yrs); mp={int(y):i for i,y in enumerate(uniq)}
    yi=np.array([mp[int(y)] for y in yrs],np.int64)
    out=sim(o,h,l,c,tr,yi,int(L),float(M),float(fee),float(slip))
    endeq,ret,mdd,pf,trades,ys,ye,ymdd,gp,gl,ty,wy=out
    annual={}
    for k,y in enumerate(uniq):
        rr=(ye[k]/ys[k]-1) if np.isfinite(ys[k]) and ys[k]>0 and np.isfinite(ye[k]) else np.nan
        pfy=gp[k]/gl[k] if gl[k]>0 else (999.0 if gp[k]>0 else 0.0)
        annual[int(y)]={"return":float(rr),"maxdd":float(ymdd[k]),"pf":float(pfy),"trades":int(ty[k]),"wins":int(wy[k])}
    return {"length":int(L),"mult":float(M),"return":float(ret),"maxdd":float(mdd),"pf":float(pf),"trades":int(trades),"end_equity":float(endeq),"annual":annual}

def yearly_stats(r,years):
    vals=[r["annual"].get(y) for y in years]
    vals=[x for x in vals if x and np.isfinite(x["return"])]
    if not vals: return None
    re=np.array([x["return"] for x in vals]); dd=np.array([x["maxdd"] for x in vals]); pf=np.array([min(x["pf"],10) for x in vals])
    return {"median":float(np.median(re)),"mean":float(np.mean(re)),"std":float(np.std(re)),"worst":float(np.min(re)),
            "positive_years":int((re>0).sum()),"max_year_dd":float(np.min(dd)),"trades":int(sum(x["trades"] for x in vals)),
            "median_pf":float(np.median(pf))}

def resample_ohlc(df,tf):
    if tf=="15min": return df.copy()
    rule={"30min":"30min","1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1D"}[tf]
    z=df.resample(rule,label="left",closed="left").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return z

def grid_one(df,tf):
    rows=[]
    for L in range(2,41):
        for M in np.arange(.5,8.0001,.25):
            r=run(df,L,float(M))
            d=yearly_stats(r,[2020,2021,2022,2023,2024])
            if d is None: continue
            devrets=np.array([r["annual"][y]["return"] for y in [2020,2021,2022,2023,2024]])
            capped=np.clip(devrets,-.75,1.5)
            # reward durable return, penalize dispersion and drawdown
            score=float(np.median(capped)-.25*np.std(capped)-.25*abs(d["max_year_dd"])+.10*np.min(capped))
            rows.append({"length":L,"mult":round(float(M),2),"score":score,
                         "dev_pos":d["positive_years"],"dev_median":d["median"],"dev_mean":d["mean"],
                         "dev_worst":d["worst"],"dev_dd":d["max_year_dd"],"dev_trades":d["trades"],"dev_pf":d["median_pf"]})
    g=pd.DataFrame(rows)
    plats=[]; shares=[]
    for _,x in g.iterrows():
        n=g[(g.length.between(x.length-2,x.length+2))&(g.mult.between(x.mult-.5,x.mult+.5))]
        plats.append(float(n.score.median()))
        shares.append(float(((n.dev_pos>=3)&(n.dev_pf>=.9)&(n.dev_dd>-.75)).mean()))
    g["plateau"]=plats; g["neighbor_share"]=shares

    # Adaptive minimum trades: high timeframes naturally produce fewer.
    mintr={"15min":80,"30min":60,"1h":40,"2h":25,"4h":15,"6h":12,"12h":8,"1d":5}[tf]
    cand=g[(g.dev_pos>=3)&(g.dev_trades>=mintr)&(g.dev_pf>=.9)&(g.dev_dd>-.75)]
    if len(cand)==0: cand=g[(g.dev_trades>=mintr)&(g.dev_dd>-.85)]
    cand=cand.sort_values(["plateau","score"],ascending=False)

    out=[]
    for rank,(_,x) in enumerate(cand.head(5).iterrows(),1):
        r=run(df,int(x.length),float(x.mult))
        out.append({"rank":rank,"length":int(x.length),"mult":float(x.mult),
                    "plateau":float(x.plateau),"neighbor_share":float(x.neighbor_share),
                    "development":yearly_stats(r,[2020,2021,2022,2023,2024]),
                    "y2025":r["annual"].get(2025,{}),"y2026":r["annual"].get(2026,{}),
                    "annual":r["annual"],
                    "full":{"return":r["return"],"maxdd":r["maxdd"],"pf":r["pf"],"trades":r["trades"]}})
    return out,g

def main(path):
    raw=pd.read_parquet(path)
    raw["datetime"]=pd.to_datetime(raw["datetime"],utc=True); raw=raw.set_index("datetime").sort_index()
    raw=raw[~raw.index.duplicated(keep="last")]
    raw=raw.loc[(raw.index>=pd.Timestamp("2020-01-01",tz="UTC"))&(raw.index<pd.Timestamp("2026-10-01",tz="UTC"))]
    raw=raw[["open","high","low","close"]].astype(float).dropna()
    print("RAW",len(raw),raw.index[0],raw.index[-1],flush=True)

    results={}
    summaries=[]
    allgrids=[]
    for tf in TIMEFRAMES:
        df=resample_ohlc(raw,tf)
        print("TF",tf,"bars",len(df),flush=True)
        # warm numba
        run(df.iloc[:min(3000,len(df))],5,4.5)
        top,g=grid_one(df,tf)
        results[tf]={"bars":len(df),"top5":top}
        gg=g.copy(); gg["timeframe"]=tf; allgrids.append(gg)
        if top:
            b=top[0]
            summaries.append({"timeframe":tf,"length":b["length"],"mult":b["mult"],
                              "dev_positive_years":b["development"]["positive_years"],
                              "dev_median_return":b["development"]["median"],
                              "dev_worst_return":b["development"]["worst"],
                              "dev_max_year_dd":b["development"]["max_year_dd"],
                              "oos_2025_return":b["y2025"].get("return"),"oos_2025_dd":b["y2025"].get("maxdd"),
                              "2026_return":b["y2026"].get("return"),"2026_dd":b["y2026"].get("maxdd"),
                              "full_return":b["full"]["return"],"full_dd":b["full"]["maxdd"],"full_pf":b["full"]["pf"],"trades":b["full"]["trades"]})

    # Do not use 2025/26 to choose. Overall candidate is based strictly on dev plateau/score,
    # with a small preference for lower drawdown and enough trades.
    sdf=pd.DataFrame(summaries)
    out={"data":{"rows":len(raw),"start":str(raw.index[0]),"end":str(raw.index[-1]),"source":"Binance spot BTCUSDT 15m; higher timeframes resampled from 15m"},
         "execution":{"capital":CAP,"commission_pct":FEE*100,"slippage_usd":SLIP,"size":"100% equity","leverage":"1x",
                      "bar_model":"TradingView default synthetic OHLC path; both stop orders may reverse intrabar"},
         "protocol":{"timeframes":TIMEFRAMES,"grid":"length 2..40, multiplier .5..8.0 step .25",
                     "development":"2020-2024 selects params separately inside each timeframe",
                     "oos":"2025 is untouched validation and 2026 is report-only; neither reorders candidates",
                     "warning":"testing multiple timeframes is another model-selection layer, so OOS behavior matters more than dev winner"},
         "summary":summaries,"results":results}
    Path("multitf_results.json").write_text(json.dumps(out,indent=2))
    sdf.to_csv("multitf_summary.csv",index=False)
    pd.concat(allgrids,ignore_index=True).to_csv("multitf_grid.csv",index=False)
    print("MULTITF_JSON_BEGIN"); print(json.dumps(out)); print("MULTITF_JSON_END",flush=True)

if __name__=="__main__": main(sys.argv[1])
