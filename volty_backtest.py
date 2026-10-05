import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

CAP=100000.0
FEE=0.0005
SLIP=10.0

@njit(cache=True)
def sim(o,h,l,c,tr,yi,L,M,fee,slip):
    n=len(c); ny=int(yi.max())+1
    cash=CAP; pos=0.0; direction=0; ep=0.0; ecomm=0.0
    peak=CAP; maxdd=0.0
    ys=np.full(ny,np.nan); ye=np.full(ny,np.nan); yp=np.full(ny,np.nan); ymdd=np.zeros(ny)
    gp=np.zeros(ny); gl=np.zeros(ny); ty=np.zeros(ny,np.int64); wy=np.zeros(ny,np.int64)
    csum=np.empty(n+1); csum[0]=0.0
    for i in range(n): csum[i+1]=csum[i]+tr[i]

    def mark(px,y,cash,pos,peak,maxdd,yp,ymdd):
        eq=cash+pos*px
        if eq>peak: peak=eq
        dd=eq/peak-1.0 if peak>0 else -1.0
        if dd<maxdd: maxdd=dd
        if math.isnan(yp[y]) or eq>yp[y]: yp[y]=eq
        ddy=eq/yp[y]-1.0 if yp[y]>0 else -1.0
        if ddy<ymdd[y]: ymdd[y]=ddy
        return eq,peak,maxdd

    for j in range(L+1,n):
        y=yi[j]
        eqo=cash+pos*o[j]
        if math.isnan(ys[y]):
            ys[y]=eqo; yp[y]=eqo
        i=j-1
        atr=(csum[i+1]-csum[i+1-L])/L
        up=c[i]+atr*M; dn=c[i]-atr*M

        # Both stop-entry orders are live for the bar. TradingView can fill both
        # sequentially in one OHLC path, reversing twice without recalculation.
        buy_active=True; sell_active=True
        # TV default synthetic path: open -> nearest extreme -> far extreme -> close.
        if abs(o[j]-h[j]) <= abs(o[j]-l[j]):
            p1=h[j]; p2=l[j]
        else:
            p1=l[j]; p2=h[j]

        # helper implemented as repeated inline blocks because numba nested mutation is awkward
        # open-gap events
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
        elif sell_active and o[j]<=dn:
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

        eq,peak,maxdd=mark(o[j],y,cash,pos,peak,maxdd,yp,ymdd)

        # Walk three segments. Only buy stop can trigger on up segments and sell stop on down segments.
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
                        eq,peak,maxdd=mark(fill,y,cash,pos,peak,maxdd,yp,ymdd)
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
                        eq,peak,maxdd=mark(fill,y,cash,pos,peak,maxdd,yp,ymdd)
            eq,peak,maxdd=mark(b,y,cash,pos,peak,maxdd,yp,ymdd)
            a=b

        ye[y]=cash+pos*c[j]

        # A 1x strategy should not be allowed to continue with negative equity.
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

def stats_years(r,years):
    a=[r["annual"].get(y,{}) for y in years]
    a=[x for x in a if np.isfinite(x.get("return",np.nan))]
    if not a:
        return {"median":-9.0,"mean":-9.0,"std":9.0,"worst":-9.0,"pos":0,"maxdd":-1.0,"trades":0,"medianpf":0.0}
    re=np.array([x["return"] for x in a]); dd=np.array([x["maxdd"] for x in a]); pfs=np.array([min(x["pf"],20) for x in a])
    return {"median":float(np.median(re)),"mean":float(np.mean(re)),"std":float(np.std(re)),"worst":float(np.min(re)),
            "pos":int((re>0).sum()),"maxdd":float(np.min(dd)),"trades":int(sum(x["trades"] for x in a)),"medianpf":float(np.median(pfs))}

def main(path):
    df=pd.read_parquet(path)
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True); df=df.set_index("datetime").sort_index()
    df=df[~df.index.duplicated(keep="last")]
    df=df.loc[(df.index>=pd.Timestamp("2020-01-01",tz="UTC"))&(df.index<pd.Timestamp("2026-10-01",tz="UTC"))]
    df=df[["open","high","low","close"]].astype(float).dropna()
    print("DATA",len(df),df.index[0],df.index[-1],flush=True)
    run(df.iloc[:3000],5,4.5)

    rows=[]
    for L in range(2,41):
        for M in np.arange(.5,8.0001,.25):
            r=run(df,L,float(M)); d=stats_years(r,[2020,2021,2022,2023,2024]); s6=stats_years(r,[2020,2021,2022,2023,2024,2025])
            capped=np.clip(np.array([r["annual"][y]["return"] for y in [2020,2021,2022,2023,2024]]),-.75,1.5)
            score=float(np.median(capped)-.30*np.std(capped)-.30*abs(d["maxdd"])+.10*np.min(capped))
            rows.append({"length":L,"mult":round(float(M),2),"score":score,
                         "dev_pos":d["pos"],"dev_median":d["median"],"dev_worst":d["worst"],"dev_dd":d["maxdd"],"dev_trades":d["trades"],"dev_pf":d["medianpf"],
                         "six_pos":s6["pos"],"six_median":s6["median"],"six_worst":s6["worst"],"six_dd":s6["maxdd"],"full_dd":r["maxdd"]})
    g=pd.DataFrame(rows)
    plats=[]; shares=[]
    for _,x in g.iterrows():
        n=g[(g.length.between(x.length-2,x.length+2))&(g.mult.between(x.mult-.5,x.mult+.5))]
        plats.append(float(n.score.median()))
        shares.append(float(((n.dev_pos>=4)&(n.dev_pf>1.0)).mean()))
    g["plateau"]=plats; g["robust_neighbor_share"]=shares

    # Strict no-overfit ranking: only 2020-24 used. 2025 and 2026 only reported afterward.
    cand=g[(g.dev_pos>=4)&(g.dev_trades>=80)&(g.dev_pf>1.0)&(g.dev_dd>-.60)&(g.robust_neighbor_share>=.45)]
    if len(cand)==0:
        cand=g[(g.dev_pos>=4)&(g.dev_trades>=50)&(g.dev_pf>1.0)&(g.dev_dd>-.70)]
    cand=cand.sort_values(["plateau","score"],ascending=False)

    top=[]
    for rank,(_,x) in enumerate(cand.head(10).iterrows(),1):
        r=run(df,int(x.length),float(x.mult))
        top.append({"dev_rank":rank,"length":int(x.length),"mult":float(x.mult),"plateau":float(x.plateau),"robust_neighbor_share":float(x.robust_neighbor_share),
                    "development":stats_years(r,[2020,2021,2022,2023,2024]),
                    "y2025":r["annual"].get(2025,{}),"y2026":r["annual"].get(2026,{}),
                    "annual":r["annual"],"full":{"return":r["return"],"maxdd":r["maxdd"],"pf":r["pf"],"trades":r["trades"]}})

    # Descriptive list specifically matching the user's tolerance: positive in >=5 of 6 completed years.
    six=g[(g.six_pos>=5)&(g.dev_trades>=50)&(g.dev_pf>1.0)&(g.six_dd>-.75)].sort_values(["plateau","score"],ascending=False)
    five=[]
    for rank,(_,x) in enumerate(six.head(20).iterrows(),1):
        r=run(df,int(x.length),float(x.mult))
        five.append({"rank":rank,"length":int(x.length),"mult":float(x.mult),"six_positive_years":int(x.six_pos),
                     "annual":{str(y):r["annual"].get(y,{}) for y in range(2020,2027)},
                     "full":{"return":r["return"],"maxdd":r["maxdd"],"pf":r["pf"],"trades":r["trades"]},
                     "plateau":float(x.plateau),"robust_neighbor_share":float(x.robust_neighbor_share)})
    out={"data":{"rows":len(df),"start":str(df.index[0]),"end":str(df.index[-1]),"source":"Binance spot BTCUSDT 15m via vaquum/HuggingFace"},
         "execution":{"capital":CAP,"commission_pct":FEE*100,"slippage_usd":SLIP,"bar_model":"TradingView default OHLC path with both stop orders allowed to fill sequentially","size":"100% equity","leverage":"1x"},
         "protocol":{"grid":"length 2..40, multiplier 0.5..8.0 step 0.25","development":"2020-2024 used for ranking","oos":"2025 and 2026 not used to rank",
                     "plateau":"median score within ±2 length and ±0.5 multiplier neighborhood"},
         "top_dev_robust":top,"five_of_six":five,"count_five_of_six":int(len(six)),
         "current_5_4_5":run(df,5,4.5)}
    Path("volty_results.json").write_text(json.dumps(out,indent=2))
    g.to_csv("volty_top100.csv",index=False)
    print("RESULT_JSON_BEGIN"); print(json.dumps(out)); print("RESULT_JSON_END",flush=True)

if __name__=="__main__": main(sys.argv[1])
