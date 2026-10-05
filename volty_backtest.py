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
    for j in range(L+1,n):
        y=yi[j]
        eqo=cash+pos*o[j]
        if math.isnan(ys[y]):
            ys[y]=eqo; yp[y]=eqo
        i=j-1
        atr=(csum[i+1]-csum[i+1-L])/L
        up=c[i]+atr*M; dn=c[i]-atr*M
        td=0; base=0.0
        if direction==1:
            if o[j]<=dn: td=-1; base=o[j]
            elif l[j]<=dn: td=-1; base=dn
        elif direction==-1:
            if o[j]>=up: td=1; base=o[j]
            elif h[j]>=up: td=1; base=up
        else:
            if o[j]>=up: td=1; base=o[j]
            elif o[j]<=dn: td=-1; base=o[j]
            else:
                high_first=abs(o[j]-h[j])<=abs(o[j]-l[j])
                if high_first:
                    if h[j]>=up: td=1; base=up
                    elif l[j]<=dn: td=-1; base=dn
                else:
                    if l[j]<=dn: td=-1; base=dn
                    elif h[j]>=up: td=1; base=up
        if td!=0 and td!=direction:
            fill=base+slip if td==1 else base-slip
            if fill<=0: fill=max(base,0.01)
            eqpre=cash+pos*fill
            target=td*(eqpre/fill)
            delta=target-pos
            comm=fee*abs(delta)*fill
            if direction!=0:
                close_comm=fee*abs(pos)*fill
                pnl=pos*(fill-ep)-ecomm-close_comm
                ty[y]+=1
                if pnl>0: gp[y]+=pnl; wy[y]+=1
                elif pnl<0: gl[y]+=-pnl
            cash-=delta*fill+comm
            pos=target; direction=td; ep=fill
            ecomm=fee*abs(target)*fill
        eq=cash+pos*c[j]
        if eq>peak: peak=eq
        dd=eq/peak-1.0 if peak>0 else -1.0
        if dd<maxdd: maxdd=dd
        if math.isnan(yp[y]) or eq>yp[y]: yp[y]=eq
        ddy=eq/yp[y]-1.0 if yp[y]>0 else -1.0
        if ddy<ymdd[y]: ymdd[y]=ddy
        ye[y]=eq
        if eq<=0:
            for yy in range(y,ny):
                if math.isnan(ys[yy]): ys[yy]=0.0
                ye[yy]=0.0; yp[yy]=0.0; ymdd[yy]=-1.0
            break
    endeq=cash+pos*c[-1]
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

def devstats(r):
    a=[r["annual"][y] for y in [2020,2021,2022,2023,2024]]
    re=np.array([x["return"] for x in a]); dd=np.array([x["maxdd"] for x in a]); pfs=np.array([min(x["pf"],20) for x in a])
    capped=np.clip(re,-.75,1.5)
    score=float(np.median(capped)-.35*np.std(capped)-.35*abs(np.min(dd))+.10*np.min(capped))
    return dict(score=score,median=float(np.median(re)),mean=float(np.mean(re)),std=float(np.std(re)),worst=float(np.min(re)),
                posyears=int((re>0).sum()),maxyeardd=float(np.min(dd)),trades=int(sum(x["trades"] for x in a)),medianpf=float(np.median(pfs)))

def main(path):
    df=pd.read_parquet(path)
    df["datetime"]=pd.to_datetime(df["datetime"],utc=True); df=df.set_index("datetime").sort_index()
    df=df[~df.index.duplicated(keep="last")]
    df=df.loc[(df.index>=pd.Timestamp("2020-01-01",tz="UTC"))&(df.index<pd.Timestamp("2026-10-01",tz="UTC"))]
    df=df[["open","high","low","close"]].astype(float).dropna()
    print("DATA",len(df),df.index[0],df.index[-1],flush=True)
    run(df.iloc[:2000],5,4.5)
    rows=[]
    for L in range(2,31):
        for M in np.arange(.5,8.0001,.25):
            r=run(df,L,float(M)); s=devstats(r)
            rows.append({"length":L,"mult":round(float(M),2),**s})
    g=pd.DataFrame(rows)
    g["eligible"]=(g.posyears>=4)&(g.trades>=40)&(g.medianpf>1.0)&(g.maxyeardd>-.50)
    plat=[]; share=[]; nn=[]
    for _,x in g.iterrows():
        n=g[(g.length.between(x.length-2,x.length+2))&(g.mult.between(x.mult-.5,x.mult+.5))]
        plat.append(float(n.score.median())); share.append(float((n.score>0).mean())); nn.append(len(n))
    g["plateau"]=plat; g["neighbor_positive_share"]=share; g["neighbor_n"]=nn
    c=g[g.eligible&(g.neighbor_positive_share>=.70)].sort_values(["plateau","score"],ascending=False)
    if len(c)==0: c=g[g.eligible].sort_values(["plateau","score"],ascending=False)
    if len(c)==0: c=g.sort_values(["plateau","score"],ascending=False)
    top=[]
    for rank,(_,x) in enumerate(c.head(15).iterrows(),1):
        r=run(df,int(x.length),float(x.mult))
        top.append({"dev_rank":rank,"length":int(x.length),"mult":float(x.mult),"plateau":float(x.plateau),
                    "neighbor_positive_share":float(x.neighbor_positive_share),"dev":{"median_return":float(x["median"]),"worst_return":float(x.worst),
                    "positive_years":int(x.posyears),"max_year_dd":float(x.maxyeardd),"trades":int(x.trades),"median_pf":float(x.medianpf)},
                    "y2025":r["annual"].get(2025,{}),"y2026":r["annual"].get(2026,{}),
                    "full":{"return":r["return"],"maxdd":r["maxdd"],"pf":r["pf"],"trades":r["trades"]}})
    sel=top[0]; L=sel["length"]; M=sel["mult"]
    base=run(df,L,M); harsh=run(df,L,M,.001,20.0)
    cur=run(df,5,4.5)
    out={"data":{"rows":len(df),"start":str(df.index[0]),"end":str(df.index[-1]),"source":"Binance spot BTCUSDT 15m via vaquum/HuggingFace"},
         "execution":{"capital":CAP,"commission_pct":FEE*100,"slippage_usd":SLIP,"bar_model":"TradingView-like 4-tick OHLC","size":"100% equity","leverage":"1x"},
         "protocol":{"development":"2020-2024 only","validation":"2025 untouched for ranking","report":"2026 untouched for ranking","grid":"length 2..30; multiplier 0.5..8.0 step 0.25",
                     "eligibility":"4/5 positive dev years, >=40 dev closed trades, median annual PF>1, no dev-year DD worse than 50%",
                     "selection":"highest median robustness score in ±2 length / ±0.5 multiplier neighborhood; OOS years not used to reorder"},
         "selected":sel,"top15":top,"selected_baseline":base,"selected_harsh_costs":harsh,"current_5_4_5":cur}
    Path("volty_results.json").write_text(json.dumps(out,indent=2))
    g.sort_values(["plateau","score"],ascending=False).head(100).to_csv("volty_top100.csv",index=False)
    print("RESULT_JSON_BEGIN"); print(json.dumps(out)); print("RESULT_JSON_END",flush=True)

if __name__=="__main__": main(sys.argv[1])
