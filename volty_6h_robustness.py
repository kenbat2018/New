import io, json, math, zipfile, sys, time
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

PARAMS=[(L,M) for L in range(23,29) for M in (2.75,3.0,3.25,3.5)]
FOCAL={(25,3.0),(26,3.0),(25,3.25),(26,3.25)}
COSTS={
    "baseline":{"fee":0.0005,"slip":10.0},
    "moderate":{"fee":0.00075,"slip":15.0},
    "harsh":{"fee":0.0010,"slip":25.0},
    "extreme":{"fee":0.0015,"slip":50.0},
}

def month_range(start,end):
    p=pd.Period(start,freq="M"); q=pd.Period(end,freq="M")
    while p<=q:
        yield str(p)
        p+=1

def download_binance_6h():
    frames=[]; ledger=[]
    for month in month_range(START_MONTH,END_MONTH):
        url=f"https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{TF}/{SYMBOL}-{TF}-{month}.zip"
        try:
            req=Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urlopen(req,timeout=45) as r:
                payload=r.read()
            z=zipfile.ZipFile(io.BytesIO(payload))
            name=z.namelist()[0]
            raw=z.read(name)
            x=pd.read_csv(io.BytesIO(raw),header=None)
            # archives occasionally acquire a header row; coerce and drop it
            for col in range(min(6,x.shape[1])):
                x[col]=pd.to_numeric(x[col],errors="coerce")
            x=x.dropna(subset=[0,1,2,3,4])
            ts=x[0].astype("int64")
            unit="us" if ts.median()>10**14 else "ms"
            dt=pd.to_datetime(ts,unit=unit,utc=True)
            f=pd.DataFrame({"datetime":dt,"open":x[1].astype(float),"high":x[2].astype(float),
                            "low":x[3].astype(float),"close":x[4].astype(float)})
            frames.append(f)
            ledger.append({"month":month,"rows":len(f),"bytes":len(payload),"status":"ok"})
        except HTTPError as e:
            ledger.append({"month":month,"rows":0,"status":f"http_{e.code}"})
        except Exception as e:
            ledger.append({"month":month,"rows":0,"status":"error","error":str(e)[:160]})
    if not frames:
        raise RuntimeError("No Binance data downloaded")
    df=pd.concat(frames,ignore_index=True).drop_duplicates("datetime").sort_values("datetime")
    df=df.set_index("datetime")
    # complete 6h grid only; do not interpolate
    diffs=df.index.to_series().diff().dropna()
    gaps=int((diffs>pd.Timedelta(hours=6)).sum())
    return df[["open","high","low","close"]],ledger,gaps

@njit(cache=True)
def simulate(o,h,l,c,tr,year_idx,L,M,fee,slip):
    n=len(c); ny=int(year_idx.max())+1
    cash=CAP; pos=0.0; direction=0; ep=0.0; ecomm=0.0; entry_eq=CAP
    peak=CAP; maxdd=0.0
    ys=np.full(ny,np.nan); ye=np.full(ny,np.nan); yp=np.full(ny,np.nan); ymdd=np.zeros(ny)
    gp=np.zeros(ny); gl=np.zeros(ny); ty=np.zeros(ny,np.int64); wy=np.zeros(ny,np.int64)
    csum=np.empty(n+1); csum[0]=0.0
    trade_pnl=np.empty(n); trade_ret=np.empty(n); trade_n=0
    for i in range(n): csum[i+1]=csum[i]+tr[i]

    for j in range(L+1,n):
        y=year_idx[j]
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

        # process helper repeated inline: gap fills then OHLC path crossings
        # Buy gap
        if buy_active and o[j]>=up:
            buy_active=False
            if direction!=1:
                fill=o[j]+slip
                eqpre=cash+pos*fill
                if direction!=0:
                    close_comm=fee*abs(pos)*fill
                    pnl=pos*(fill-ep)-ecomm-close_comm
                    trade_pnl[trade_n]=pnl; trade_ret[trade_n]=pnl/max(entry_eq,1e-12); trade_n+=1
                    ty[y]+=1
                    if pnl>0: gp[y]+=pnl; wy[y]+=1
                    elif pnl<0: gl[y]+=-pnl
                target=eqpre/fill
                delta=target-pos
                comm=fee*abs(delta)*fill
                cash-=delta*fill+comm; pos=target; direction=1; ep=fill; ecomm=fee*abs(target)*fill
                entry_eq=max(cash+pos*fill,1e-12)
        # Sell gap
        if sell_active and o[j]<=dn:
            sell_active=False
            if direction!=-1:
                fill=max(o[j]-slip,0.01)
                eqpre=cash+pos*fill
                if direction!=0:
                    close_comm=fee*abs(pos)*fill
                    pnl=pos*(fill-ep)-ecomm-close_comm
                    trade_pnl[trade_n]=pnl; trade_ret[trade_n]=pnl/max(entry_eq,1e-12); trade_n+=1
                    ty[y]+=1
                    if pnl>0: gp[y]+=pnl; wy[y]+=1
                    elif pnl<0: gl[y]+=-pnl
                target=-eqpre/fill
                delta=target-pos
                comm=fee*abs(delta)*fill
                cash-=delta*fill+comm; pos=target; direction=-1; ep=fill; ecomm=fee*abs(target)*fill
                entry_eq=max(cash+pos*fill,1e-12)

        # mark open
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
            if b>a and buy_active and a<up and up<=b:
                buy_active=False
                if direction!=1:
                    fill=up+slip
                    eqpre=cash+pos*fill
                    if direction!=0:
                        close_comm=fee*abs(pos)*fill
                        pnl=pos*(fill-ep)-ecomm-close_comm
                        trade_pnl[trade_n]=pnl; trade_ret[trade_n]=pnl/max(entry_eq,1e-12); trade_n+=1
                        ty[y]+=1
                        if pnl>0: gp[y]+=pnl; wy[y]+=1
                        elif pnl<0: gl[y]+=-pnl
                    target=eqpre/fill
                    delta=target-pos
                    comm=fee*abs(delta)*fill
                    cash-=delta*fill+comm; pos=target; direction=1; ep=fill; ecomm=fee*abs(target)*fill
                    entry_eq=max(cash+pos*fill,1e-12)
            elif b<a and sell_active and b<=dn and dn<a:
                sell_active=False
                if direction!=-1:
                    fill=max(dn-slip,0.01)
                    eqpre=cash+pos*fill
                    if direction!=0:
                        close_comm=fee*abs(pos)*fill
                        pnl=pos*(fill-ep)-ecomm-close_comm
                        trade_pnl[trade_n]=pnl; trade_ret[trade_n]=pnl/max(entry_eq,1e-12); trade_n+=1
                        ty[y]+=1
                        if pnl>0: gp[y]+=pnl; wy[y]+=1
                        elif pnl<0: gl[y]+=-pnl
                    target=-eqpre/fill
                    delta=target-pos
                    comm=fee*abs(delta)*fill
                    cash-=delta*fill+comm; pos=target; direction=-1; ep=fill; ecomm=fee*abs(target)*fill
                    entry_eq=max(cash+pos*fill,1e-12)

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
            cash=0.0; pos=0.0; direction=0
            for yy in range(y,ny):
                if math.isnan(ys[yy]): ys[yy]=0.0
                ye[yy]=0.0
                if math.isnan(yp[yy]): yp[yy]=0.0
                ymdd[yy]=-1.0
            break

    endeq=max(cash+pos*c[-1],0.0)
    gpt=gp.sum(); glt=gl.sum()
    pf=gpt/glt if glt>0 else (999.0 if gpt>0 else 0.0)
    return endeq,endeq/CAP-1,maxdd,pf,trade_n,ys,ye,ymdd,gp,gl,ty,wy,trade_pnl[:trade_n],trade_ret[:trade_n]

def run(df,L,M,fee,slip):
    o=df.open.to_numpy(float); h=df.high.to_numpy(float); lo=df.low.to_numpy(float); c=df.close.to_numpy(float)
    p=np.empty_like(c); p[0]=c[0]; p[1:]=c[:-1]
    tr=np.maximum(h-lo,np.maximum(np.abs(h-p),np.abs(lo-p)))
    years=df.index.year.to_numpy(); uniq=np.unique(years); mp={int(y):i for i,y in enumerate(uniq)}
    yi=np.array([mp[int(y)] for y in years],dtype=np.int64)
    out=simulate(o,h,lo,c,tr,yi,int(L),float(M),float(fee),float(slip))
    endeq,ret,mdd,pf,nt,ys,ye,ymdd,gp,gl,ty,wy,tpnl,tret=out
    annual={}
    for k,y in enumerate(uniq):
        rr=(ye[k]/ys[k]-1) if np.isfinite(ys[k]) and ys[k]>0 and np.isfinite(ye[k]) else np.nan
        annual[int(y)]={"return":float(rr),"maxdd":float(ymdd[k]),"pf":float(gp[k]/gl[k]) if gl[k]>0 else (999.0 if gp[k]>0 else 0.0),
                        "trades":int(ty[k]),"wins":int(wy[k])}
    days=max((df.index[-1]-df.index[0]).total_seconds()/86400.0,1)
    years_n=days/365.25
    cagr=float((endeq/CAP)**(1/years_n)-1) if endeq>0 else -1.0
    wins=tpnl[tpnl>0]; gross=float(wins.sum()) if len(wins) else 0.0
    sorted_w=np.sort(wins)[::-1] if len(wins) else wins
    def share(k):
        return float(sorted_w[:k].sum()/gross) if gross>0 else np.nan
    net=float(tpnl.sum()) if len(tpnl) else 0.0
    def netshare(k):
        return float(sorted_w[:k].sum()/net) if net>0 else np.nan
    conc={"top1_gross_share":share(1),"top3_gross_share":share(3),"top5_gross_share":share(5),
          "top1_net_share":netshare(1),"top3_net_share":netshare(3),"top5_net_share":netshare(5)}
    trades={"count":int(nt),"win_rate":float((tpnl>0).mean()) if nt else np.nan,
            "median_return":float(np.median(tret)) if nt else np.nan,"mean_return":float(np.mean(tret)) if nt else np.nan,
            "p10_return":float(np.quantile(tret,.10)) if nt else np.nan,"p90_return":float(np.quantile(tret,.90)) if nt else np.nan}
    return {"length":int(L),"mult":float(M),"return":float(ret),"cagr":cagr,"maxdd":float(mdd),"pf":float(pf),
            "end_equity":float(endeq),"annual":annual,"concentration":conc,"trade_stats":trades}

def period_compound(annual,years):
    x=1.0
    used=[]
    worst_dd=0.0
    for y in years:
        a=annual.get(y)
        if a and np.isfinite(a["return"]):
            x*=1+a["return"]; used.append(y); worst_dd=min(worst_dd,a["maxdd"])
    return {"return":float(x-1),"years":used,"worst_calendar_dd":float(worst_dd)}

def main():
    df,ledger,gaps=download_binance_6h()
    print("DATA",len(df),df.index[0],df.index[-1],"gaps",gaps,flush=True)
    run(df.iloc[:min(3000,len(df))],25,3.0,0.0005,10.0)

    all_results={}
    baseline_rows=[]
    for L,M in PARAMS:
        key=f"{L}_{M:.2f}"
        all_results[key]={}
        for cname,cfg in COSTS.items():
            r=run(df,L,M,cfg["fee"],cfg["slip"])
            r["periods"]={
                "pre_sample_2017_2019":period_compound(r["annual"],[2017,2018,2019]),
                "selection_2020_2024":period_compound(r["annual"],[2020,2021,2022,2023,2024]),
                "post_selection_2025":period_compound(r["annual"],[2025]),
                "post_selection_2026_ytd":period_compound(r["annual"],[2026])
            }
            all_results[key][cname]=r
        b=all_results[key]["baseline"]
        baseline_rows.append({
            "length":L,"mult":M,"focal":(L,M) in FOCAL,
            "full_cagr":b["cagr"],"full_return":b["return"],"maxdd":b["maxdd"],"pf":b["pf"],"trades":b["trade_stats"]["count"],
            "pre_2017_19":b["periods"]["pre_sample_2017_2019"]["return"],
            "dev_2020_24":b["periods"]["selection_2020_2024"]["return"],
            "oos_2025":b["periods"]["post_selection_2025"]["return"],
            "oos_2026":b["periods"]["post_selection_2026_ytd"]["return"],
            "top3_gross_share":b["concentration"]["top3_gross_share"],
            "harsh_cagr":all_results[key]["harsh"]["cagr"],"harsh_dd":all_results[key]["harsh"]["maxdd"],
            "extreme_cagr":all_results[key]["extreme"]["cagr"],"extreme_dd":all_results[key]["extreme"]["maxdd"],
        })

    tab=pd.DataFrame(baseline_rows)
    # neighborhood robustness, not a winner search
    robustness={
        "n_params":int(len(tab)),
        "baseline_positive_full_share":float((tab.full_cagr>0).mean()),
        "baseline_positive_pre_sample_share":float((tab.pre_2017_19>0).mean()),
        "baseline_positive_2025_share":float((tab.oos_2025>0).mean()),
        "baseline_positive_2026_share":float((tab.oos_2026>0).mean()),
        "baseline_positive_both_post_share":float(((tab.oos_2025>0)&(tab.oos_2026>0)).mean()),
        "harsh_positive_cagr_share":float((tab.harsh_cagr>0).mean()),
        "extreme_positive_cagr_share":float((tab.extreme_cagr>0).mean()),
        "median_full_cagr":float(tab.full_cagr.median()),
        "median_maxdd":float(tab.maxdd.median()),
        "median_pre_sample_return":float(tab.pre_2017_19.median()),
        "median_2025_return":float(tab.oos_2025.median()),
        "median_2026_return":float(tab.oos_2026.median()),
        "median_top3_gross_share":float(tab.top3_gross_share.median()),
    }

    focal={}
    for L,M in sorted(FOCAL):
        focal[f"{L}_{M:.2f}"]=all_results[f"{L}_{M:.2f}"]

    out={
        "data":{"symbol":SYMBOL,"timeframe":TF,"rows":len(df),"start":str(df.index[0]),"end":str(df.index[-1]),
                "source":"Official Binance spot monthly 6h kline archives","gaps_gt_6h":gaps,
                "months_ok":sum(1 for x in ledger if x["status"]=="ok"),"months_failed":sum(1 for x in ledger if x["status"]!="ok")},
        "protocol":{"selection_origin":"6h plateau discovered using 2020-2024 only",
                    "backward_oos":"2017-2019 was not part of original selection",
                    "forward_oos":"2025 and 2026 were not used to select the plateau",
                    "parameters":"Lengths 23-28 x ATR multipliers 2.75,3.0,3.25,3.5; report the whole neighborhood, do not pick a new best cell",
                    "execution":"TradingView-like 4-point OHLC path; signal from prior completed 6h bar; both stop entries can reverse intrabar; 100% equity; 1x"},
        "costs":COSTS,"robustness":robustness,"focal":focal,"all_results":all_results
    }
    Path("volty_6h_robustness.json").write_text(json.dumps(out,indent=2))
    tab.to_csv("volty_6h_robustness.csv",index=False)
    Path("volty_6h_download_ledger.json").write_text(json.dumps(ledger,indent=2))
    print("ROBUST_JSON_BEGIN"); print(json.dumps(out)); print("ROBUST_JSON_END",flush=True)

if __name__=="__main__":
    main()
