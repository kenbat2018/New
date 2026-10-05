import csv, gzip, io, json, math, os, random, tempfile, time
from pathlib import Path
from urllib.parse import urlparse
import requests
import numpy as np
import pandas as pd
from numba import njit

# ----- Research protocol -----
EXCHANGE="binance-futures"
SYMBOL="BTCUSDT"
BASE_SEC=1
TICK_SIZE=0.1
DEPTHS=np.array([1,3,5,10,15,20,25],dtype=np.int64)
CADENCES=np.array([1,2,5,10,30,60,120,300,900],dtype=np.int64)
ZWINS=np.array([2,3,5,8,10,13,21,34],dtype=np.int64)
PRWINS=np.array([2,3,5,8,10,13,21,34],dtype=np.int64)
ZTHS=np.array([0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5],dtype=np.float64)
PTHS=np.array([0.0,0.2,0.5,1.0,2.0,5.0,10.0,20.0],dtype=np.float64)
DIRECTIONS=np.array([0,1],dtype=np.int64)  # 0=both, 1=long-only

DEV_DATES=[f"{y}-{m:02d}-01" for y in range(2020,2025) for m in (1,5,9)]
BACK_DATES=["2019-10-01","2019-12-01"]
FWD_DATES=[f"{y}-{m:02d}-01" for y in (2025,2026) for m in (1,5,9)]
ALL_DATES=BACK_DATES+DEV_DATES+FWD_DATES

BASE_FEE=0.0005
BASE_SLIP_TICKS=1.0
HARSH_FEE=0.0010
HARSH_SLIP_TICKS=2.0

def url_for(date):
    y,m,d=date.split("-")
    return f"https://datasets.tardis.dev/v1/{EXCHANGE}/book_snapshot_25/{y}/{m}/{d}/{SYMBOL}.csv.gz"

def open_dataset_to_temp(date):
    url=url_for(date)
    r=requests.get(url,stream=True,timeout=(20,180),headers={"User-Agent":"Mozilla/5.0","Accept-Encoding":"identity"})
    if r.status_code!=200:
        raise RuntimeError(f"{date} HTTP {r.status_code}")
    fd,path=tempfile.mkstemp(prefix="tardis_",suffix=".dat")
    os.close(fd)
    total=0
    with open(path,"wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk); total+=len(chunk)
    return path,total,url

def text_reader(path):
    with open(path,"rb") as f:
        magic=f.read(2)
    if magic==b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path,"rb"),encoding="utf-8",newline="")
    return open(path,"r",encoding="utf-8",newline="")

def finalize_row(row, idx, day_bucket0, buckets, bids, asks, ofis):
    try:
        ts=int(row[idx["timestamp"]])
        b0=float(row[idx["bids[0].price"]]); a0=float(row[idx["asks[0].price"]])
        bv=np.empty(25,dtype=np.float64); av=np.empty(25,dtype=np.float64)
        for k in range(25):
            bv[k]=float(row[idx[f"bids[{k}].amount"]]) if row[idx[f"bids[{k}].amount"]] else 0.0
            av[k]=float(row[idx[f"asks[{k}].amount"]]) if row[idx[f"asks[{k}].amount"]] else 0.0
        bc=np.cumsum(bv); ac=np.cumsum(av)
        oo=np.empty(len(DEPTHS),dtype=np.float32)
        for j,d in enumerate(DEPTHS):
            tot=bc[d-1]+ac[d-1]
            oo[j]=(bc[d-1]-ac[d-1])/tot if tot>0 else 0.0
        buckets.append(ts//1_000_000)
        bids.append(b0); asks.append(a0); ofis.append(oo)
        return True
    except Exception:
        return False

def load_day(date):
    path,nbytes,url=open_dataset_to_temp(date)
    buckets=[]; bids=[]; asks=[]; ofis=[]
    try:
        with text_reader(path) as f:
            rd=csv.reader(f)
            header=next(rd)
            idx={name:i for i,name in enumerate(header)}
            needed=["timestamp","bids[0].price","asks[0].price"]
            for k in range(25):
                needed += [f"bids[{k}].amount",f"asks[{k}].amount"]
            miss=[x for x in needed if x not in idx]
            if miss: raise RuntimeError(f"Missing columns: {miss[:5]}")
            current_bucket=None; pending=None; raw_rows=0; kept=0
            for row in rd:
                raw_rows+=1
                try: ts=int(row[idx["timestamp"]])
                except: continue
                b=ts//1_000_000
                if current_bucket is None:
                    current_bucket=b; pending=row
                elif b==current_bucket:
                    pending=row
                else:
                    if pending is not None:
                        kept += 1 if finalize_row(pending,idx,0,buckets,bids,asks,ofis) else 0
                    current_bucket=b; pending=row
            if pending is not None:
                kept += 1 if finalize_row(pending,idx,0,buckets,bids,asks,ofis) else 0
    finally:
        try: os.remove(path)
        except: pass

    if not buckets: raise RuntimeError("no parsed snapshots")
    buckets=np.asarray(buckets,dtype=np.int64)
    bids=np.asarray(bids,dtype=np.float64); asks=np.asarray(asks,dtype=np.float64); ofis=np.asarray(ofis,dtype=np.float32)

    # Put data on a regular 1-second grid and forward-fill the latest known book.
    dt=pd.Timestamp(date,tz="UTC")
    start=int(dt.timestamp()); end=start+86400
    slots=buckets-start
    ok=(slots>=0)&(slots<86400)
    slots=slots[ok]; bids=bids[ok]; asks=asks[ok]; ofis=ofis[ok]
    grid_bid=np.full(86400,np.nan,dtype=np.float64); grid_ask=np.full(86400,np.nan,dtype=np.float64)
    grid_ofi=np.full((86400,len(DEPTHS)),np.nan,dtype=np.float32)
    grid_bid[slots]=bids; grid_ask[slots]=asks; grid_ofi[slots]=ofis
    valid=np.isfinite(grid_bid)&np.isfinite(grid_ask)
    if not valid.any(): raise RuntimeError("no valid in-day snapshots")
    first=int(np.argmax(valid)); last=int(np.where(valid)[0][-1])
    ids=np.where(valid,np.arange(86400),-1)
    ids=np.maximum.accumulate(ids)
    sl=slice(first,last+1)
    take=ids[sl]
    grid_bid=grid_bid[take]; grid_ask=grid_ask[take]; grid_ofi=grid_ofi[take]
    return {"bid":grid_bid,"ask":grid_ask,"ofi":grid_ofi,
            "meta":{"date":date,"bytes":nbytes,"raw_rows":raw_rows,"kept_seconds":kept,
                    "grid_seconds":len(grid_bid),"url":url}}

def build_split(days, dates, cadence, depth_index):
    mids=[]; bids=[]; asks=[]; ofis=[]; dayids=[]
    did=0
    for date in dates:
        if date not in days: continue
        x=days[date]; step=int(cadence)
        idx=np.arange(0,len(x["bid"]),step,dtype=np.int64)
        b=x["bid"][idx]; a=x["ask"][idx]; o=x["ofi"][idx,depth_index].astype(np.float64)
        good=np.isfinite(b)&np.isfinite(a)&np.isfinite(o)&(a>=b)
        b=b[good]; a=a[good]; o=o[good]
        if len(b)<5: continue
        mids.append((a+b)/2); bids.append(b); asks.append(a); ofis.append(o); dayids.append(np.full(len(b),did,dtype=np.int32))
        did+=1
    if not mids:
        return tuple(np.array([]) for _ in range(5))
    return np.concatenate(mids),np.concatenate(bids),np.concatenate(asks),np.concatenate(ofis),np.concatenate(dayids)

@njit(cache=True)
def evaluate(mid,bid,ask,ofi,dayid,zwin,prwin,zth,pth,direction,fee,slip_ticks,tick_size):
    if len(mid)==0: return (-9.,-9.,0.,-1.,0.,0.,0.,0.,0.)
    az=2.0/(zwin+1.0); ap=2.0/(prwin+1.0)
    curday=dayid[0]
    cash=1.0; pos=0.0; peak=1.0; day_start=1.0
    mean=ofi[0]; mean_sqr=ofi[0]*ofi[0]; ez=0.0; epr=0.0; last_mid=mid[0]; initz=False; initpr=False
    nday=int(dayid.max())+1
    dayrets=np.empty(nday,dtype=np.float64); daydds=np.zeros(nday,dtype=np.float64)
    dpeak=1.0; dmin=0.0
    gp=0.0; gl=0.0; trades=0; wins=0
    entry_equity=1.0; entry_cash=1.0

    def close_pos(cash,pos,px,fee):
        cash2=cash+pos*px-fee*abs(pos)*px
        return cash2

    def open_long(cash,px,fee):
        # choose qty so gross notional equals pre-trade equity (=cash while flat)
        eq=cash
        q=eq/px
        return cash-q*px-fee*q*px,q

    def open_short(cash,px,fee):
        eq=cash
        q=eq/px
        return cash+q*px-fee*q*px,-q

    for i in range(len(mid)):
        d=dayid[i]
        if d!=curday:
            # flatten at prior book before reset
            if pos>0:
                px=bid[i-1]-slip_ticks*tick_size
                before=entry_equity
                cash=close_pos(cash,pos,px,fee)
                pnl=cash-before
                if pnl>0: gp+=pnl; wins+=1
                else: gl+=-pnl
                trades+=1; pos=0.0
            elif pos<0:
                px=ask[i-1]+slip_ticks*tick_size
                before=entry_equity
                cash=close_pos(cash,pos,px,fee)
                pnl=cash-before
                if pnl>0: gp+=pnl; wins+=1
                else: gl+=-pnl
                trades+=1; pos=0.0
            eq=cash
            dayrets[curday]=eq/day_start-1.0
            daydds[curday]=dmin
            curday=d
            cash=1.0; pos=0.0; peak=1.0; day_start=1.0; dpeak=1.0; dmin=0.0
            mean=ofi[i]; mean_sqr=ofi[i]*ofi[i]; ez=0.0; epr=0.0; last_mid=mid[i]; initz=False; initpr=False
            entry_equity=1.0; entry_cash=1.0

        x=ofi[i]
        mean=az*x+(1.0-az)*mean
        mean_sqr=az*x*x+(1.0-az)*mean_sqr
        var=mean_sqr-mean*mean
        std=math.sqrt(var) if var>1e-12 else 1e-6
        z=(x-mean)/std
        if not initz:
            ez=z; initz=True
        else:
            ez=ap*z+(1.0-ap)*ez
        pd=(mid[i]-last_mid)/tick_size
        last_mid=mid[i]
        if not initpr:
            epr=pd; initpr=True
        else:
            epr=ap*pd+(1.0-ap)*epr

        # Exit first, exactly matching the user's conditional order.
        if pos>0 and (ez<0.0 or epr<0.0):
            px=bid[i]-slip_ticks*tick_size
            before=entry_equity
            cash=close_pos(cash,pos,px,fee); pos=0.0
            pnl=cash-before
            if pnl>0: gp+=pnl; wins+=1
            else: gl+=-pnl
            trades+=1
        elif pos<0 and (ez>0.0 or epr>0.0):
            px=ask[i]+slip_ticks*tick_size
            before=entry_equity
            cash=close_pos(cash,pos,px,fee); pos=0.0
            pnl=cash-before
            if pnl>0: gp+=pnl; wins+=1
            else: gl+=-pnl
            trades+=1

        # Then allow the new entry on the same sample.
        if pos==0.0:
            if ez>=zth and epr>=pth:
                px=ask[i]+slip_ticks*tick_size
                entry_equity=cash
                cash,pos=open_long(cash,px,fee)
            elif direction==0 and ez<=-zth and epr<=-pth:
                px=bid[i]-slip_ticks*tick_size
                entry_equity=cash
                cash,pos=open_short(cash,px,fee)

        eq=cash+pos*mid[i]
        if eq>dpeak: dpeak=eq
        dd=eq/dpeak-1.0 if dpeak>0 else -1.0
        if dd<dmin: dmin=dd

    # final day flatten
    i=len(mid)-1
    if pos>0:
        px=bid[i]-slip_ticks*tick_size
        before=entry_equity; cash=close_pos(cash,pos,px,fee)
        pnl=cash-before
        if pnl>0: gp+=pnl; wins+=1
        else: gl+=-pnl
        trades+=1
    elif pos<0:
        px=ask[i]+slip_ticks*tick_size
        before=entry_equity; cash=close_pos(cash,pos,px,fee)
        pnl=cash-before
        if pnl>0: gp+=pnl; wins+=1
        else: gl+=-pnl
        trades+=1
    dayrets[curday]=cash/day_start-1.0; daydds[curday]=dmin

    med=np.median(dayrets); avg=np.mean(dayrets); sd=np.std(dayrets)
    posdays=0
    for x in dayrets:
        if x>0: posdays+=1
    maxdd=np.min(daydds)
    pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
    winrate=wins/trades if trades>0 else 0.0
    return med,avg,sd,maxdd,float(posdays),float(nday),float(trades),pf,winrate

def metrics_tuple(t):
    return {"median_day":float(t[0]),"mean_day":float(t[1]),"std_day":float(t[2]),"max_sample_day_dd":float(t[3]),
            "positive_days":int(t[4]),"days":int(t[5]),"trades":int(t[6]),"pf":float(t[7]),"win_rate":float(t[8])}

def score_metric(m):
    if m["trades"]<30 or m["days"]<5: return -99.0
    posshare=m["positive_days"]/max(m["days"],1)
    return m["median_day"] + 0.35*m["mean_day"] - 0.35*m["std_day"] - 0.30*abs(m["max_sample_day_dd"]) + 0.015*(posshare-0.5)

def config_key(c):
    return tuple(c)

def evaluate_config(cache, split, c, fee=BASE_FEE, slip=BASE_SLIP_TICKS):
    cadence,depth,zwin,prwin,zth,pth,direction=c
    di=int(np.where(DEPTHS==depth)[0][0])
    key=(split,cadence,depth)
    if key not in cache:
        dates=DEV_DATES if split=="dev" else BACK_DATES if split=="back" else FWD_DATES
        cache[key]=build_split(DAYS,dates,cadence,di)
    arr=cache[key]
    t=evaluate(*arr,zwin,prwin,zth,pth,direction,fee,slip,TICK_SIZE)
    return metrics_tuple(t)

def one_step_neighbors(c):
    domains=[list(CADENCES),list(DEPTHS),list(ZWINS),list(PRWINS),list(ZTHS),list(PTHS),list(DIRECTIONS)]
    out={tuple(c)}
    for j,dom in enumerate(domains):
        val=c[j]; ix=dom.index(val)
        for k in (ix-1,ix+1):
            if 0<=k<len(dom):
                z=list(c); z[j]=dom[k]; out.add(tuple(z))
    # threshold rectangle around candidate
    zi=domains[4].index(c[4]); pi=domains[5].index(c[5])
    for a in range(max(0,zi-1),min(len(domains[4]),zi+2)):
        for b in range(max(0,pi-1),min(len(domains[5]),pi+2)):
            z=list(c); z[4]=domains[4][a]; z[5]=domains[5][b]; out.add(tuple(z))
    return sorted(out)

def main():
    global DAYS
    DAYS={}
    ledger=[]
    for date in ALL_DATES:
        t=time.time()
        try:
            x=load_day(date); DAYS[date]=x
            rec=x["meta"].copy(); rec["status"]="ok"; rec["seconds"]=round(time.time()-t,2)
            ledger.append(rec); print("DAY",date,"ok",rec["bytes"],rec["raw_rows"],rec["kept_seconds"],flush=True)
        except Exception as e:
            ledger.append({"date":date,"status":"error","error":str(e)[:250]}); print("DAY",date,"ERR",e,flush=True)

    have_dev=[d for d in DEV_DATES if d in DAYS]; have_fwd=[d for d in FWD_DATES if d in DAYS]
    if len(have_dev)<10 or len(have_fwd)<3:
        raise RuntimeError(f"insufficient sample days dev={len(have_dev)} fwd={len(have_fwd)}")

    cache={}
    # Warm numba
    _=evaluate_config(cache,"dev",(10,5,5,10,0.2,0.2,0))

    rng=random.Random(20261005)
    configs=set()
    # Original-like settings at every cadence/depth maximum available.
    for cad in CADENCES:
        configs.add((int(cad),25,5,10,0.2,0.2,0))
        configs.add((int(cad),25,5,10,0.2,0.2,1))
    while len(configs)<3000:
        configs.add((int(rng.choice(CADENCES)),int(rng.choice(DEPTHS)),int(rng.choice(ZWINS)),int(rng.choice(PRWINS)),
                     float(rng.choice(ZTHS)),float(rng.choice(PTHS)),int(rng.choice(DIRECTIONS))))
    rows=[]
    for n,c in enumerate(configs,1):
        m=evaluate_config(cache,"dev",c)
        rows.append({"config":c,"score":score_metric(m),"dev":m})
        if n%300==0: print("SEARCH",n,flush=True)
    rows.sort(key=lambda x:x["score"],reverse=True)

    # Robustness: inspect local neighborhoods around top 20, using DEV only.
    robust=[]
    seen=set()
    for base in rows[:20]:
        neigh=one_step_neighbors(base["config"])
        vals=[]
        for c in neigh:
            if c in seen:
                pass
            m=evaluate_config(cache,"dev",c)
            vals.append((c,m,score_metric(m)))
        scores=np.array([v[2] for v in vals])
        good=np.array([(v[1]["mean_day"]>0 and v[1]["pf"]>1 and v[1]["trades"]>=30) for v in vals])
        robust.append({"config":base["config"],"base_dev":base["dev"],"base_score":base["score"],
                       "neighbor_median_score":float(np.median(scores)),"neighbor_good_share":float(good.mean()),"neighbor_n":len(vals)})
    robust.sort(key=lambda x:(x["neighbor_median_score"],x["neighbor_good_share"],x["base_score"]),reverse=True)

    selected=robust[0]["config"]
    # Freeze selection BEFORE any OOS readout.
    selected_dev=evaluate_config(cache,"dev",selected)
    selected_back=evaluate_config(cache,"back",selected)
    selected_fwd=evaluate_config(cache,"fwd",selected)
    selected_harsh_dev=evaluate_config(cache,"dev",selected,HARSH_FEE,HARSH_SLIP_TICKS)
    selected_harsh_fwd=evaluate_config(cache,"fwd",selected,HARSH_FEE,HARSH_SLIP_TICKS)

    # top five robust candidates get OOS shown, but selection remains based on dev-neighborhood ranking.
    top5=[]
    for r in robust[:5]:
        c=r["config"]
        top5.append({"config":c,"robust":r,
                     "backward_oos":evaluate_config(cache,"back",c),
                     "forward_oos":evaluate_config(cache,"fwd",c),
                     "forward_oos_harsh":evaluate_config(cache,"fwd",c,HARSH_FEE,HARSH_SLIP_TICKS)})

    # Approximate the original settings across cadences, but don't use OOS to choose them.
    orig=[]
    for cad in CADENCES:
        c=(int(cad),25,5,10,0.2,0.2,0)
        orig.append({"config":c,"dev":evaluate_config(cache,"dev",c)})
    orig.sort(key=lambda x:score_metric(x["dev"]),reverse=True)
    orig_best=orig[0]
    orig_best["forward_oos"]=evaluate_config(cache,"fwd",orig_best["config"])

    out={
      "data":{"venue":"Binance USDT perpetual via Tardis book_snapshot_25 free sample days","symbol":SYMBOL,
              "available_depth":25,"base_sampling_seconds":BASE_SEC,
              "dev_dates":have_dev,"backward_oos_dates":[d for d in BACK_DATES if d in DAYS],"forward_oos_dates":have_fwd,
              "ledger":ledger},
      "execution":{"fill":"buy/cover at best ask; sell/short at best bid; plus adverse tick slippage",
                   "fee_per_side":BASE_FEE,"slippage_ticks":BASE_SLIP_TICKS,"tick_size":TICK_SIZE,
                   "position_size":"100% equity, 1x notional","day_handling":"positions flattened at end of each isolated sample day"},
      "search":{"random_configs":len(configs),
                "cadences_seconds":CADENCES.tolist(),"depths":DEPTHS.tolist(),"z_windows":ZWINS.tolist(),"pr_windows":PRWINS.tolist(),
                "z_thresholds":ZTHS.tolist(),"price_threshold_ticks":PTHS.tolist(),"directions":{"0":"long+short","1":"long-only"},
                "selection":"2020-2024 sample days only; local-neighborhood median score; OOS never reorders"},
      "selected":{"config":selected,"dev":selected_dev,"backward_oos":selected_back,"forward_oos":selected_fwd,
                  "dev_harsh":selected_harsh_dev,"forward_oos_harsh":selected_harsh_fwd,
                  "robustness":robust[0]},
      "top5_robust":top5,
      "original_like_best_dev_cadence":orig_best,
      "warning":"This is an L2 microstructure test on sparse first-of-month sample days, not a full continuous-history proof."
    }
    Path("ofi_btc_optimization.json").write_text(json.dumps(out,indent=2))
    pd.DataFrame([{"cadence":x["config"][0],"depth":x["config"][1],"zwin":x["config"][2],"prwin":x["config"][3],
                   "zth":x["config"][4],"pth":x["config"][5],"direction":x["config"][6],"score":x["score"],**x["dev"]} for x in rows[:250]]).to_csv("ofi_btc_top250.csv",index=False)
    print("OFI_JSON_BEGIN"); print(json.dumps(out)); print("OFI_JSON_END",flush=True)

if __name__=="__main__":
    main()
