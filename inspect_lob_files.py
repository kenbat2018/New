from huggingface_hub import hf_hub_download
import pandas as pd, os
repo="ibrahimdaud/binance-btcusdt"
for d in ["2021-01-01","2022-01-01","2023-01-01","2024-01-01","2025-01-01","2026-01-01","2026-05-31"]:
    f=f"features/BTCUSDT/{d}.parquet"
    try:
        p=hf_hub_download(repo,f,repo_type="dataset")
        x=pd.read_parquet(p)
        print(d,"size",os.path.getsize(p),"shape",x.shape,"cols",list(x.columns))
        for c in ["bar_time_ms","close","depth_imbalance_1pct"]:
            if c in x: print(c,"nonnull",int(x[c].notna().sum()),"min",x[c].min(),"max",x[c].max())
        print(x[["bar_time_ms","close","depth_imbalance_1pct"]].dropna().head(3).to_dict("records") if "depth_imbalance_1pct" in x else "")
    except Exception as e:
        print("ERR",d,repr(e))
