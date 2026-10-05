from huggingface_hub import hf_hub_download
import pandas as pd, os
for month in ["2026-06","2026-07"]:
    p=hf_hub_download("MaximumLeverage/crypto-lob-stream",f"snapshots/binance/BTCUSDT/{month}.parquet",repo_type="dataset")
    df=pd.read_parquet(p)
    print(month, os.path.getsize(p), df.shape, list(df.columns))
    print(df.head())
    print(df.tail())
    if "timestamp_ms" in df:
        u=df[["timestamp_ms","last_update_id"]].drop_duplicates()
        print("unique snapshots",len(u),"first",u.head(10).to_dict("records"),"last",u.tail(10).to_dict("records"))
        print("interval sec",u.timestamp_ms.diff().dropna().describe().to_dict())
