from huggingface_hub import hf_hub_download
import pandas as pd, os
for month in ["2026-06","2026-07"]:
    p=hf_hub_download("MaximumLeverage/crypto-lob-stream",f"depth/binance/BTCUSDT/{month}.parquet",repo_type="dataset")
    df=pd.read_parquet(p)
    print("DEPTH",month,os.path.getsize(p),df.shape,list(df.columns))
    print(df.head(8).to_dict("records"))
    print(df.tail(8).to_dict("records"))
    for col in ["timestamp_ms","event_time_ms","transaction_time_ms","first_update_id","final_update_id","update_id"]:
        if col in df:
            print(col,df[col].min(),df[col].max(),df[col].nunique(),df[col].diff().dropna().describe().to_dict())
