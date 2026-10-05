from huggingface_hub import list_repo_files
files=list_repo_files("ibrahimdaud/binance-btcusdt",repo_type="dataset")
print("COUNT",len(files))
for f in files[:500]:
    print(f)
