from huggingface_hub import list_repo_files
files=list_repo_files("MaximumLeverage/crypto-lob-stream",repo_type="dataset")
for f in files:
    if "BTCUSDT" in f:
        print(f)
