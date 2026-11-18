########################################################################################################
# Download one pre-tokenized Pile shard (binidx, 20B tokenizer) and link it as data/pile.bin / .idx,
# the path train.py reads.
########################################################################################################

import os
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "RichardErkhov/RWKV-LM_pile_binidx_dataset"

idx_files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".idx"))
if not idx_files:
    raise SystemExit(f"no .idx files found in {REPO}")
name = idx_files[0][:-4]

os.makedirs("data", exist_ok=True)
for ext in (".idx", ".bin"):
    path = hf_hub_download(REPO, name + ext, repo_type="dataset", local_dir="data")
    link = os.path.join("data", "pile" + ext)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(path), link)

print(f"downloaded {name}: {os.path.getsize('data/pile.bin') / 1e9:.1f} GB, "
      f"~{os.path.getsize('data/pile.bin') / 2e9:.1f}B tokens -> data/pile")
