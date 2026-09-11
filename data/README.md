# Data

This project uses the **DataCo Smart Supply Chain Dataset**, available publicly on Kaggle:
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

## Setup

1. Download `DataCoSupplyChainDataset.csv` (and optionally `DescriptionDataCoSupplyChain.csv` and
   `tokenized_access_logs.csv`, not required for this project) from the Kaggle link above.
2. Place `DataCoSupplyChainDataset.csv` in this `data/` folder.
3. Note: the file is typically encoded in `latin1`/`ISO-8859-1`, not UTF-8 — `01_eda.py` already handles
   this.

The raw CSV is not committed to this repo (see `.gitignore`) — only scripts and generated
outputs/charts are tracked.
