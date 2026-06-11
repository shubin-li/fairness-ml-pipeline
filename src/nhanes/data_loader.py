"""
Shubin Li

load 10 xpt files from August 2021-August 2023 nhanes into a dict[name,df]
"""

import pandas as pd
from pathlib import Path


TABLE_NAMES = [
    "DEMO_L", "DPQ_L", "SLQ_L", "PAQ_L", "ALQ_L",
    "SMQ_L", "INQ_L", "HIQ_L", "DIQ_L", "BMX_L"
]

DATA_DIR=Path(__file__).parent.parent / "data" / "nhanes"

def load_xpt_files() -> dict[str,pd.DataFrame]:
    tables={}
    for t in TABLE_NAMES:
        data_path= DATA_DIR / f"{t}.xpt"
        df=pd.read_sas(data_path,format="xport", encoding="utf-8")
        tables[t]=df
        
        print(f"load {t} success, shape {df.shape}")
    return tables

if __name__ == "__main__":
    tables = load_xpt_files()
    print("All tables loaded")
