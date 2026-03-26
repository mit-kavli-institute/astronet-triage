import json
import time
import requests
import pandas as pd

MAST_URL = "https://mast.stsci.edu/api/v0/invoke"

def mast_query(request):
    r = requests.post(MAST_URL, data={"request": json.dumps(request)})
    r.raise_for_status()
    return r.json()

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def fetch_tic_rows_by_id(
    tic_ids,
    columns=("ID", "lum", "lumclass", "e_lum", "d", "contratio", "numcont", "plx", "Teff", "logg", "MH"),
    chunk_size=1000,
    sleep_s=0.1,
):
    all_rows = []
    batches = list(chunks(list(map(int, tic_ids)), chunk_size))

    for idx, batch in enumerate(batches):
        print(f'Fetching batch {idx+1}/{len(batches)}')
        request = {
            "service": "Mast.Catalogs.Filtered.Tic.Rows",
            "format": "json",
            "params": {
                "columns": ",".join(columns),
                "filters": [
                    {
                        "paramName": "ID",
                        "values": batch
                    }
                ]
            }
        }

        result = mast_query(request)
        rows = result.get("data", [])
        all_rows.extend(rows)

        time.sleep(sleep_s)  # polite throttling

    df = pd.DataFrame(all_rows)

    order = pd.Series(range(len(tic_ids)), index=pd.Index(tic_ids, name="ID"))
    if "ID" in df.columns:
        df["__order"] = df["ID"].map(order)
        df = df.sort_values("__order").drop(columns="__order")

    return df