"""Add a "true_label" column to a CSV file with astronet-vetting results."""

from argparse import ArgumentParser
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd
import re


def parse_args():
    parser = ArgumentParser(description='Add a "true_label" column to a CSV file with astronet-vetting results')
    parser.add_argument("tce_table", type=Path, help="CSV file with astronet-vetting results")
    parser.add_argument("-o", "--outfile", type=Path, help="CSV file to write results with labels")
    parser.add_argument("-s", "--sector", type=int, help="Sector of data being analyzed")
    parser.add_argument("-d", "--delivery", type=Path, help="Directory with TEV delivery to use as true labels")

    args = parser.parse_args()
    if args.sector is None:
        # Detect sector from current directory/parents
        current_directory = Path().resolve()
        sector_pattern = re.compile(r"^sector-(\d+)$")
        while not sector_pattern.match(current_directory.name):
            current_directory = current_directory.parent
            if current_directory == current_directory.parent:
                parser.error("Could not determine sector from current directory")
        args.sector = int(sector_pattern.match(current_directory.name)[1])
    if args.outfile is None:
        # Write results to "{result_file}_with_labels.csv"
        args.outfile = args.tce_table.parent / f"{args.tce_table.stem}_with_labels{args.tce_table.suffix}"

    return args

def get_delivered_candidates(sector: int, delivery_directory: Optional[Path]):
    if delivery_directory is None:
        delivery_directory = Path(f"/pdo/qlp-data/tev/qlp-delivery/sector-{sector}/batch3")
    cand_data = pd.read_csv(delivery_directory / "cand.csv", index_col=("star_tic", "planet_planetno"))
    return list(cand_data.index.values)

def convert_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    def to_snake_case(name: str) -> str:
        # Replace spaces, dashes, parentheses with underscores
        name = re.sub(r'[\s\-\(\)]+', '_', name)
        # Convert CamelCase or PascalCase to snake_case
        name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name)
        name = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', name)
        # Lowercase and cleanup
        name = name.lower()
        name = re.sub(r'_+', '_', name)
        return name.strip('_')
    
    df = df.copy()
    df.columns = [to_snake_case(col) for col in df.columns]
    return df

if __name__ == "__main__":
    args = parse_args()
    print(f"Reading astronet-vetting results from {args.tce_table.resolve()}")
    tce_table = pd.read_csv(args.tce_table, index_col=0)
    # predictions = astronet_results.groupby("Astro ID").mean().reset_index()
    # predictions["pred_label"] = predictions[["disp_p", "disp_e", "disp_n", "disp_j"]].idxmax(axis=1)
    # print(f"Found predictions for {len(predictions)} candidates")
    # results_with_predictions = pd.merge(astronet_results, predictions[["Astro ID", "pred_label"]], how="left", on="Astro ID")

    delivered_candidates = get_delivered_candidates(args.sector, args.delivery)
    print(f"Found {len(delivered_candidates)} delivered candidates from QLP")
    tce_table = tce_table.reset_index()
    were_results_delivered = tce_table.apply(lambda row: (row["TIC ID"], row["planetno"]) in delivered_candidates, axis=1)
    print('Setting "true_label" column to "p" if candidate was delivered, otherwise "j"')
    tce_table["true_label"] = np.where(were_results_delivered, "p", "j")
    print(f"Saving results with predicted and true labels to {args.outfile.resolve()}")
    tce_table = convert_columns_to_snake_case(tce_table)
    tce_table.to_csv(args.outfile)

