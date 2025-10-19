import pandas as pd
import argparse
from pathlib import Path
from modules.spike_drop_counter import count_spikes_and_drops
from modules.total_change import calculate_total_increase

def load_file(filepath: str) -> pd.DataFrame:
    if filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format (use .csv, .xlsx, or .xls)")

    required = {'File', 'Total Dependencies'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")
    return df

def infer_default_out(input_path: str) -> str:
    p = Path(input_path)
    return str(p.with_name(p.stem + "_summary.csv"))

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    # Map File -> Version for output naming
    if 'File' in df.columns:
        df = df.rename(columns={'File': 'Version'})
    # Keep these columns in the output
    for c in ['Version', 'Total Dependencies', '% Change']:
        if c in df.columns:
            cols.append(c)
    if not cols:
        raise ValueError("No summary columns found (need one of: Version, Total Dependencies, % Change)")
    return df[cols]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize dependency trends in software versions.")
    parser.add_argument("filepath", help="Path to the Excel or CSV file")
    parser.add_argument("--out", help="Output path for the summary table (.csv or .xlsx)")
    args = parser.parse_args()

    # Load & compute
    df = load_file(args.filepath)
    df = count_spikes_and_drops(df)
    calculate_total_increase(df)
    summary = build_summary(df)

    # Save (CSV or XLSX, inferred by extension; default is CSV)
    out_path = args.out or infer_default_out(args.filepath)
    ext = Path(out_path).suffix.lower()
    if ext == ".xlsx":
        summary.to_excel(out_path, index=False)
    elif ext == ".csv" or ext == "":
        if ext == "":
            out_path += ".csv"
        summary.to_csv(out_path, index=False)
    else:
        raise ValueError("Output must be .csv or .xlsx")

    print("\nSummary Table:")
    print(summary)
    print(f"\nSaved summary to: {out_path}")
