import pandas as pd
import argparse
from modules.metrics.spike_drop_counter import count_spikes_and_drops
from modules.metrics.total_change import calculate_total_increase

def load_file(filepath):
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format")

    if 'File' not in df.columns or 'Total Dependencies' not in df.columns:
        raise ValueError("Expected columns: 'File', 'Total Dependencies'")
    
    return df

# --- CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize dependency trends in software versions.")
    parser.add_argument("filepath", help="Path to the Excel or CSV file")
    args = parser.parse_args()

    df = load_file(args.filepath)
    df = count_spikes_and_drops(df)
    calculate_total_increase(df)

    print("\nSummary Table:")
    print(df[['File', 'Total Dependencies', '% Change']])
