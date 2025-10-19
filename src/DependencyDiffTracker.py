import pandas as pd
import argparse
from pathlib import Path
from TSVReader import extract_version

# --- Main logic
def track_dependency_differences(folder_path, output_path="dependency_diffs.txt", categories=None):
    folder = Path(folder_path)
    found_tsv_files = sorted(folder.glob("*.tsv"), key=lambda f: extract_version(f.name))

    if not found_tsv_files:
        print(f"No .tsv files found in: {folder}")
        return

    previous_df = None

    with open(output_path, "w", encoding="utf-8") as outfile:
        for tsv in found_tsv_files:
            current_version = tsv.stem
            df = pd.read_csv(tsv, sep="\t", skiprows=26)

            if categories:
                df = df[df["Category"].isin(categories)]

            # Columns to compare and extract data from.
            compare_cols = ["# Source", "Target", "Category", "Details"] # "# Source" is done deliberately due to TSV file format.
            df_clean = df[compare_cols].fillna("").applymap(str.strip) 

            if previous_df is not None:
                merged = previous_df.merge(df_clean, how='outer', indicator=True)
                diffs = merged[merged['_merge'] != 'both']

                # Print Version Change Header
                version_header = f"\n==== Diff: {previous_version} -> {current_version} ===="
                print(version_header)
                outfile.write(version_header + "\n")
                
                if diffs.empty: # Print no changes
                    print(" (No changes in dependencies)")
                    outfile.write(" (No changes in dependencies)\n")
                else: # Print changes
                    added = diffs[diffs['_merge'] == 'right_only']
                    removed = diffs[diffs['_merge'] == 'left_only']

                    for label, group in [("ADDED", added), ("REMOVED", removed)]:
                        if not group.empty:
                            header = f" [{label}]"
                            print(header)
                            outfile.write(header + "\n")


                            for _, row in group.iterrows():
                                msg = f" {row['# Source']} -> {row['Target']} :: {row['Category']} | {row['Details']}"
                                print(msg)
                                outfile.write(msg + "\n")
                    
            previous_df = df_clean
            previous_version = current_version

    print(f"\n Detailed dependency diffs written to '{output_path}'")


# --- CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track detailed dependency changes across TSV versions.")
    parser.add_argument("folder", help="Folder containing .tsv dependency files")
    parser.add_argument("--out", default="dependency_diffs.txt", help="Output file path")
    parser.add_argument("--category", nargs='+', help="Dependency categories to include (e.g., Return Cast Field)")

    args = parser.parse_args()
    track_dependency_differences(args.folder, args.out, categories=args.category)