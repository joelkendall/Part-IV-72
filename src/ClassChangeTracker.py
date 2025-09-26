import pandas as pd
import argparse
from pathlib import Path
from modules.metrics import count_classes, count_source_classes
from TSVReader import extract_version

# --- Main logic
def track_class_dependency_changes(folder_path, output_path="class_changes_summary.txt", use_source=True, categories=None):
    folder = Path(folder_path)
    found_tsv_files = sorted(folder.glob("*.tsv"), key=lambda f: extract_version(f.name))

    if not found_tsv_files:
        print(f"No .tsv files found in: {folder}")
        return

    previous = None

    with open(output_path, "w", encoding="utf-8") as outfile:
        print_information(outfile)
        for tsv in found_tsv_files:
            df = pd.read_csv(tsv, sep="\t", skiprows=26)

            # Apply category filtering if provided
            if categories:
                df = df[df["Category"].isin(categories)]

            current = count_source_classes(df) if use_source else count_classes(df, to_return=True)

            if previous is not None:
                
                # Calculate changed classes
                common = current.index.intersection(previous.index)
                changed = current[common].compare(previous[common])

                # Extract unique changed class names
                changed_class_names = changed.index.get_level_values(0).unique().tolist()

                # Output header
                print_version_header(tsv, outfile)

                # Output changed class names with categories and change type
                if changed_class_names:
                    for name in sorted(changed_class_names):
                        
                        # Calculate change type
                        current_count = current.loc[name].sum() if name in current.index else 0
                        previous_count = previous.loc[name].sum() if name in previous.index else 0

                        change_type = "(+)" if current_count > previous_count else "(-)"
                        
                        
                        class_categories = df[df["# Source"] == name]["Category"].dropna().unique()
                        categories_str = ", ".join(sorted(class_categories))

                        # Result Formatting
                        line = f"{change_type} {name}"
                        if categories:
                            line = f"{change_type} {name} : {categories_str}"
                        
                        # Output
                        print(f" {line}")
                        outfile.write(f" {line}\n")
                else:
                    print(" (No changed classes)")
                    outfile.write(" (No changed classes)\n")

            previous = current

    print(f"\n Summary written to '{output_path}'")

def print_version_header(tsv, outfile):
    version_header = f"\n==== Version {tsv.stem} ===="
    print(version_header.strip())
    outfile.write(version_header + "\n")

def print_information(outfile):
    header = f"\n==== Symbol Information ===="
    information = f" (+) : added Class\n" \
                f" (-) : removed Class\n"
    print(header.strip())
    outfile.write(header + "\n")
    outfile.write(information + "\n")

# --- CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track and print classes with changed dependency counts.")
    parser.add_argument("folder", help="Folder containing .tsv dependency files")
    parser.add_argument("--out", default="class_changes_summary.txt", help="File to write the output")
    parser.add_argument("--source", action="store_true", help="Use source classes (default: source, to match metrics.py)")
    parser.add_argument("--category", nargs='+', help="One or more dependency categories to filter by (e.g., Return Field Parameter)")

    args = parser.parse_args()
    track_class_dependency_changes(args.folder, args.out, use_source=args.source, categories=args.category)
