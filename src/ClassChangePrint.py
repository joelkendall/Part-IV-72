import pandas as pd
from pathlib import Path
from packaging.version import parse
from modules.metrics.metrics import count_classes, count_source_classes
from TSVReader import extract_version
from utils.ChangeTracker import ChangeTracker

# --- Main logic
def track_class_dependency_changes(folder_path, output_path="class_changes_summary.txt", use_source=True, return_df=False):
    folder = Path(folder_path)
    tsv_files = sorted(folder.glob("*.tsv"), key=lambda f: extract_version(f.name))

    if not tsv_files:
        print(f"No .tsv files found in: {folder}")
        return

    previous_names = None
    tracker = ChangeTracker()
    if not return_df:
        with open(output_path, "w", encoding="utf-8") as outfile:
            for tsv in tsv_files:
                df = pd.read_csv(tsv, sep="\t", skiprows=26)
                current = count_source_classes(df) if use_source else count_classes(df, to_return=True)

                if previous_names is not None:
                    common = current.index.intersection(previous_names.index)
                    changed = current[common].compare(previous_names[common])

                    # Extract unique changed class names
                    changed_class_names = changed.index.get_level_values(0).unique().tolist()

                    
                    # Output header
                    version_header = f"\n==== Version {tsv.stem} ====\n"
                    print(version_header.strip())
                    outfile.write(version_header)

                    # Output changed class names
                    if changed_class_names:
                        for name in sorted(changed_class_names):
                            print(f" - {name}")
                            outfile.write(f" - {name}\n")
                    else:
                        print(" (No changed classes)")
                        outfile.write(" (No changed classes)\n")

                previous_names = current
    else:
        for tsv in tsv_files:
            df = pd.read_csv(tsv, sep="\t", skiprows=26)
            tracker.add_release(tsv.stem, df, use_source=use_source)
        tracker.print_release_summary()


    print(f"\n Summary written to '{output_path}'")

# --- CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track and print classes with changed dependency counts.")
    parser.add_argument("folder", help="Folder containing .tsv dependency files")
    parser.add_argument("--out", default="class_changes_summary.txt", help="File to write the output")
    parser.add_argument("--source", action="store_true", help="Use source classes (default: source, to match metrics.py)")
    parser.add_argument("--return_df", action="store_true", help="Export DataFrame for class counting (default: False)")

    args = parser.parse_args()
    track_class_dependency_changes(args.folder, args.out, use_source=args.source, return_df=args.return_df)
