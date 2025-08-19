import pandas as pd
import argparse
from pathlib import Path
from packaging.version import parse
from modules.metrics import count_classes, count_source_classes
from TSVReader import extract_version


# --- Main logic
def track_class_dependency_changes(folder_path, output_path="class_changes_summary.txt", use_source=True):
    folder = Path(folder_path)
    tsv_files = sorted(folder.glob("*.tsv"), key=lambda f: extract_version(f.name))

    if not tsv_files:
        print(f"No .tsv files found in: {folder}")
        return

    previous = None
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tsv in tsv_files:
            df = pd.read_csv(tsv, sep="\t", skiprows=26)
            current = count_source_classes(df) if use_source else count_classes(df, to_return=True)

            if previous is not None:
                common = current.index.intersection(previous.index)
                changed = current[common].compare(previous[common])

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

            previous = current

    print(f"\n Summary written to '{output_path}'")

# --- CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track and print classes with changed dependency counts.")
    parser.add_argument("folder", help="Folder containing .tsv dependency files")
    parser.add_argument("--out", default="class_changes_summary.txt", help="File to write the output")
    parser.add_argument("--source", action="store_true", help="Use source classes (default: source, to match metrics.py)")

    args = parser.parse_args()
    track_class_dependency_changes(args.folder, args.out, use_source=args.source)
