import argparse
from pathlib import Path
import pandas as pd

from modules.metrics import (
    count_classes,
    count_source_classes,
    correct_dataframe,  
)
from TSVReader import extract_version


def print_information(outfile):
    header = "==== Symbol Information ===="
    legend = " (+) : added Class\n (-) : removed Class\n"
    print(header)
    outfile.write(header + "\n")
    outfile.write(legend + "\n")


def print_version_header(tsv_path: Path, outfile):
    version_header = f"\n==== Version {tsv_path.stem} ===="
    print(version_header.strip())
    outfile.write(version_header + "\n")


def build_per_category(df_norm: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build per-class-per-category counts from the **normalized** dataframe,
    so the row keys (class names) match those used by count_*() outputs.
    After correct_dataframe(), the class column is 'Source' and categories are in 'Category'.
    """
    if "Source" in df_norm.columns and "Category" in df_norm.columns:
        return (
            df_norm
            .groupby(["Source", "Category"])
            .size()
            .unstack(fill_value=0)
        )
    return None


def track_class_dependency_changes(
    folder_path,
    output_path="class_changes_summary.txt",
    use_source=True,
    categories=None
):
    folder = Path(folder_path)
    found_tsv_files = sorted(folder.glob("*.tsv"), key=lambda f: extract_version(f.name))

    if not found_tsv_files:
        print(f"No .tsv files found in: {folder}")
        return

    previous_totals: pd.Series | None = None   # per-class totals (Series)
    previous_cats: pd.DataFrame | None = None  # per-class per-category total (DataFrame)

    with open(output_path, "w", encoding="utf-8") as outfile:
        print_information(outfile)

        for tsv in found_tsv_files:
            df_raw = pd.read_csv(tsv, sep="\t", skiprows=26)

            # Apply user category filter (union of selected categories)
            df_filtered = (
                df_raw[df_raw["Category"].isin(categories)]
                if categories
                else df_raw
            )

            # Normalize dataframe so it can be used with count_*() methods in metrics.py
            df_norm = correct_dataframe(df_filtered.copy())

            
            if use_source:
                current_totals = count_source_classes(df_norm) # track package names only
            else:
                current_totals = count_classes(df_norm, to_return=True) # track full class names

            # Build per-category table for this version using the same df
            current_cats = build_per_category(df_norm)

            # Build per-category table for the previous version
            if previous_totals is not None:
                # only classes present in both are comparable for .compare()
                common = current_totals.index.intersection(previous_totals.index)
                changed = current_totals[common].compare(previous_totals[common])
                changed_class_names = changed.index.get_level_values(0).unique().tolist()
                # capture classes that are new or removed (not in 'common')
                new_class_names = current_totals.index.difference(previous_totals.index).tolist()
                removed_class_names = previous_totals.index.difference(current_totals.index).tolist()


                print_version_header(tsv, outfile)
                print_new_classes(categories, outfile, current_cats, new_class_names)
                print_removed_classes(categories, previous_cats, outfile, removed_class_names)

                
                print_common_class_changes(categories, previous_totals, previous_cats, outfile, current_totals, current_cats, changed_class_names)

            # Carry forward for next version
            previous_totals = current_totals
            previous_cats = current_cats

    print(f"\n Summary written to '{output_path}'")

def print_common_class_changes(categories, previous_totals, previous_cats, outfile, current_totals, current_cats, changed_class_names):
    if changed_class_names:
        for class_x in sorted(changed_class_names):
                        # track total class changes and determine if class is added or removed
            cur_total = current_totals.loc[class_x] if class_x in current_totals.index else 0
            prev_total = previous_totals.loc[class_x] if class_x in previous_totals.index else 0
            delta_total = cur_total - prev_total
            sign = "(+)" if delta_total > 0 else "(-)" if delta_total < 0 else "(±)"

                        # idenfiy categories that changed for this class
            cats_changed = []
            if current_cats is not None and previous_cats is not None:
                            # Get per-category counts for this class (0s if class not present in current version)
                cur_row = (
                                current_cats.loc[class_x]
                                if class_x in current_cats.index
                                else pd.Series(0, index=current_cats.columns)
                            )
                prev_row = (
                                previous_cats.loc[class_x]
                                if class_x in previous_cats.index
                                else pd.Series(0, index=previous_cats.columns)
                            )
                            
                            #c combine all unique category names that exist in either
                all_cols = sorted(set(cur_row.index) | set(prev_row.index))

                            # fill with 0s for any missing categories 
                cur_row = cur_row.reindex(all_cols, fill_value=0)
                prev_row = prev_row.reindex(all_cols, fill_value=0)

                            # Determine which categories to compare for this class
                            # if user did not specify categories, compare all
                cols_to_check = [c for c in all_cols if (not categories) or (c in categories)]

                            # Determine which categories have changed
                cats_changed = [c for c in cols_to_check if cur_row[c] != prev_row[c]]

                        # Format output: 
                        # defult: signs and class name only
                        # with categories: signs, class name, and changed categories involved
                        # no changes: (No changed classes)
            if cats_changed:
                cats_str = ", ".join(sorted(cats_changed))
                line = f"{sign} {class_x} : {cats_str}"
            else:
                line = f"{sign} {class_x}"

            print(f" {line}")
            outfile.write(f" {line}\n")
    else:
        print(" (No changed classes)")
        outfile.write(" (No changed classes)\n")

def print_removed_classes(categories, previous_cats, outfile, removed_class_names):
    for class_x in sorted(removed_class_names):
        cats_str = ""
        if previous_cats is not None:
                        # categories the class had in the previous version
            prev_row = previous_cats.loc[class_x] if class_x in previous_cats.index else None
            if prev_row is not None:
                cols = categories if categories else prev_row.index.tolist()
                changed_cols = [c for c in cols if prev_row.get(c, 0) != 0]
                if changed_cols:
                    cats_str = " : " + ", ".join(sorted(changed_cols))
        line = f"(-) {class_x}{cats_str}"
        print(f" {line}")
        outfile.write(f" {line}\n")

def print_new_classes(categories, outfile, current_cats, new_class_names):
    for class_x in sorted(new_class_names):
        cats_str = ""
        if current_cats is not None:
                        # categories present for this new class in the current version
            cur_row = current_cats.loc[class_x] if class_x in current_cats.index else None
            if cur_row is not None:
                            # if user passed --category, only show those; else show any non-zero
                cols = categories if categories else cur_row.index.tolist()
                changed_cols = [c for c in cols if cur_row.get(c, 0) != 0]
                if changed_cols:
                    cats_str = " : " + ", ".join(sorted(changed_cols))
        line = f"(+) {class_x}{cats_str}"
        print(f" {line}")
        outfile.write(f" {line}\n")


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track and print classes with changed dependency counts.")
    parser.add_argument("folder", help="Folder containing .tsv dependency files")
    parser.add_argument("--out", default="class_changes_summary.txt", help="File to write the output")
    parser.add_argument(
        "--source",
        action="store_true",
        help="Use source classes (default: source, to match metrics.py)"
    )
    parser.add_argument(
        "--category",
        nargs="+",
        help="One or more dependency categories to filter by (e.g., Return Field Parameter)"
    )
    args = parser.parse_args()

    track_class_dependency_changes(
        args.folder,
        args.out,
        use_source=args.source,
        categories=args.category
    )
