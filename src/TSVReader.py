import pandas as pd
import argparse
import sys
from pathlib import Path
from modules.metrics import compute_metrics
from modules.output_options import print_metrics, write_tsv

# To run this file:
# python TSVReader.py <directory>
def main():
    # ---- ARGUMENTS
    parser = argparse.ArgumentParser(description='Analyze dependencies from TSV files')
    parser.add_argument('directory', help='Path to directory containing the TSV files')
    parser.add_argument('--oj', '--omitjavalang', action='store_true', help='Omit java.lang dependencies')
    parser.add_argument('--oja', '--omitjavaall', action='store_true', help='Omit all java dependencies')
    parser.add_argument('--output', default='aggregated_output.tsv', help='TSV output file')
    parser.add_argument('--no-print', action='store_true', help='Suppress terminal output')

    args = parser.parse_args()

    # ---- FILE RESOLUTION
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MATCHED_DIRS = [d for d in PROJECT_ROOT.rglob(args.directory) if d.is_dir()]

    # Print that multiple directories were found and exit
    if len(MATCHED_DIRS) != 1:
        print("\n" + ("Multiple" if MATCHED_DIRS else "No") + f" directories found for {args.directory}")
        sys.exit(1)

    # No TSV files found
    DIR_PATH = MATCHED_DIRS[0]
    tsv_files = list(DIR_PATH.glob('*.tsv'))
    if not tsv_files:
        print(f"No TSV files found in {DIR_PATH}")
        sys.exit(1)


    # ---- MAIN
    all_metrics = []

    for TSV_PATH in tsv_files:
        # ---- TSV PROCESSING
        # there are hashes in the details columns so we cant use the comment param to omit hashes as comments, just skip the commented lines instead
        # hope all tsv files have the same amount of commented lines otherwise were gonna have to preprocess them
        print(f"Processing {TSV_PATH.relative_to(PROJECT_ROOT)}...")
        deps = pd.read_csv(TSV_PATH, sep='\t', skiprows=26)

        if args.oj:
            deps = deps[~deps['Target'].str.contains('java.lang')]
        elif args.oja:
            deps = deps[~deps['Target'].str.contains('java')]

        metrics = compute_metrics(deps)
        metrics['File'] = TSV_PATH.name
        all_metrics.append(metrics)

        if not args.no_print:
            print_metrics(metrics)

    write_tsv(all_metrics, args.output)
    print(f"\n Aggregated results saved to {args.output}")

if __name__ == '__main__':
    main()
