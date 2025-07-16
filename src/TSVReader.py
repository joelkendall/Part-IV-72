import pandas as pd
import argparse
import sys
import re
from pathlib import Path
from packaging.version import parse
from modules.metrics import compute_metrics
from modules.output_options import print_metrics, write_tsv


# ---- VERSION PARSING FOR SORTING
def extract_version(filename):
    match = re.search(r'(\d+(?:\.\d+)+)', filename)
    return parse(match.group(1)) if match else parse("0")

def process_directory(directory, omit_javalang=False, omit_javaall=False, output='aggregated_output.tsv', suppress_print=False):
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MATCHED_DIRS = [d for d in PROJECT_ROOT.rglob(directory) if d.is_dir()]

    if len(MATCHED_DIRS) != 1:
        yield False, f"{'Multiple' if MATCHED_DIRS else 'No'} directories found for {directory}"
        return

    DIR_PATH = MATCHED_DIRS[0]
    tsv_files = sorted(list(DIR_PATH.glob('*.tsv')), key=lambda f: extract_version(f.name))
    
    if not tsv_files:
        yield False, f"No TSV files found in {DIR_PATH}"
        return

    all_metrics = []
    total_files = len(tsv_files)

    for index, tsv_path in enumerate(tsv_files, 1):
        yield True, f"Processing file {index}/{total_files}"
        
        deps = pd.read_csv(tsv_path, sep='\t', skiprows=26)
        
        if omit_javalang:
            deps = deps[~deps['Target'].str.contains('java.lang')]
        elif omit_javaall:
            deps = deps[~deps['Target'].str.contains('java')]

        metrics = compute_metrics(deps)
        metrics['File'] = tsv_path.name
        all_metrics.append(metrics)

        if not suppress_print:
            print_metrics(metrics)

    write_tsv(all_metrics, output)
    yield True, f"Completed! Results saved to {output}"


# To run via cmd line from src directory:
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

    for success, message in process_directory(
        directory=args.directory,
        omit_javalang=args.oj,
        omit_javaall=args.oja,
        output=args.output,
        suppress_print=args.no_print
    ):
        if not success:
            print(message)
            sys.exit(1)
        print(message)

    

if __name__ == '__main__':
    main()
