
import pandas as pd
import argparse
import os 
import sys
from pathlib import Path
from utils.fileResolver import FileResolver
from utils.exceptions import TSVFileNotFoundError, TSVMultipleFilesFoundError

# To run this file:
# python TSVReader.py <path_to_tsv_file> 
def main():
    # ---- ARGUMENTS
    parser = argparse.ArgumentParser(description='Count the number of dependencied in a given TSV file')
    parser.add_argument('file', help='TSV file name or path')
    parser.add_argument('--oj', '--omitjavalang', action='store_true', help='Omit java.lang dependencies')
    parser.add_argument('--oja', '--omitjavaall', action='store_true', help='Omit all java dependencies')

    args = parser.parse_args()

    # ---- CONSTANTS
    FILE_OBTAINED = args.file
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # ---- FILE RESOLUTION
    try: 
        TSV_PATH = FileResolver(FILE_OBTAINED, PROJECT_ROOT).resolve()
    except TSVFileNotFoundError as e:
        print(f"{e}")
        sys.exit(1)

    except TSVMultipleFilesFoundError as e:
        print(f"{e}")
        sys.exit(1)

    # ---- TSV PROCESSING
    # there are hashes in the details columns so we cant use the comment param to omit hashes as comments, just skip the commented lines instead
    # hope all tsv files have the same amount of commented lines otherwise were gonna have to preprocess them
    deps = pd.read_csv(TSV_PATH, sep='\t', skiprows=26) # skiprows=26 as it is for now, but we will need to preprocess for sure.

    def countDependencies(deps):
        deps.loc[:, 'Locations'] = deps['Locations'].fillna('')
        return deps['Locations'].str.split(',').str.len().sum()

    # ---- OUTPUT
    if args.oj:
        noJavaLang = deps[~deps['Target'].str.contains('java.lang')]
        print(f"Number of dependencies (excluding java.lang dependencies): {countDependencies(noJavaLang)}")
    elif args.oja:
        noJava = deps[~deps['Target'].str.contains('java')]
        print(f"Number of dependencies (excluding java dependencies): {countDependencies(noJava)}")
    else:
        print(f"Number of dependencies: {countDependencies(deps)}")

# Entry point for setup.py
if __name__ == '__main__':
    main()