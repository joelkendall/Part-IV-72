
import pandas as pd
import argparse
import os 
import sys
from pathlib import Path

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
    IS_DIRECT_PATH = os.path.exists(FILE_OBTAINED)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MATCHED_FILES = list(PROJECT_ROOT.rglob(FILE_OBTAINED)) if not IS_DIRECT_PATH else []
    def separator(): print("-" * 80)

    # ---- FILE RESOLUTION & VALIDATIONS
    if IS_DIRECT_PATH: 
        TSV_PATH = Path(FILE_OBTAINED).resolve()

    elif len(MATCHED_FILES) == 1:
        TSV_PATH = MATCHED_FILES[0]

    elif len(MATCHED_FILES) > 1:
        print(f"\nMultiple TSV files found for {FILE_OBTAINED}")
        separator()

        shown_suggestions = set()
        
        for match in MATCHED_FILES:
            rel_path = match.relative_to(PROJECT_ROOT)
            suggestion = Path(match.parent.name) / match.name
            
            #Avoid showing the same suggestion multiple times
            if suggestion in shown_suggestions: continue

            #Option 1: without setup.py
            print(f"- {rel_path} \nTry: python src/TSVReader.py {suggestion}")
            print("OR")
            #Option 2: with setup.py
            print(f"Try: tsvreader {suggestion}")
            separator()

            shown_suggestions.add(suggestion)
        sys.exit(1)
    else: ## FILE NOT FOUND
        print(f"TSV file {FILE_OBTAINED} not found")
        sys.exit(1)


    # ---- TSV PROCESSING
    # there are hashes in the details columns so we cant use the comment param to omit hashes as comments, just skip the commented lines instead
    # hope all tsv files have the same amount of commented lines otherwise were gonna have to preprocess them
    deps = pd.read_csv(TSV_PATH, sep='\t', skiprows=26) # skiprows=26 as it is for now, but we will need to preprocess for sure.

    # ---- COUNTING ALL DEPENDENCIES
    def countDependencies(deps):
        deps.loc[:, 'Locations'] = deps['Locations'].fillna('')
        deps = deps[~deps['Inheritance'].str.contains('Polymorphic', na=False)]
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