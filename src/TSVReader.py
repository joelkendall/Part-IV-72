
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
    parser.add_argument('directory', help='name or path to directory containing the TSV files')
    parser.add_argument('--oj', '--omitjavalang', action='store_true', help='Omit java.lang dependencies')
    parser.add_argument('--oja', '--omitjavaall', action='store_true', help='Omit all java dependencies')

    args = parser.parse_args()

    # ---- CONSTANTS
    directory = Path(args.directory)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MATCHED_DIRS = [d for d in PROJECT_ROOT.rglob(args.directory) if d.is_dir()]
    def separator(): print("-" * 80)

    # ---- FILE RESOLUTION & VALIDATIONS

    if len(MATCHED_DIRS) == 1:
        DIR_PATH = MATCHED_DIRS[0]

    elif len(MATCHED_DIRS) > 1:
        print(f"\nMultiple directories found for {directory}")
        separator()

        shown_suggestions = set()
        
        for match in MATCHED_DIRS:
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
        print(f"Directory {directory} not found")
        sys.exit(1)


    # ---- GETTING TSVs FROM DIRECTORY
    tsv_files = list(DIR_PATH.glob('*.tsv'))
    if not tsv_files:
        print(f"No TSV files found in {DIR_PATH}")
        sys.exit(1)

    for TSV_PATH in tsv_files:
        print(f"Processing {TSV_PATH.relative_to(PROJECT_ROOT)}...")
        # ---- TSV PROCESSING
        # there are hashes in the details columns so we cant use the comment param to omit hashes as comments, just skip the commented lines instead
        # hope all tsv files have the same amount of commented lines otherwise were gonna have to preprocess them
        deps = pd.read_csv(TSV_PATH, sep='\t', skiprows=26) # skiprows=26 as it is for now, but we will need to preprocess for sure.

        # ---- CREATING CORRECT DATAFRAME (new row for each location and removing polymorphic dependencies)
        def correctDataFrame(deps):
            deps.loc[:, 'Locations'] = deps['Locations'].fillna('')
            deps = deps[~deps['Inheritance'].str.contains('Polymorphic', na=False)]
            deps.loc[:, 'Locations'] = deps['Locations'].str.split(',')
            deps= deps.explode('Locations').reset_index(drop=True)
            # renaming duplicate package columns
            deps = deps.rename(columns={
                deps.columns[1]: 'SourcePackage',
                deps.columns[5]: 'TargetPackage'
            })
            return deps
        
        # ---- COUNTING ALL DEPENDENCIES
        def countDependencies(deps):
            deps = correctDataFrame(deps)
            return deps['Locations'].str.split(',').str.len().sum()
        
        # ---- COUNTING DEPENDENCIES BY CATEGORY
        def displayCategories(deps, column, includeNa):
            deps = correctDataFrame(deps)
            categoriesData = deps[column].value_counts(dropna=not includeNa)
            print(f"\n{column} Types:")
            print(categoriesData)
            print(f"Total: {categoriesData.sum()}")

        # ---- COUNTING CLASSES
        # basically all unique source packages also contained in target packages, and then unique target names of those rows
        def countClasses(deps):
            deps = correctDataFrame(deps)
            packages = deps['SourcePackage'].unique()
            print(f"Number of Source Packages: {len(packages)}")
            targetMatches = deps[deps['TargetPackage'].isin(packages)]
            classes = targetMatches['Target'].unique()
            noTests = [c for c in classes if 'Test' not in str(c)] # needed this cause .unique() returns NumPy array not a pandas series
            print(f"Number of Classes: {len(classes)}")
            print(f"Number of Classes (excluding tests): {len(noTests)}")


        # ---- OUTPUT
        if args.oj:
            noJavaLang = deps[~deps['Target'].str.contains('java.lang')]
            print(f"Number of dependencies (excluding java.lang dependencies): {countDependencies(noJavaLang)}")
            displayCategories(noJavaLang, 'Category', True)
            countClasses(noJavaLang)
        elif args.oja:
            noJava = deps[~deps['Target'].str.contains('java')]
            print(f"Number of dependencies (excluding java dependencies): {countDependencies(noJava)}")
            displayCategories(noJava, 'Category', True)
            countClasses(noJava)
        else:
            print(f"Number of dependencies: {countDependencies(deps)}")
            displayCategories(deps, 'Category', True)
            countClasses(deps)
    
    
# Entry point for setup.py
if __name__ == '__main__':
    main()