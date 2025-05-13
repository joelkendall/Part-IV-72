import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Count the number of dependencied in a given TSV file')
parser.add_argument('--oj', '--omitjavalang', action='store_true', help='Omit java.lang dependencies')
parser.add_argument('--oja', '--omitjavaall', action='store_true', help='Omit all java dependencies')

args = parser.parse_args()

# there are hashes in the details columns so we cant use the comment param to omit hashes as comments, just skip the commented lines instead
# hope all tsv files have the same amount of commented lines otherwise were gonna have to preprocess them
deps = pd.read_csv('data/junit-depfiles/junit-depfiles/junit-2.0-deps.tsv', sep='\t', skiprows=26)

def countDependencies(deps):
    deps.loc[:, 'Locations'] = deps['Locations'].fillna('')
    return deps['Locations'].str.split(',').str.len().sum()


if args.oj:
    noJavaLang = deps[~deps['Target'].str.contains('java.lang')]
    print(f"Number of dependencies (excluding java.lang dependencies): {countDependencies(noJavaLang)}")
elif args.oja:
    noJava = deps[~deps['Target'].str.contains('java')]
    print(f"Number of dependencies (excluding java dependencies): {countDependencies(noJava)}")
else:
    print(f"Number of dependencies: {countDependencies(deps)}")




