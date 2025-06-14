import pandas as pd

# ---- CREATING CORRECT DATAFRAME (new row for each location and removing polymorphic dependencies)
def correct_dataframe(deps):
    deps.loc[:, 'Locations'] = deps['Locations'].fillna('')
    deps = deps[~deps['Inheritance'].str.contains('Polymorphic', na=False)]
    deps.loc[:, 'Locations'] = deps['Locations'].str.split(',')
    deps = deps.explode('Locations').reset_index(drop=True)
    deps = deps.rename(columns={
        deps.columns[1]: 'SourcePackage',
        deps.columns[5]: 'TargetPackage'
    })
    return deps

def count_dependencies(deps):
    deps = correct_dataframe(deps)
    return deps['Locations'].str.split(',').str.len().sum()

def count_classes(deps):
    """
    Counts the number of unique source packages, classes, and classes excluding tests in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.

    Returns
    -------
    tuple
        A tuple containing:
        - The number of unique source packages.
        - The number of unique classes.
        - The number of unique classes excluding those containing 'Test'.
    """

    deps = correct_dataframe(deps)
    packages = deps['SourcePackage'].unique()
    target_matches = deps[deps['TargetPackage'].isin(packages)]
    classes = target_matches['Target'].unique()
    no_tests = [c for c in classes if 'Test' not in str(c)]
    return len(packages), len(classes), len(no_tests)

  # ---- COUNTING METHODS
# count all the number of rows with Return as Category, then remove the rows from the count with init keyword
def count_methods(deps):
    """
    Counts the number of methods in the given dependency data. Removes constructor methods via the 'init' keyword.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to count methods from.

    Returns
    -------
    int
        The number of methods in the given data.
    """
    deps = correct_dataframe(deps)
    return_deps = deps[deps['Category'] == 'Return']
    non_constructors = return_deps[~return_deps['Details'].str.contains('init', na=False)]
    return len(non_constructors)

def count_by_category(deps):
    """
    Counts the number of occurrences of each category in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.

    Returns
    -------
    dict
        A dictionary where keys are categories and values are their respective counts, including NaN values.
    """

    deps = correct_dataframe(deps)
    return deps['Category'].value_counts(dropna=False).to_dict()

def compute_metrics(deps):
    total_deps = count_dependencies(deps)
    num_pkgs, num_classes, num_no_tests = count_classes(deps)
    num_methods = count_methods(deps)
    category_counts = count_by_category(deps)

    return {
        'Total Dependencies': total_deps,
        'Num Source Packages': num_pkgs,
        'Num Classes': num_classes,
        'Num Classes (No Tests)': num_no_tests,
        'Num Methods (No Constructors)': num_methods,
        **category_counts
    }
