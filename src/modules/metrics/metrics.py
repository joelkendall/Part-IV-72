import pandas as pd

previous_class_data = None

# ---- CREATING CORRECT DATAFRAME (new row for each location and removing polymorphic dependencies)
def correct_dataframe(deps):
    deps.loc[:, 'Locations'] = deps['Locations'].fillna('')
    deps = deps[~deps['Inheritance'].str.contains('Polymorphic', na=False)]
    deps.loc[:, 'Locations'] = deps['Locations'].str.split(',')
    deps = deps.explode('Locations').reset_index(drop=True)
    # renaming duplicate package columns
    deps = deps.rename(columns={
        deps.columns[0]: 'Source',
        deps.columns[1]: 'SourcePackage',
        deps.columns[5]: 'TargetPackage'
    })
    return deps

def count_dependencies(deps):
    deps = correct_dataframe(deps)
    return deps['Locations'].str.split(',').str.len().sum()

def count_classes(deps, to_return=False):
    """
    Counts the number of unique source packages, classes, and classes excluding tests in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.
    to_return : bool, optional
        If True, returns a pandas.Series of classes instead of counts. Default is False.

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
    no_tests = [c for c in classes if 'Test' not in str(c)] # needed this cause .unique() returns NumPy array not a pandas series
    if to_return:
        return target_matches['Target'].value_counts(dropna=False)
    else:
        return len(packages), len(classes), len(no_tests)

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
    # fill NaN values with 'Unknown'
    counts = deps['Category'].fillna('Unknown').value_counts(dropna=False)
    return counts.to_dict()

def count_source_classes(deps):
    """
    Helper method to count the number of unique source classes in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.

    Returns
    -------
    pandas.Series
        The count of unique source classes in the 'SourcePackage' column.
    """
    
    deps = correct_dataframe(deps)
    counts = deps['Source'].value_counts()
    return counts

def count_class_changes(deps, previous, source):
    """
    Counts the number of class changes since the last release in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.
    previous : pandas.Series or None
        The previous release's class data. If None, this is the first iteration and no changes
    source : boolean
        If true, uses the source class counting method, otherwise uses the target class counting method.

    Returns
    -------
    tuple
        A tuple containing:
        - The number of new classes since the last release.
        - The number of removed classes since the last release.
        - The number of changed classes since the last release.
    """
    
    deps = correct_dataframe(deps)
    current_counts = count_source_classes(deps) if source else count_classes(deps, True)
    global previous_class_data
    if previous is not None:
        new_classes = current_counts[~current_counts.index.isin(previous.index)]
        removed_classes = previous[~previous.index.isin(current_counts.index)]
        common_classes = current_counts.index.intersection(previous.index)
        changed_counts = current_counts[common_classes].compare(previous[common_classes])
        previous_class_data = current_counts
        return len(new_classes), len(removed_classes), len(changed_counts)
    
    previous_class_data = current_counts
    return 0, 0, 0

def average_dependencies_per_class(deps):
    """
    Calculates the average number of dependencies per class in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.

    Returns
    -------
    int
        The average number of dependencies per class.
    """
    
    deps = correct_dataframe(deps)
    total_deps = count_dependencies(deps)
    num_classes = count_classes(deps)[1]
    
    if num_classes == 0:
        return 0.0
    
    return int(total_deps / num_classes)

def count_dependencies_per_class(deps):
    """
    Counts the number of dependencies for each class in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.

    Returns
    -------
    dict
        dictionary where keys are class names and values are the number of dependencies for each class.
    """
    
    deps = correct_dataframe(deps)
    counts = deps['Source'].value_counts()
    return counts.to_dict()

def categories_per_class(deps):
    """
    Counts the number of occurrences of each category for each class in the given dependency data.

    Parameters
    ----------
    deps : pandas.DataFrame
        The dependency data to analyze.
    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a class and each column represents a category, with counts as values.
    """
    deps = correct_dataframe(deps)
    category_percentages = deps.groupby('Source')['Category'].value_counts(normalize=True).unstack(fill_value=0)
    return category_percentages
    
def compute_metrics(deps):
    total_deps = count_dependencies(deps)
    num_pkgs, num_classes, num_no_tests = count_classes(deps)
    num_methods = count_methods(deps)
    category_counts = count_by_category(deps)
    new_classes, removed_classes, changed_classes = count_class_changes(deps, previous_class_data, True)
    deps_per_class = count_dependencies_per_class(deps)

    return {
        'Total Dependencies': total_deps,
        'Num Source Packages': num_pkgs,
        'Num Classes': num_classes,
        'Num Classes (No Tests)': num_no_tests,
        'Num Methods (No Constructors)': num_methods,
        'Average Dependencies per Class': average_dependencies_per_class(deps),
        'Class Changes': changed_classes,
        'New Classes': new_classes,
        'Removed Classes': removed_classes,
        **category_counts,
        **deps_per_class
    }

def compute_class_metrics(deps):
    all_classes = count_dependencies_per_class(deps)
    cat_percents = categories_per_class(deps)



