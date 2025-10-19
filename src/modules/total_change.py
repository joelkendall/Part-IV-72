
def calculate_total_increase(df):
    """
    Calculate the total percentage change in dependencies from the first to the last version
    in the given DataFrame of dependency data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dependency data to analyze.

    Returns
    -------
    float
        The total percentage change in dependencies from the first to the last version.
    """
    first = df['Total Dependencies'].iloc[0]
    last = df['Total Dependencies'].iloc[-1]
    overall_percent_change = ((last - first) / first) * 100
    print(f"Overall % increase from first to last version: {overall_percent_change:.2f}%")
    return overall_percent_change
