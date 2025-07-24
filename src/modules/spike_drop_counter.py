# spike_and_drop_counter.py
def count_spikes_and_drops(df):
    """
    Calculates and prints the number of spikes and drops in the percentage change of total dependencies (>= 10%, <= -10%).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing dependency data with a 'Total Dependencies' column.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an added '% Change' column that contains the calculated percentage changes.
    """

    df['% Change'] = df['Total Dependencies'].pct_change() * 100
    df['% Change'] = df['% Change'].round(2)

    spikes = (df['% Change'] >= 10).sum()
    drops = (df['% Change'] <= -10).sum()

    print(f"Spikes (≥ +10%): {spikes}")
    print(f"Drops (≤ -10%): {drops}")

    return df
