import pandas as pd
import argparse
from itertools import combinations
from utils.excel_utils import ExcelUtils

def calculate_spearman_correlation(df, col1, col2):
    """
    Calculate the Spearman correlation between two columns in the given DataFrame.
    """
    return df[col1].corr(df[col2], method='spearman')

def generate_spearman_correlation_matrix(file_path, sheet_name=1):
    """
    Reads an Excel file, calculates Spearman correlations between all numeric column pairs,
    and returns the correlation matrix.
    
    Parameters
    ----------
    file_path : str
        Path to the Excel file.
    sheet_name : int or str
        Sheet index or name to read.

    Returns
    -------
    pandas.DataFrame
        Spearman correlation matrix (symmetric).
    """
    df = ExcelUtils.load_with_prompt(file_path)
    numeric_cols = df.select_dtypes(include='number').columns

    # Initialize empty correlation matrix
    corr_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=float)

    for col1, col2 in combinations(numeric_cols, 2):
        corr = calculate_spearman_correlation(df, col1, col2)
        corr_matrix.at[col1, col2] = corr
        corr_matrix.at[col2, col1] = corr

    # Fill diagonal with 1.0 (self-correlation)
    for col in numeric_cols:
        corr_matrix.at[col, col] = 1.0

    return corr_matrix

# --- CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze correlation coefficients between categories from TSV files')
    parser.add_argument('directory', help='Path to directory containing the TSV files')
    args = parser.parse_args()

    file_path = args.directory
    corr_matrix = generate_spearman_correlation_matrix(file_path)
    # Save correlation matrix to a TSV file
    corr_matrix.to_csv('spearman_correlation_matrix.tsv', sep='\t', float_format='%.6f')
    
    print(corr_matrix.round(3)) 

