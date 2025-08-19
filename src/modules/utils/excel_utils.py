import pandas as pd
from typing import List, Union, Optional
from pathlib import Path

class ExcelUtils:
    """
    Utilities for loading tabular data from Excel/CSV, including interactive
    sheet selection for Excel workbooks.

    Requires: pandas (and openpyxl for .xlsx/.xls)
    """

    # ---------- Sheet discovery ----------

    @staticmethod
    def list_sheets(file_path: Union[str, Path]) -> List[str]:
        """
        Return a list of sheet names for an Excel file.
        Raises if the file is not Excel.
        """
        path = Path(file_path)
        if path.suffix.lower() not in (".xlsx", ".xls"):
            raise ValueError(f"Not an Excel file: {path}")
        xls = pd.ExcelFile(path)
        return xls.sheet_names

    @staticmethod
    def resolve_sheet(
        choice: Union[str, int],
        sheet_names: List[str],
    ) -> Union[str, int]:
        """
        Resolve a user-provided choice (index or name) to a valid sheet.
        Returns the original int index or the exact string name.
        Raises ValueError if invalid.
        """
        # If choice already an int, validate range
        if isinstance(choice, int):
            if 0 <= choice < len(sheet_names):
                return choice
            raise ValueError(f"Sheet index out of range: {choice}")

        # Otherwise try parse int-like string
        choice_str = str(choice).strip()
        if choice_str.isdigit():
            idx = int(choice_str)
            if 0 <= idx < len(sheet_names):
                return idx
            raise ValueError(f"Sheet index out of range: {idx}")

        # Fallback: treat as name
        if choice_str in sheet_names:
            return choice_str
        raise ValueError(
            f"Unknown sheet '{choice_str}'. Available: {sheet_names}"
        )

 # ---------- Loading helpers ----------

    @staticmethod
    def load_table(
        file_path: Union[str, Path],
        sheet: Optional[Union[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Load a table from CSV/TSV/Excel.
        - CSV/TSV: 'sheet' is ignored.
        - Excel: if 'sheet' is None → loads first sheet (index 0).
                 if 'sheet' is int or str → passed directly to pandas.
        """
        path = Path(file_path)
        suf = path.suffix.lower()

        if suf in (".csv", ".tsv"):
            sep = "," if suf == ".csv" else "\t"
            return pd.read_csv(path, sep=sep)

        if suf in (".xlsx", ".xls"):
            sheet_to_use = 0 if sheet is None else sheet
            return pd.read_excel(path, sheet_name=sheet_to_use)

        raise ValueError(f"Unsupported file type: {suf}")

    @staticmethod
    def pick_sheet_interactive(
        file_path: Union[str, Path],
        prompt: str = "Enter sheet index or name: ",
        default: Optional[Union[int, str]] = None,
    ) -> Union[int, str]:
        """
        Print sheet list, prompt the user for an index or name, and return it.
        If 'default' is provided and user presses Enter, returns the default.
        """
        sheet_names = ExcelUtils.list_sheets(file_path)
        print("Available sheets:")
        for i, name in enumerate(sheet_names):
            print(f"  {i}: {name}")

        if default is not None:
            print(f"(Press Enter for default: {default})")

        choice = input(prompt).strip()
        if choice == "" and default is not None:
            return ExcelUtils.resolve_sheet(default, sheet_names)
        return ExcelUtils.resolve_sheet(choice, sheet_names)

    @staticmethod
    def load_with_prompt(
        file_path: Union[str, Path],
        default: Optional[Union[int, str]] = 0,
    ) -> pd.DataFrame:
        """
        For Excel files: interactive sheet selection, then load that sheet.
        For CSV/TSV: loads file directly (no prompt).
        """
        path = Path(file_path)
        if path.suffix.lower() in (".xlsx", ".xls"):
            chosen = ExcelUtils.pick_sheet_interactive(file_path, default=default)
            print(f"\nLoading sheet: {chosen}")
            return ExcelUtils.load_table(file_path, sheet=chosen)
        # Non-Excel: just load
        return ExcelUtils.load_table(file_path)