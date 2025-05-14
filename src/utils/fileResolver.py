import sys
from pathlib import Path
from utils.exceptions import TSVFileNotFoundError, TSVMultipleFilesFoundError

class FileResolver():
    def __init__(self, filename: str, project_root: Path):
        self.filename = filename
        self.project_root = project_root
         #search for file in project root if not absolute path
        self.matched_files = list(project_root.rglob(filename)) if not Path(filename).is_absolute() else [] 
        self.is_direct_path = Path(filename).exists()
        self.separator = lambda: print("-" * 80)

    def resolve(self) -> Path: 
        if self.is_direct_path: 
            return Path(self.filename).resolve()

        if len(self.matched_files) == 1:
            return self.matched_files[0]

        if len(self.matched_files) > 1:
            self._handle_multiple_matches()
            
        self._handle_file_not_found()
    
    # This function is called when multiple TSV files are found 
    def _handle_multiple_matches(self):
        print(f"\nMultiple TSV files found for '{self.filename}'")
        self.separator()
        
        #Prevent duplicate suggestions
        shown_suggestions = set()
        for match in self.matched_files:
            rel_path = match.relative_to(self.project_root)
            suggestion = Path(match.parent.name) / match.name

            if suggestion in shown_suggestions:
                continue

            print(f"- {rel_path} \nTry: python src/TSVReader.py {suggestion}")
            print("OR")
            print(f"Try: tsvreader {suggestion}")
            self.separator()
            shown_suggestions.add(suggestion)

        raise TSVMultipleFilesFoundError(f"Multiple TSV files found for '{self.filename}'")

    def _handle_file_not_found(self):
        raise TSVFileNotFoundError(f"File '{self.filename}' not found")