from pathlib import Path
from utils.ChangeTracker import ChangeTracker
import ClassChangeModel
import pandas as pd

def train_change_model(folder_path: str):
    
    model = ClassChangeModel()
    tracker = ChangeTracker()
    
    
    tsv_files = sorted(Path(folder_path).glob("*.tsv"))
    for tsv in tsv_files:
        print(f"Processing {tsv.name}...")
        df = pd.read_csv(tsv, sep="\t", skiprows=26)
        tracker.add_release(tsv.stem, df)
    
    
    training_data = tracker.get_training_data()
    
    
    results = model.train(training_data)
    
    # gotta choose other shit not feature importance
    print("\nModel Training Results:")
    print("----------------------")
    print("\nFeature Importance:")
    print(results['feature_importance'].head(10))
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    return model

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Train dependency change prediction model")
    parser.add_argument("folder", help="Folder containing TSV dependency files")
    args = parser.parse_args()
    
    trained_model = train_change_model(args.folder)