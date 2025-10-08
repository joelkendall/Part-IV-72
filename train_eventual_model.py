#!/usr/bin/env python3
"""
Train a model using eventual changes instead of immediate changes
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from eventual_change_tracker import EventualChangeTracker
from src.ml.ClassChangeModel import ClassChangeModel
import pandas as pd

class EventualChangeModel(ClassChangeModel):
    """Model that predicts eventual changes rather than immediate ones"""
    
    def process_data(self, df):
        """Override to use Eventual Change instead of Incoming Change"""
        df_clean = df.copy()
        
        data_columns = [col for col in df_clean.columns if col not in ['Class', 'Release', 'Eventual Change', 'Change Release']]
        category_cols = [col for col in data_columns if col not in ['Total Dependencies', 'Class', 'Release']]
        
        for col in category_cols:
            df_clean[col] = df_clean[col].fillna(0)
        
        X = df_clean[data_columns]
        y = df_clean['Eventual Change'].apply(self.change_buckets)  # Use eventual change!

        self.data_columns = data_columns
        self.cat_cols = category_cols

        print(f"Data shape after cleaning: {X.shape}")
        print(f"Target distribution: {y.value_counts()}")
        print(f"Features with NaN: {X.isnull().sum().sum()}")

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale only Total Dependencies column
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled['Total Dependencies'] = self.scaler.fit_transform(X_train[['Total Dependencies']])
        X_test_scaled['Total Dependencies'] = self.scaler.transform(X_test[['Total Dependencies']])
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def train_eventual_change_model():
    """Train model using eventual changes approach"""
    
    print("=== TRAINING EVENTUAL CHANGE MODEL ===")
    
    # Load multiple datasets with eventual change tracking
    all_training_data = []
    
    datasets = {
        "junit": {"path": "data/junit-depfiles/junit-depfiles", "files": 8},
        "jstock": {"path": "data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock", "files": 8}
    }
    
    for dataset_name, config in datasets.items():
        print(f"\nLoading {dataset_name} with eventual change tracking...")
        
        # Create tracker for this dataset
        tracker = EventualChangeTracker(look_ahead_releases=3)
        
        # Load files
        files = sorted(Path(config["path"]).glob('*.tsv'))[:config["files"]]
        
        for f in files:
            df = pd.read_csv(f, sep='\t', skiprows=26)
            tracker.add_release(f.stem, df)
            print(f"  ✓ {f.name}")
        
        # Get eventual change training data
        training_data = tracker.get_training_data_eventual()
        if len(training_data) > 0:
            training_data['Dataset'] = dataset_name
            all_training_data.append(training_data)
            print(f"  → {dataset_name}: {len(training_data)} eventual change samples")
        else:
            print(f"  ✗ No eventual change data for {dataset_name}")
    
    if not all_training_data:
        print("ERROR: No training data available!")
        return None
    
    # Combine all datasets
    combined_data = pd.concat(all_training_data, ignore_index=True)
    print(f"\nTotal combined eventual change training data: {len(combined_data)} samples")
    
    # Show dataset distribution
    dataset_counts = combined_data['Dataset'].value_counts()
    for dataset, count in dataset_counts.items():
        percentage = count / len(combined_data) * 100
        print(f"  {dataset}: {count} ({percentage:.1f}%)")
    
    # Show eventual change distribution
    model_temp = EventualChangeModel()
    bucket_distribution = combined_data['Eventual Change'].apply(model_temp.change_buckets).value_counts()
    print(f"\nEventual change distribution:")
    for category, count in bucket_distribution.items():
        percentage = count / len(combined_data) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Remove dataset column before training
    training_features = combined_data.drop('Dataset', axis=1)
    
    # Train the eventual change model
    print(f"\n=== TRAINING EVENTUAL CHANGE MODEL ===")
    model = EventualChangeModel()
    results = model.train(training_features)
    
    # Save the model
    model.save_model("eventual_change_model.pkl")
    print(f"\nEventual change model saved as 'eventual_change_model.pkl'")
    
    return model

def compare_model_approaches():
    """Compare eventual vs immediate change models"""
    
    print("\n" + "="*60)
    print("COMPARING EVENTUAL vs IMMEDIATE CHANGE MODELS")
    print("="*60)
    
    try:
        # Load both models
        immediate_model = ClassChangeModel()
        immediate_model.load_model("model_trained_on_4_datasets.pkl")
        print("✓ Loaded immediate change model")
        
        eventual_model = EventualChangeModel()
        eventual_model.load_model("eventual_change_model.pkl")
        print("✓ Loaded eventual change model")
        
        # Test on same data
        test_files = sorted(Path('data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock').glob('*.tsv'))[-3:]
        
        # Create test data
        tracker = EventualChangeTracker(look_ahead_releases=2)
        for f in test_files:
            df = pd.read_csv(f, sep='\t', skiprows=26)
            tracker.add_release(f.stem, df)
        
        test_data = tracker.releases[tracker.ordered_releases[0]]  # Use first release for prediction
        
        print(f"\nTesting on {len(test_data)} classes...")
        
        # Get predictions from both models
        immediate_preds = immediate_model.predict_changes(test_data)
        eventual_preds = eventual_model.predict_changes(test_data)
        
        print("\n--- IMMEDIATE MODEL PREDICTIONS ---")
        print(immediate_preds['Predicted_Change'].value_counts())
        
        print("\n--- EVENTUAL MODEL PREDICTIONS ---")
        print(eventual_preds['Predicted_Change'].value_counts())
        
        # Show some examples where they differ
        merged = immediate_preds.merge(eventual_preds, left_index=True, right_index=True, suffixes=('_imm', '_evt'))
        different = merged[merged['Predicted_Change_imm'] != merged['Predicted_Change_evt']]
        
        print(f"\nPredictions that differ: {len(different)}")
        if len(different) > 0:
            print("\nTop 10 differences:")
            for i, (class_name, row) in enumerate(different.head(10).iterrows()):
                print(f"  {class_name[:50]:<50} | Immediate: {row['Predicted_Change_imm']:<15} | Eventual: {row['Predicted_Change_evt']:<15}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")

if __name__ == "__main__":
    # Train the eventual change model
    model = train_eventual_change_model()
    
    if model:
        # Compare approaches
        compare_model_approaches()