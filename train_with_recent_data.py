#!/usr/bin/env python3
"""Train on more recent data to avoid temporal domain shift"""

import pickle
import pandas as pd
from pathlib import Path
from next_actual_change_tracker import NextActualChangeTracker
from src.ml.ClassChangeModel import ClassChangeModel

def train_with_recent_data():
    """Train using more recent JStock versions"""
    print("=== TRAINING WITH RECENT DATA (avoiding temporal gap) ===\n")
    
    # Use files 3-12 for training (overlapping with test data area)
    jstock_path = "data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock"
    jstock_files = sorted(Path(jstock_path).glob('*.tsv'))
    
    # More recent training data
    training_files = jstock_files[3:13]  # Skip earliest files, use more recent ones
    
    print(f"Training on files 3-12: {[f.name for f in training_files]}")
    
    tracker = NextActualChangeTracker()
    for file_path in training_files:
        df = pd.read_csv(file_path, sep='\t', skiprows=26)
        tracker.add_release(file_path.stem, df)
    
    training_data = tracker.get_training_data_next_actual()
    df_training = pd.DataFrame(training_data)
    df_training['Incoming Change'] = df_training['Next Actual Change']
    df_training = df_training.drop('Next Actual Change', axis=1)
    
    print(f"Generated {len(df_training)} training samples")
    
    # Train model
    model = ClassChangeModel()
    training_results = model.train(df_training)
    
    print(f"\n✓ Model trained successfully")
    print("\nClassification Report:")
    print(training_results['classification_report'])
    
    # Save model
    model_filename = 'recent_next_actual_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved as '{model_filename}'")
    
    # Now test on files 13-17 (completely unseen but recent)
    test_files = jstock_files[13:18] if len(jstock_files) > 17 else jstock_files[13:]
    
    if len(test_files) > 0:
        print(f"\n=== TESTING ON RECENT UNSEEN DATA ===")
        print(f"Test files: {[f.name for f in test_files]}")
        
        test_tracker = NextActualChangeTracker()
        for file_path in test_files:
            df = pd.read_csv(file_path, sep='\t', skiprows=26)
            test_tracker.add_release(file_path.stem, df)
        
        test_data = test_tracker.get_training_data_next_actual()
        test_df = pd.DataFrame(test_data)
        test_df['Incoming Change'] = test_df['Next Actual Change']
        test_df = test_df.drop('Next Actual Change', axis=1)
        
        if len(test_df) > 0:
            print(f"Generated {len(test_df)} test samples")
            
            # Test predictions
            feature_columns = [col for col in test_df.columns 
                              if col not in ['Release', 'Class', 'Incoming Change', 'Releases Until Change']]
            
            X_test = test_df[feature_columns]
            y_actual = test_df['Incoming Change'].apply(model.change_buckets)
            
            prediction_results = model.predict_changes(X_test)
            y_pred = prediction_results['Predicted_Change'].values
            
            accuracy = (y_pred == y_actual).mean()
            print(f"Test accuracy: {accuracy:.1%}")
            
            # Large change analysis
            large_actual = [i for i, label in enumerate(y_actual) if 'Large' in label]
            large_pred = [i for i, label in enumerate(y_pred) if 'Large' in label]
            true_positives = [i for i in large_actual if 'Large' in y_pred[i]]
            
            print(f"Large changes - Actual: {len(large_actual)}, Predicted: {len(large_pred)}, Correct: {len(true_positives)}")
            
            if large_actual:
                recall = len(true_positives) / len(large_actual)
                print(f"Large change recall: {recall:.1%}")
        else:
            print("No test data generated from recent files")
    else:
        print("No additional files available for testing")
    
    return model

if __name__ == "__main__":
    model = train_with_recent_data()