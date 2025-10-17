#!/usr/bin/env python3
"""Train on more recent data to avoid temporal domain shift"""

import pickle
import pandas as pd
from pathlib import Path
from next_actual_change_tracker import NextActualChangeTracker
from src.ml.ClassChangeModel import ClassChangeModel
from sklearn.metrics import classification_report

def train_with_recent_data():
    print("=== TRAINING WITH RECENT DATA ===\n")
    
    # hardcoded path mb its easier for now
    data_path = "data/jgraph-jmeter-jstock-jung-lucene-weka/lucene_deps/lucene"
    data_files = sorted(Path(data_path).glob('*.tsv'))
    
    
    training_files = data_files[11:21] 
    
    print(f"Training on files: {len(training_files)}")
    for f in training_files:
        print(f"  - {f.name}")
    
    tracker = NextActualChangeTracker()
    for file_path in training_files:
        df = pd.read_csv(file_path, sep='\t', skiprows=26)
        tracker.add_release(file_path.stem, df)
        print(f"  Loaded for training: {file_path.name}")
    
    training_data = tracker.get_training_data_next_actual()
    df_training = pd.DataFrame(training_data)
    df_training['Incoming Change'] = df_training['Next Actual Change']
    df_training = df_training.drop('Next Actual Change', axis=1)
    
    print(f"\nGenerated {len(df_training)} training samples")
    
    # model training
    model = ClassChangeModel()
    training_results = model.train(df_training)
    
    print(f"\nModel trained successfully")
    print("\nClassification Report:")
    print(training_results['classification_report'])
    
    model_filename = 'recent_next_actual_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n\u2713 Model saved as '{model_filename}'")
    
    # testing
    test_files = data_files[21:36] if len(data_files) > 17 else data_files[13:]
    
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
            print(f"\n=== TEST RESULTS ===")
            print(f"Accuracy: {accuracy:.3f} ({(y_pred == y_actual).sum()}/{len(y_actual)})")
            print(f"Test accuracy: {accuracy:.1%}")
            
            # Large change analysis
            large_actual = [i for i, label in enumerate(y_actual) if 'Large' in label]
            large_pred = [i for i, label in enumerate(y_pred) if 'Large' in label]
            true_positives = [i for i in large_actual if 'Large' in y_pred[i]]
            
            print(f"Large changes - Actual: {len(large_actual)}, Predicted: {len(large_pred)}, Correct: {len(true_positives)}")
            
            if large_actual:
                recall = len(true_positives) / len(large_actual)
                print(f"Large change recall: {recall:.1%}")
            
            # Build results frame for more detailed reporting (mirrors simple_test.py)
            results = []
            try:
                for class_name in prediction_results.index:
                    if class_name in test_df.index:
                        pred_change = prediction_results.loc[class_name, 'Predicted_Change']
                        confidence = prediction_results.loc[class_name, 'Confidence']
                        actual_change = test_df.loc[class_name, 'Incoming Change']
                        actual_category = model.change_buckets(actual_change)
                        results.append({
                            'Class': class_name,
                            'Predicted': pred_change,
                            'Actual': actual_category,
                            'Actual_Change': actual_change,
                            'Confidence': confidence,
                            'Correct': pred_change == actual_category
                        })
            except Exception:
                pass

            import pandas as _pd
            if len(results) > 0:
                results_df = _pd.DataFrame(results)
                print("\nClassification Report:")
                try:
                    print(classification_report(results_df['Actual'].tolist(), results_df['Predicted'].tolist(), zero_division=0))
                except Exception:
                    pass

                # Correctly predicted large changes
                large_changes = results_df[(results_df['Correct'] == True) & (results_df['Predicted'].isin(['Large Decrease', 'Large Increase']))]
                if len(large_changes) > 0:
                    print("--- CORRECTLY PREDICTED LARGE DECREASES/INCREASES ---")
                    large_sorted = large_changes.sort_values('Confidence', ascending=False)
                    for i, (_, result) in enumerate(large_sorted.iterrows()):
                        print(f"{str(result['Class'])[:60]:<60} | {result['Predicted']:<15} | Conf: {result['Confidence']:.3f}")
                else:
                    print("--- NO LARGE CHANGES CORRECTLY PREDICTED ---")
                    # Show actual large changes if present
                    actual_large = results_df[results_df['Actual'].isin(['Large Decrease', 'Large Increase'])]
                    if len(actual_large) > 0:
                        print(f"\nActual Large Changes (Total: {len(actual_large)}):")
                        for i, (_, result) in enumerate(actual_large.head(10).iterrows()):
                            print(f"{str(result['Class'])[:60]:<60} | Actual: {result['Actual']:<15} | Predicted: {result['Predicted']:<15}")
                # Accuracy by predicted category
                print(f"\nAccuracy by Predicted Category:")
                for category in sorted(results_df['Predicted'].unique()):
                    cat_results = results_df[results_df['Predicted'] == category]
                    cat_accuracy = cat_results['Correct'].mean()
                    print(f"  {category}: {cat_accuracy:.3f} ({cat_results['Correct'].sum()}/{len(cat_results)})")
                # Actual distribution
                print(f"\nActual Category Distribution:")
                actual_dist = results_df['Actual'].value_counts().sort_index()
                for category, count in actual_dist.items():
                    percentage = count / len(results_df) * 100
                    print(f"  {category}: {count} ({percentage:.1f}%)")
            else:
                print("No detailed per-class results available for reporting.")
        else:
            print("No test data generated from recent files")
    else:
        print("No additional files available for testing")
    
    return model

if __name__ == "__main__":
    model = train_with_recent_data()