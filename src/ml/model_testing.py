import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Now import from the modules
from src.ml.ClassChangeModel import ClassChangeModel
from src.utils.ChangeTracker import ChangeTracker
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def test_model_predictions(model_path, test_dataset_path, exclude_last_n=1):
    """
    Test a trained model by predicting the last release(s) of a dataset
    """
    print(f"Loading model from: {model_path}")
    model = ClassChangeModel()
    model.load_model(model_path)
    
    print(f"Loading test data from: {test_dataset_path}")
    path = Path(test_dataset_path)
    tsv_files = sorted(path.glob('*.tsv'))
    
    if len(tsv_files) < 2:
        print("Need at least 2 releases for testing!")
        return None, 0
    
    # Split files: use all but last N for training context, last N for testing
    context_files = tsv_files[:-exclude_last_n]
    test_files = tsv_files[-exclude_last_n:]
    
    print(f"Using {len(context_files)} releases for context")
    print(f"Testing predictions on {len(test_files)} releases")
    
    # Build context (second-to-last release data)
    tracker = ChangeTracker()
    for f in context_files:
        try:
            df = pd.read_csv(f, sep='\t', skiprows=26)
            tracker.add_release(f.stem, df)
            print(f"  Context: {f.name}")
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
            continue
    
    # Get the second-to-last release data for prediction
    if len(tracker.ordered_releases) == 0:
        print("No valid context data!")
        return None, 0
    
    second_last_release = tracker.ordered_releases[-1]
    prediction_data = tracker.releases[second_last_release]
    
    print(f"\nMaking predictions for release: {second_last_release}")
    print(f"Classes to predict: {len(prediction_data)}")
    
    # Make predictions
    try:
        predictions = model.predict_changes(prediction_data)
        print(f"Predictions made for {len(predictions)} classes")
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, 0
    
    # Load actual next release to compare
    actual_tracker = ChangeTracker()
    for f in tsv_files[:-1]:  # All files except the very last
        try:
            df = pd.read_csv(f, sep='\t', skiprows=26)
            actual_tracker.add_release(f.stem, df)
        except Exception as e:
            print(f"Error loading {f.name} for actual data: {e}")
            continue
    
    # Add the final release to get actual changes
    final_release_file = test_files[0]
    try:
        df_final = pd.read_csv(final_release_file, sep='\t', skiprows=26)
        actual_tracker.add_release(final_release_file.stem, df_final)
    except Exception as e:
        print(f"Error loading final release {final_release_file.name}: {e}")
        return None, 0
    
    # Get actual changes
    if second_last_release not in actual_tracker.releases:
        print(f"Release {second_last_release} not found in actual tracker!")
        print(f"Available releases: {actual_tracker.ordered_releases}")
        return None, 0
    
    actual_data = actual_tracker.releases[second_last_release]
    
    # Compare predictions vs actual
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    
    correct_predictions = 0
    total_predictions = 0
    comparison_results = []
    
    print(f"Prediction data shape: {predictions.shape}")
    print(f"Actual data shape: {actual_data.shape}")
    print(f"Prediction columns: {list(predictions.columns)}")
    print(f"Actual data columns: {list(actual_data.columns)}")
    
    for class_name in predictions.index:
        if class_name in actual_data.index:
            try:
                predicted_category = predictions.loc[class_name, 'Predicted Change Category']
                actual_change = actual_data.loc[class_name, 'Incoming Change']
                actual_category = model.change_buckets(actual_change)
                
                is_correct = predicted_category == actual_category
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                confidence = predictions.loc[class_name, f'Probability_{predicted_category}']
                
                comparison_results.append({
                    'Class': class_name,
                    'Predicted': predicted_category,
                    'Actual': actual_category,
                    'Actual_Change': actual_change,
                    'Correct': is_correct,
                    'Confidence': confidence
                })
            except Exception as e:
                print(f"Error processing class {class_name}: {e}")
                continue
    
    if not comparison_results:
        print("No valid comparisons could be made!")
        return None, 0
    
    # Create results DataFrame
    results_df = pd.DataFrame(comparison_results)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    # Show detailed results only if we have data
    if len(results_df) > 0 and 'Actual' in results_df.columns:
        print(f"\nDetailed Classification Report:")
        y_true = results_df['Actual'].tolist()
        y_pred = results_df['Predicted'].tolist()
        
        try:
            print(classification_report(y_true, y_pred))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        
        # Show some example predictions
        print(f"\nExample Predictions (showing first 10):")
        print("-" * 80)
        for _, row in results_df.head(10).iterrows():
            status = "✓" if row['Correct'] else "✗"
            print(f"{status} {row['Class'][:40]:<40} | Pred: {row['Predicted']:<15} | Actual: {row['Actual']:<15} | Conf: {row['Confidence']:.3f}")
        
        # Show most confident correct predictions
        if len(results_df[results_df['Correct'] == True]) > 0:
            print(f"\nMost Confident CORRECT Predictions:")
            print("-" * 80)
            correct_preds = results_df[results_df['Correct'] == True].nlargest(5, 'Confidence')
            for _, row in correct_preds.iterrows():
                print(f"✓ {row['Class'][:40]:<40} | {row['Predicted']:<15} | Conf: {row['Confidence']:.3f}")
        
        # Show most confident wrong predictions  
        if len(results_df[results_df['Correct'] == False]) > 0:
            print(f"\nMost Confident WRONG Predictions:")
            print("-" * 80)
            wrong_preds = results_df[results_df['Correct'] == False].nlargest(5, 'Confidence')
            for _, row in wrong_preds.iterrows():
                print(f"✗ {row['Class'][:40]:<40} | Pred: {row['Predicted']:<12} | Act: {row['Actual']:<12} | Conf: {row['Confidence']:.3f}")
    
    return results_df, accuracy

def test_multiple_datasets():
    """Test the model on multiple datasets"""
    
    # Test datasets (adjust paths as needed)
    test_cases = [
        {
            'name': 'JUnit',
            'path': 'data/junit-depfiles/junit-depfiles',
            'model': 'model_trained_on_4_datasets.pkl'
        },
        {
            'name': 'JStock',
            'path': 'data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock',
            'model': 'model_trained_on_4_datasets.pkl'
        }
    ]
    
    overall_results = []
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"TESTING ON {test_case['name'].upper()} DATASET")
        print(f"{'='*80}")
        
        if not Path(test_case['model']).exists():
            print(f"Model not found: {test_case['model']}")
            continue
            
        if not Path(test_case['path']).exists():
            print(f"Dataset not found: {test_case['path']}")
            continue
        
        results_df, accuracy = test_model_predictions(
            test_case['model'], 
            test_case['path']
        )
        
        if results_df is not None and len(results_df) > 0:
            overall_results.append({
                'Dataset': test_case['name'],
                'Accuracy': accuracy,
                'Total_Predictions': len(results_df)
            })
        else:
            print(f"No valid results for {test_case['name']}")
    
    # Summary
    print(f"\n{'='*80}")
    print("OVERALL TESTING SUMMARY")
    print(f"{'='*80}")
    
    if overall_results:
        summary_df = pd.DataFrame(overall_results)
        print(summary_df.to_string(index=False))
        
        avg_accuracy = summary_df['Accuracy'].mean()
        print(f"\nAverage Accuracy Across All Datasets: {avg_accuracy:.3f}")
    else:
        print("No valid test results to summarize.")

if __name__ == "__main__":
    # Test on multiple datasets
    test_multiple_datasets()