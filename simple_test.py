import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.ml.ClassChangeModel import ClassChangeModel
from src.utils.ChangeTracker import ChangeTracker
import pandas as pd
from sklearn.metrics import classification_report

def simple_prediction_test():
    """Simple test on JStock data"""
    
    print("=== JSTOCK PREDICTION TEST ===")
    
    # Load model
    model = ClassChangeModel()
    model.load_model("eventual_change_model.pkl")
    
    # Load JStock data instead of JUnit
    path = Path('data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock')
    tsv_files = sorted(path.glob('*.tsv'))[:20]  # Use first 8 files for more data
    
    if len(tsv_files) < 2:
        print("Need at least 2 files!")
        return
    
    print(f"Found {len(tsv_files)} JStock files")
    
    # Use first 18 files for context, predict on 19th, compare with 20th
    context_files = tsv_files[:18]
    predict_file = tsv_files[18]
    
    print(f"Context files: {len(context_files)} files from {context_files[0].name} to {context_files[-1].name}")
    print(f"Predicting on: {predict_file.name}")
    
    # Build tracker with context + prediction file
    tracker = ChangeTracker()
    for f in context_files + [predict_file]:
        df = pd.read_csv(f, sep='\t', skiprows=26)
        tracker.add_release(f.stem, df)
        print(f"  Loaded: {f.name}")
    
    # Get data from the prediction release
    predict_release = predict_file.stem
    prediction_data = tracker.releases[predict_release]
    
    print(f"\nMaking predictions for {len(prediction_data)} classes")
    
    # Make predictions
    predictions = model.predict_changes(prediction_data)
    
    # Now add the 20th file to get actual changes
    if len(tsv_files) > 19:
        final_file = tsv_files[19]
        df_final = pd.read_csv(final_file, sep='\t', skiprows=26)
        tracker.add_release(final_file.stem, df_final)
        
        # Get actual changes that happened FROM predict_release TO final_release
        actual_data = tracker.releases[predict_release]  # This now has "Incoming Change"
        
        print(f"Comparing with actual changes to: {final_file.name}")
        
        # Compare predictions
        correct = 0
        total = 0
        results = []
        
        for class_name in predictions.index:
            if class_name in actual_data.index:
                pred_change = predictions.loc[class_name, 'Predicted_Change']
                actual_change = actual_data.loc[class_name, 'Incoming Change']
                actual_category = model.change_buckets(actual_change)
                
                is_correct = pred_change == actual_category
                confidence = predictions.loc[class_name, 'Confidence']
                
                results.append({
                    'Class': class_name,
                    'Predicted': pred_change,
                    'Actual': actual_category,
                    'Actual_Change': actual_change,
                    'Correct': is_correct,
                    'Confidence': confidence
                })
                
                if is_correct:
                    correct += 1
                total += 1
        
        if total > 0:
            accuracy = correct / total
            print(f"\n=== RESULTS ===")
            print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
            
            # Show some examples
            results_df = pd.DataFrame(results)
            
            print(f"\nClassification Report:")
            y_true = results_df['Actual'].tolist()
            y_pred = results_df['Predicted'].tolist()
            print(classification_report(y_true, y_pred))
            
            # Show correctly predicted large changes
            print(f"\nCorrectly Predicted Large Changes:")
            
            # Filter for correct predictions of Large Decrease or Large Increase
            large_changes = results_df[
                (results_df['Correct'] == True) & 
                (results_df['Predicted'].isin(['Large Decrease', 'Large Increase']))
            ]
            
            if len(large_changes) > 0:
                print("--- CORRECTLY PREDICTED LARGE DECREASES/INCREASES ---")
                large_sorted = large_changes.sort_values('Confidence', ascending=False)
                for i, (_, result) in enumerate(large_sorted.iterrows()):
                    print(f"{result['Class'][:60]:<60} | {result['Predicted']:<15} | Conf: {result['Confidence']:.3f}")
            else:
                print("--- NO LARGE CHANGES CORRECTLY PREDICTED ---")
                print("The model did not correctly predict any Large Decrease or Large Increase cases.")
                
                # Show what large changes actually occurred
                actual_large = results_df[results_df['Actual'].isin(['Large Decrease', 'Large Increase'])]
                if len(actual_large) > 0:
                    print(f"\nActual Large Changes (Total: {len(actual_large)}):")
                    for i, (_, result) in enumerate(actual_large.head(10).iterrows()):
                        print(f"{result['Class'][:60]:<60} | Actual: {result['Actual']:<15} | Predicted: {result['Predicted']:<15}")
            
            # Show any correctly predicted smaller changes for context
            small_changes = results_df[
                (results_df['Correct'] == True) & 
                (results_df['Predicted'].isin(['Small Decrease', 'Small Increase']))
            ]
            
            if len(small_changes) > 0:
                print(f"\n--- CORRECTLY PREDICTED SMALL CHANGES (Sample) ---")
                small_sorted = small_changes.sort_values('Confidence', ascending=False)
                for i, (_, result) in enumerate(small_sorted.head(5).iterrows()):
                    print(f"{result['Class'][:60]:<60} | {result['Predicted']:<15} | Conf: {result['Confidence']:.3f}")
            
            # Show accuracy by category
            print(f"\nAccuracy by Predicted Category:")
            for category in sorted(results_df['Predicted'].unique()):
                cat_results = results_df[results_df['Predicted'] == category]
                cat_accuracy = cat_results['Correct'].mean()
                print(f"  {category}: {cat_accuracy:.3f} ({cat_results['Correct'].sum()}/{len(cat_results)})")
            
            # Show actual category distribution
            print(f"\nActual Category Distribution:")
            actual_dist = results_df['Actual'].value_counts().sort_index()
            for category, count in actual_dist.items():
                percentage = count / len(results_df) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
                
        else:
            print("No matching classes found!")
    else:
        print("Not enough files for testing!")

if __name__ == "__main__":
    simple_prediction_test()