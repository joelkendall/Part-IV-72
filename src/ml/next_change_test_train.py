import pickle
import pandas as pd
from pathlib import Path
from next_actual_change_tracker import NextActualChangeTracker
from ClassChangeModel import ClassChangeModel
from sklearn.metrics import classification_report

def next_change_test_train(
    data_path="data/jgraph-jmeter-jstock-jung-lucene-weka/weka_deps/weka",
    train_start=15,
    train_end=20,
    test_start=20,
    test_end=25,
    model_filename='recent_next_actual_model.pkl'
):
    print("Training with recent data, next change approach ...")
    
    # Load data files
    data_files = sorted(Path(data_path).glob('*.tsv'))
    
    if len(data_files) < test_end:
        print(f"Warning: Only {len(data_files)} files available, adjusting ranges...")
        test_end = len(data_files)
        if test_start >= test_end:
            test_start = max(0, test_end - 10)
        if train_end > test_start:
            train_end = test_start
        if train_start >= train_end:
            train_start = max(0, train_end - 10)
    
    training_files = data_files[train_start:train_end]
    test_files = data_files[test_start:test_end]
    
    print(f"\nDataset: {Path(data_path).name}")
    print(f"Training on {len(training_files)} files (indices {train_start}-{train_end-1}):")
    for f in training_files:
        print(f"  - {f.name}")
    
    # Build training tracker
    tracker = NextActualChangeTracker()
    print(f"\nLoading training data...")
    for file_path in training_files:
        df = pd.read_csv(file_path, sep='\t', skiprows=26)
        tracker.add_release(file_path.stem, df)
        print(f"  Loaded: {file_path.name}")
    
    training_data = tracker.get_training_data_next_actual()
    df_training = pd.DataFrame(training_data)
    df_training['Incoming Change'] = df_training['Next Actual Change']
    df_training = df_training.drop('Next Actual Change', axis=1)
    
    if 'Releases Until Change' in df_training.columns:
        df_training = df_training.drop('Releases Until Change', axis=1)
    
    print(f"\nGenerated {len(df_training)} training samples")
    
    #model training
    print(f"\nTraining model...")
    model = ClassChangeModel()
    training_results = model.train(df_training)
    
    print(f"\nModel trained successfully")
    print("\nTraining Classification Report:")
    print(training_results['classification_report'])
    
    # Save model
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved as '{model_filename}'")
    
    #testing
    if len(test_files) == 0:
        print("\nNo test files available!")
        return model
    
    print(f"Testing")
    print(f"\nTest files: {len(test_files)} files (indices {test_start}-{test_end-1}):")
    for f in test_files:
        print(f"  - {f.name}")
    
    test_tracker = NextActualChangeTracker()
    print(f"\nLoading test data...")
    for file_path in test_files:
        df = pd.read_csv(file_path, sep='\t', skiprows=26)
        test_tracker.add_release(file_path.stem, df)
        print(f"  Loaded: {file_path.name}")
    
    first_test_release = test_files[0].stem
    first_release_data = test_tracker.releases[first_test_release]
    
    print(f"\nMaking predictions from first test release: {first_test_release}")
    print(f"Predicting for {len(first_release_data)} classes")
    
    feature_columns = [col for col in first_release_data.columns 
                      if col not in ['Next Actual Change', 'Releases Until Change']]
    
    X_predict = first_release_data[feature_columns]
    
    # Making predictions
    predictions = model.predict_changes(X_predict)
    
    # comparing
    if 'Next Actual Change' not in first_release_data.columns:
        print("\nError: Next Actual Change not computed for first test release!")
        return model
    
    results = []
    for class_name in predictions.index:
        if class_name not in first_release_data.index:
            continue
        
        pred_change = predictions.loc[class_name, 'Predicted_Change']
        confidence = predictions.loc[class_name, 'Confidence']
        
        actual_change_value = first_release_data.loc[class_name, 'Next Actual Change']
        actual_category = model.change_buckets(actual_change_value)
        
        releases_until = first_release_data.loc[class_name, 'Releases Until Change'] if 'Releases Until Change' in first_release_data.columns else None
        
        results.append({
            'Class': class_name,
            'Predicted': pred_change,
            'Actual': actual_category,
            'Actual_Change_Value': actual_change_value,
            'Confidence': confidence,
            'Releases_Until_Change': releases_until,
            'Correct': pred_change == actual_category
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\nNo results to evaluate!")
        return model
    
    # accuracy
    correct_predictions = results_df['Correct'].sum()
    total_predictions = len(results_df)
    accuracy = correct_predictions / total_predictions
    
    print(f"- TEST RESULTS -")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        results_df['Actual'].tolist(), 
        results_df['Predicted'].tolist(), 
        zero_division=0
    ))
    
    # large change analysis
    large_actual = results_df[results_df['Actual'].isin(['Large Decrease', 'Large Increase'])]
    large_correct = large_actual[large_actual['Correct'] == True]
    
    print(f"\nLarge Change Analysis:")
    print(f"  Actual large changes: {len(large_actual)}")
    print(f"  Correctly predicted: {len(large_correct)}")
    if len(large_actual) > 0:
        large_recall = len(large_correct) / len(large_actual)
        print(f"  Large change recall: {large_recall:.3f} ({large_recall*100:.1f}%)")
    
    if len(large_correct) > 0:
        print(f"\nCorrectly Predicted Large Changes:")
        large_sorted = large_correct.sort_values('Confidence', ascending=False)
        for _, result in large_sorted.iterrows():
            class_str = str(result['Class'])[:50]
            releases = f"after {int(result['Releases_Until_Change'])} releases" if result['Releases_Until_Change'] is not None else ""
            print(f"{class_str:<50} | {result['Predicted']:<15} | Conf: {result['Confidence']:.3f} {releases}")
    else:
        print(f"\nNo Large Changes Correctly Predicted")
        if len(large_actual) > 0:
            print(f"\nActual Large Changes That Were Missed (showing first 10):")
            for _, result in large_actual.head(10).iterrows():
                class_str = str(result['Class'])[:50]
                print(f"{class_str:<50} | Actual: {result['Actual']:<15} | Predicted: {result['Predicted']:<15}")
    
    print(f"\nAccuracy by Predicted Category:")
    for category in sorted(results_df['Predicted'].unique()):
        cat_results = results_df[results_df['Predicted'] == category]
        cat_correct = cat_results['Correct'].sum()
        cat_total = len(cat_results)
        cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
        print(f"  {category}: {cat_accuracy:.3f} ({cat_correct}/{cat_total})")
    
    print(f"\nActual Category Distribution:")
    actual_dist = results_df['Actual'].value_counts().sort_index()
    for category, count in actual_dist.items():
        percentage = count / len(results_df) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    return model

if __name__ == "__main__":
    model = next_change_test_train()
    