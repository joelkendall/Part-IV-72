import ClassChangeModel
from utils.ChangeTracker import ChangeTracker
import pandas as pd
from pathlib import Path

def train_with_multiple_datasets():
    model = ClassChangeModel()
    
    # DATASET PATHS
    datasets = [
        {
            'name': 'JUnit',
            'path': 'data/junit-depfiles/junit-depfiles',
            'max_files': 8  
        },
        {
            'name': 'JStock', 
            'path': 'data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock',
            'max_files': 15
        },
        {
            'name': 'Jung',
            'path': 'data/jgraph-jmeter-jstock-jung-lucene-weka/jung_deps/jung',
            'max_files': 10
        },
        {
            'name': 'Lucene',
            'path': 'data/jgraph-jmeter-jstock-jung-lucene-weka/lucene_deps/lucene',
            'max_files': 20
        }
    ]
    
    all_training_data = []
    
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']} Dataset")
        path = Path(dataset['path'])
        
        if not path.exists():
            print(f"Skipping {dataset['name']} - path not found: {path}")
            continue
            
        tracker = ChangeTracker()
        tsv_files = sorted(path.glob('*.tsv'))[:dataset['max_files']]
        
        if not tsv_files:
            print(f"No TSV files found in {path}")
            continue
            
        print(f"Found {len(tsv_files)} files")
        
        for f in tsv_files:
            try:
                df = pd.read_csv(f, sep='\t', skiprows=26)
                tracker.add_release(f.stem, df)
                print(f"  ✓ {f.name}: {len(df)} dependencies")
            except Exception as e:
                print(f"  ✗ {f.name}: Error - {e}")
        
        training_data = tracker.get_training_data()
        if len(training_data) > 0:
            print(f"Generated {len(training_data)} training samples")
            all_training_data.append(training_data)
        else:
            print("No training data generated")
    
    if not all_training_data:
        print("No training data available!")
        return None
    
    combined_data = pd.concat(all_training_data, ignore_index=True)
    print(f"\n=== Combined Training Data ===")
    print(f"Total samples: {len(combined_data)}")
    print(f"Features: {len([col for col in combined_data.columns if col not in ['Class', 'Release', 'Incoming Change']])}")
    
    print("\nChange distribution:")
    bucket_distribution = combined_data['Incoming Change'].apply(model.change_buckets).value_counts()
    print(bucket_distribution)
    
    print(f"\n=== Training Model ===")
    results = model.train(combined_data)
    
    model_file = f"model_trained_on_{len(all_training_data)}_datasets.pkl"
    model.save_model(model_file)
    
    print(f"\n=== Training Results ===")
    print(f"Model saved to: {model_file}")
    print(f"Training accuracy: {model.training_history[-1]['test_accuracy']:.3f}")
    
    print("\nTop 10 Most Important Features:")
    top_features = results['feature_importance'].head(10)
    for _, row in top_features.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return model

def show_results():
    
    model_files = list(Path('.').glob('*.pkl'))
    if not model_files:
        print("No saved models found. Train a model first.")
        return
    
    print("Available trained models:")
    for f in model_files:
        print(f"  - {f.name}")
    
    # Load the most recent model
    latest_model = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\nLoading latest model: {latest_model.name}")
    
    model = ClassChangeModel()
    model.load_model(str(latest_model))
    
    print(f"Model training history:")
    for i, session in enumerate(model.training_history):
        accuracy = session.get('test_accuracy', 'N/A')
        print(f"  Session {i+1}: {session['data_shape']} - Accuracy: {accuracy}")
    
    # Example prediction (you could load new data here)
    print(f"\nModel is ready for predictions!")
    print(f"Use model.predict_changes(new_data_df) to make predictions")
    
    return model

if __name__ == "__main__":
    # Train with available data
    trained_model = train_with_multiple_datasets()
    
    # Demonstrate reusing the model
    if trained_model:
        show_results()
