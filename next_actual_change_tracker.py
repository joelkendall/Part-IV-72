import pandas as pd
import numpy as np
from pathlib import Path
from modules.metrics.metrics import count_dependencies_per_class, categories_per_class

class NextActualChangeTracker:

    # Tracks the next actual change that happens to each class
    
    def __init__(self, min_change_threshold=2):
        self.releases = {}
        self.ordered_releases = []
        self.current_release = None
        self.min_change_threshold = min_change_threshold

    def add_release(self, release_name, df):
        release_metrics = pd.DataFrame(
            count_dependencies_per_class(df).items(),
            columns=['Class', 'Total Dependencies']
        ).set_index('Class')

        cat_data = categories_per_class(df)
        release_metrics = release_metrics.join(cat_data)

        self.releases[release_name] = release_metrics
        self.ordered_releases.append(release_name)
        
        # Update next actual changes for all previous releases
        self._update_next_actual_changes()
        
        self.current_release = release_name
        return release_metrics
    
    def _update_next_actual_changes(self):
        """
        For each release, find the next actual significant change
        that happens to each class (regardless of when)
        """
        
        for i, release in enumerate(self.ordered_releases[:-1]):  # Skip last release
            current_metrics = self.releases[release].copy()
            
            # Find next actual change for each class
            next_actual_change = {}
            releases_until_change = {}
            
            for class_name in current_metrics.index:
                current_deps = current_metrics.loc[class_name, 'Total Dependencies']
                next_change = None
                releases_ahead = None
                
                # Look through ALL future releases until we find a significant change
                for j in range(i + 1, len(self.ordered_releases)):
                    future_release = self.ordered_releases[j]
                    future_metrics = self.releases[future_release]
                    
                    if class_name in future_metrics.index:
                        future_deps = future_metrics.loc[class_name, 'Total Dependencies']
                        change = future_deps - current_deps
                        
                        # Check if this is a significant change
                        if abs(change) >= self.min_change_threshold:
                            next_change = change
                            releases_ahead = j - i
                            break  # Found the next actual change!
                    else:
                        # Class was removed - this is definitely a significant change
                        next_change = -current_deps
                        releases_ahead = j - i
                        break
                
                # If no significant change found, mark as "stable"
                if next_change is None:
                    next_change = 0  # No significant change in remaining releases
                    releases_ahead = len(self.ordered_releases) - i - 1
                
                next_actual_change[class_name] = next_change
                releases_until_change[class_name] = releases_ahead
            
            # Update the release with next actual changes
            self.releases[release]['Next Actual Change'] = pd.Series(next_actual_change)
            self.releases[release]['Releases Until Change'] = pd.Series(releases_until_change)
    
    def get_training_data_next_actual(self):
        """
        Get training data using next actual changes
        """
        training_data = []
        
        # Use all releases except the last few (since they don't have enough future data)
        end_idx = len(self.ordered_releases) - 2  # Keep at least 2 releases for future
        if end_idx <= 0:
            end_idx = len(self.ordered_releases) - 1
        
        for i, release in enumerate(self.ordered_releases[:end_idx]):
            release_metrics = self.releases[release]
            
            # Only include classes that have next actual change data
            if 'Next Actual Change' not in release_metrics.columns:
                continue
            
            for class_name in release_metrics.index:
                row_data = {
                    'Release': release,
                    'Class': class_name,
                    'Total Dependencies': release_metrics.loc[class_name, 'Total Dependencies'],
                    'Next Actual Change': release_metrics.loc[class_name, 'Next Actual Change'],
                    'Releases Until Change': release_metrics.loc[class_name, 'Releases Until Change']
                }
                
                # Add category percentages
                for col in release_metrics.columns:
                    if col not in ['Total Dependencies', 'Next Actual Change', 'Releases Until Change']:
                        row_data[col] = release_metrics.loc[class_name, col]
                
                training_data.append(row_data)
        
        return pd.DataFrame(training_data)
    
    def analyze_next_actual_approach(self):
        """
        Analyze the next actual change approach
        """
        
        print("=== NEXT ACTUAL CHANGE ANALYSIS ===")
        
        # Get training data
        training_data = self.get_training_data_next_actual()
        
        if len(training_data) == 0:
            print("No training data available!")
            return training_data
        
        print(f"Total training samples: {len(training_data)}")
        
        # Analyze change distribution
        from src.ml.ClassChangeModel import ClassChangeModel
        model = ClassChangeModel()
        
        print("\n--- NEXT ACTUAL CHANGES ---")
        change_buckets = training_data['Next Actual Change'].apply(model.change_buckets).value_counts()
        for category, count in change_buckets.items():
            percentage = count / len(training_data) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Analyze timing
        print(f"\n--- TIMING ANALYSIS ---")
        timing_stats = training_data['Releases Until Change'].describe()
        print(f"Average releases until change: {timing_stats['mean']:.1f}")
        print(f"Median releases until change: {timing_stats['50%']:.1f}")
        print(f"Max releases until change: {timing_stats['max']:.0f}")
        
        # Show examples of classes that take time to change
        slow_changers = training_data[training_data['Releases Until Change'] > 2]
        if len(slow_changers) > 0:
            print(f"\n--- EXAMPLES: Classes that change after 3+ releases ---")
            for i, (_, row) in enumerate(slow_changers.head(5).iterrows()):
                change_cat = model.change_buckets(row['Next Actual Change'])
                print(f"  {row['Class'][:50]:<50} | {change_cat:<15} | After {row['Releases Until Change']:.0f} releases")
        
        return training_data
    
    def create_training_data(self):
        """Create training data for machine learning"""
        from src.ml.ClassChangeModel import ClassChangeModel
        model = ClassChangeModel()
        
        # Get raw training data 
        raw_data = self.get_training_data_next_actual()
        
        features = []
        labels = []
        
        for row in raw_data:
            # Extract features (dependency percentages)
            feature_row = []
            for col in ['InvokeInstance', 'Parameter', 'Extends', 'InvokeSpecial', 'GetField', 
                       'ArrayLoad', 'ArrayStore', 'PutField', 'Cast', 'InvokeStatic', 
                       'InvokeVirtual', 'Conditional', 'New', 'InstanceOf', 'Return', 
                       'InvokeInterface', 'Switch', 'NewArray', 'Throw', 'Monitor', 
                       'Load', 'Store']:
                if col in row:
                    feature_row.append(row[col])
                else:
                    feature_row.append(0.0)
            
            # Get label using change buckets
            change = row['Next Actual Change']
            label = model.change_buckets(change)
            
            features.append(feature_row)
            labels.append(label)
        
        return features, labels

def test_next_actual_changes():
    """
    Test the next actual change approach
    """
    
    print("=== TESTING NEXT ACTUAL CHANGE APPROACH ===")
    
    # Load JStock data
    jstock_files = sorted(Path('data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock').glob('*.tsv'))[:10]  # Use more files
    
    print(f"Loading {len(jstock_files)} JStock files...")
    
    tracker = NextActualChangeTracker(min_change_threshold=2)
    
    for f in jstock_files:
        df = pd.read_csv(f, sep='\t', skiprows=26)
        tracker.add_release(f.stem, df)
        print(f"  ✓ {f.name}")
    
    # Analyze the approach
    training_data = tracker.analyze_next_actual_approach()
    
    return tracker, training_data

def create_next_actual_test_scenario():
    """
    Create a test scenario to validate the approach
    """
    
    print("\n=== CREATING TEST SCENARIO ===")
    
    # Load data
    tracker, training_data = test_next_actual_changes()
    
    if len(training_data) == 0:
        print("No training data to create test scenario!")
        return
    
    # Example: Find a class that had a big change and show prediction vs reality
    from src.ml.ClassChangeModel import ClassChangeModel
    model = ClassChangeModel()
    
    big_changes = training_data[
        training_data['Next Actual Change'].apply(model.change_buckets).isin(['Large Increase', 'Large Decrease'])
    ]
    
    if len(big_changes) > 0:
        print(f"\n--- EXAMPLE VALIDATION CASES ---")
        print("These classes had features that should predict their eventual change:")
        
        for i, (_, row) in enumerate(big_changes.head(3).iterrows()):
            predicted_category = model.change_buckets(row['Next Actual Change'])
            print(f"\nClass: {row['Class']}")
            print(f"  Current dependencies: {row['Total Dependencies']:.0f}")
            print(f"  Next actual change: {row['Next Actual Change']:+.0f}")
            print(f"  Category: {predicted_category}")
            print(f"  Releases until change: {row['Releases Until Change']:.0f}")
            print(f"  Key features: InvokeInstance={row.get('InvokeInstance', 0):.0f}, Parameter={row.get('Parameter', 0):.0f}, Extends={row.get('Extends', 0):.0f}")

if __name__ == "__main__":
    create_next_actual_test_scenario()