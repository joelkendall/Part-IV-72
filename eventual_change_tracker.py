#!/usr/bin/env python3
"""
Enhanced ChangeTracker that looks at eventual changes, not just next-release changes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from modules.metrics.metrics import count_dependencies_per_class, categories_per_class

class EventualChangeTracker:
    """
    Tracks changes that happen eventually (within N future releases)
    rather than just the immediate next release
    """
    
    def __init__(self, look_ahead_releases=3):
        """
        look_ahead_releases: How many future releases to check for changes
        """
        self.releases = {}
        self.ordered_releases = []
        self.current_release = None
        self.look_ahead_releases = look_ahead_releases

    def add_release(self, release_name, df):
        release_metrics = pd.DataFrame(
            count_dependencies_per_class(df).items(),
            columns=['Class', 'Total Dependencies']
        ).set_index('Class')

        cat_data = categories_per_class(df)
        release_metrics = release_metrics.join(cat_data)

        self.releases[release_name] = release_metrics
        self.ordered_releases.append(release_name)
        
        # Update eventual changes for all previous releases
        self._update_eventual_changes()
        
        self.current_release = release_name
        return release_metrics
    
    def _update_eventual_changes(self):
        """
        For each release, look at the maximum change that occurs 
        within the next N releases
        """
        
        for i, release in enumerate(self.ordered_releases[:-1]):  # Skip last release
            current_metrics = self.releases[release].copy()
            
            # Look ahead at future releases
            max_eventual_change = {}
            earliest_change_release = {}
            
            for class_name in current_metrics.index:
                current_deps = current_metrics.loc[class_name, 'Total Dependencies']
                max_change = 0
                earliest_release = None
                
                # Check up to look_ahead_releases future releases
                end_idx = min(i + self.look_ahead_releases + 1, len(self.ordered_releases))
                
                for j in range(i + 1, end_idx):
                    future_release = self.ordered_releases[j]
                    future_metrics = self.releases[future_release]
                    
                    if class_name in future_metrics.index:
                        future_deps = future_metrics.loc[class_name, 'Total Dependencies']
                        change = future_deps - current_deps
                        
                        # Keep track of the largest absolute change
                        if abs(change) > abs(max_change):
                            max_change = change
                            earliest_release = future_release
                    else:
                        # Class was removed - count as large decrease
                        removal_change = -current_deps
                        if abs(removal_change) > abs(max_change):
                            max_change = removal_change
                            earliest_release = future_release
                
                max_eventual_change[class_name] = max_change
                earliest_change_release[class_name] = earliest_release
            
            # Update the release with eventual changes
            self.releases[release]['Eventual Change'] = pd.Series(max_eventual_change)
            self.releases[release]['Change Release'] = pd.Series(earliest_change_release)
    
    def get_training_data_eventual(self):
        """
        Get training data using eventual changes instead of immediate changes
        """
        training_data = []
        
        # Use all releases except the last few (since they don't have full look-ahead)
        end_idx = len(self.ordered_releases) - self.look_ahead_releases
        if end_idx <= 0:
            end_idx = len(self.ordered_releases) - 1
        
        for i, release in enumerate(self.ordered_releases[:end_idx]):
            release_metrics = self.releases[release]
            
            # Only include classes that have eventual change data
            if 'Eventual Change' not in release_metrics.columns:
                continue
            
            for class_name in release_metrics.index:
                row_data = {
                    'Release': release,
                    'Class': class_name,
                    'Total Dependencies': release_metrics.loc[class_name, 'Total Dependencies'],
                    'Eventual Change': release_metrics.loc[class_name, 'Eventual Change'],
                    'Change Release': release_metrics.loc[class_name, 'Change Release']
                }
                
                # Add category percentages
                for col in release_metrics.columns:
                    if col not in ['Total Dependencies', 'Eventual Change', 'Change Release']:
                        row_data[col] = release_metrics.loc[class_name, col]
                
                training_data.append(row_data)
        
        return pd.DataFrame(training_data)
    
    def compare_approaches(self):
        """
        Compare immediate vs eventual change approaches
        """
        
        print("=== COMPARISON: IMMEDIATE vs EVENTUAL CHANGES ===")
        
        # Get both types of training data
        immediate_data = self.get_training_data_immediate()
        eventual_data = self.get_training_data_eventual()
        
        print(f"\nImmediate approach: {len(immediate_data)} samples")
        print(f"Eventual approach: {len(eventual_data)} samples")
        
        # Compare change distributions
        from src.ml.ClassChangeModel import ClassChangeModel
        model = ClassChangeModel()
        
        if 'Immediate Change' in immediate_data.columns:
            print("\n--- IMMEDIATE CHANGES ---")
            immediate_buckets = immediate_data['Immediate Change'].apply(model.change_buckets).value_counts()
            for category, count in immediate_buckets.items():
                percentage = count / len(immediate_data) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
        
        if 'Eventual Change' in eventual_data.columns:
            print("\n--- EVENTUAL CHANGES ---") 
            eventual_buckets = eventual_data['Eventual Change'].apply(model.change_buckets).value_counts()
            for category, count in eventual_buckets.items():
                percentage = count / len(eventual_data) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Show examples of classes that would be misclassified
        print(f"\n--- MISCLASSIFICATION EXAMPLES ---")
        
        # Find classes where immediate != eventual
        merged = immediate_data.merge(
            eventual_data[['Class', 'Release', 'Eventual Change']], 
            on=['Class', 'Release'], 
            how='inner'
        )
        
        immediate_cats = merged['Immediate Change'].apply(model.change_buckets)
        eventual_cats = merged['Eventual Change'].apply(model.change_buckets)
        
        mismatched = merged[immediate_cats != eventual_cats]
        
        print(f"Classes misclassified by immediate approach: {len(mismatched)}")
        
        if len(mismatched) > 0:
            print("\nTop 10 examples:")
            for i, (_, row) in enumerate(mismatched.head(10).iterrows()):
                immediate_cat = model.change_buckets(row['Immediate Change'])
                eventual_cat = model.change_buckets(row['Eventual Change'])
                print(f"  {row['Class'][:50]:<50} | Immediate: {immediate_cat:<15} | Eventual: {eventual_cat:<15}")
        
        return immediate_data, eventual_data
    
    def get_training_data_immediate(self):
        """
        Get training data using immediate (next-release) changes for comparison
        """
        training_data = []
        
        for i, release in enumerate(self.ordered_releases[:-1]):
            current_metrics = self.releases[release]
            next_release = self.ordered_releases[i + 1]
            next_metrics = self.releases[next_release]
            
            for class_name in current_metrics.index:
                current_deps = current_metrics.loc[class_name, 'Total Dependencies']
                
                if class_name in next_metrics.index:
                    next_deps = next_metrics.loc[class_name, 'Total Dependencies']
                    immediate_change = next_deps - current_deps
                else:
                    immediate_change = -current_deps  # Class removed
                
                row_data = {
                    'Release': release,
                    'Class': class_name,
                    'Total Dependencies': current_deps,
                    'Immediate Change': immediate_change
                }
                
                # Add category percentages
                for col in current_metrics.columns:
                    if col not in ['Total Dependencies']:
                        if col in current_metrics.columns:
                            row_data[col] = current_metrics.loc[class_name, col]
                
                training_data.append(row_data)
        
        return pd.DataFrame(training_data)

def test_eventual_changes():
    """
    Test the eventual change approach with JStock data
    """
    
    print("=== TESTING EVENTUAL CHANGE APPROACH ===")
    
    # Load JStock data
    jstock_files = sorted(Path('data/jgraph-jmeter-jstock-jung-lucene-weka/jstock_deps/jstock').glob('*.tsv'))[:8]
    
    print(f"Loading {len(jstock_files)} JStock files...")
    
    tracker = EventualChangeTracker(look_ahead_releases=3)
    
    for f in jstock_files:
        df = pd.read_csv(f, sep='\t', skiprows=26)
        tracker.add_release(f.stem, df)
        print(f"  ✓ {f.name}")
    
    # Compare approaches
    immediate_data, eventual_data = tracker.compare_approaches()
    
    return tracker, immediate_data, eventual_data

if __name__ == "__main__":
    test_eventual_changes()