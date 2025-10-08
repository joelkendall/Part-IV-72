# Tracks releases so we can see how classes changed in the future i hope
import pandas as pd
from pathlib import Path
from packaging.version import parse
from modules.metrics.metrics import count_dependencies_per_class, categories_per_class

class ChangeTracker:
    def __init__(self):
        self.releases = {}
        self.ordered_releases = []
        self.current_release = None

    def add_release(self, release_name, df, use_source = True):
        release_metrics = pd.DataFrame(
            count_dependencies_per_class(df).items(),
            columns=['Class', 'Total Dependencies']
        ).set_index('Class')

        cat_data = categories_per_class(df)
        release_metrics = release_metrics.join(cat_data)

        self.releases[release_name] = release_metrics
        self.ordered_releases.append(release_name)

        if len(self.ordered_releases) > 1:
            previous_release = self.ordered_releases[-2]
            self._update_changes(previous_release, release_name)
        
        self.current_release = release_name
        return release_metrics
    
    def _update_changes(self, previous_release, current_release):
        previous_metrics = self.releases[previous_release]
        current_metrics = self.releases[current_release]

        previous_metrics['Incoming Change'] = (
            current_metrics['Total Dependencies'].reindex(previous_metrics.index).fillna(0) - previous_metrics['Total Dependencies']
        )

        missing_classes = ~previous_metrics.index.isin(current_metrics.index)
        previous_metrics.loc[missing_classes, 'Incoming Change'] = -previous_metrics.loc[missing_classes, 'Total Dependencies']

    def print_release_summary(self):
        for i, release in enumerate(self.ordered_releases[:-1]):
            current_metrics = self.releases[release]
            print(f"\nRelease: {release}")
            print(f"Total Classes: {len(current_metrics)}")
            
            changes = current_metrics['Incoming Change']
            print(f"Classes with changes: {(changes != 0).sum()}")
            print(f"Removed classes: {(changes < 0).sum()}")
            print(f"Modified classes: {((changes != 0) & (changes > -current_metrics['Total Dependencies'])).sum()}")
            
            print("\nTop 5 largest changes:")
            print(current_metrics.nlargest(5, 'Incoming Change')[['Total Dependencies', 'Incoming Change']])
    
    def get_training_data(self):
        """
        Get training data from all releases except the last one.
        Returns a DataFrame with features and changes for ML training.
        """
        training_data = []
        
        # Skip the last release as it won't have future changes
        for release in self.ordered_releases[:-1]:
            release_metrics = self.releases[release]
            
            # Create a row for each class
            for class_name in release_metrics.index:
                row_data = {
                    'Release': release,
                    'Class': class_name,
                    'Total Dependencies': release_metrics.loc[class_name, 'Total Dependencies'],
                    'Incoming Change': release_metrics.loc[class_name, 'Incoming Change']
                }
                
                # Add category percentages
                for col in release_metrics.columns:
                    if col not in ['Total Dependencies', 'Incoming Change']:
                        row_data[col] = release_metrics.loc[class_name, col]
                
                training_data.append(row_data)
        
        return pd.DataFrame(training_data)

