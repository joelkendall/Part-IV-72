import pandas as pd
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
        
        self._update_next_actual_changes()
        
        self.current_release = release_name
        return release_metrics
    
    def _update_next_actual_changes(self):
        
        for i, release in enumerate(self.ordered_releases[:-1]):  # Skiping prev. release
            current_metrics = self.releases[release].copy()
            
            next_actual_change = {}
            releases_until_change = {}
            
            for class_name in current_metrics.index:
                current_deps = current_metrics.loc[class_name, 'Total Dependencies']
                next_change = None
                releases_ahead = None
                
                #looks through all future releases till we find a significant change
                for j in range(i + 1, len(self.ordered_releases)):
                    future_release = self.ordered_releases[j]
                    future_metrics = self.releases[future_release]
                    
                    if class_name in future_metrics.index:
                        future_deps = future_metrics.loc[class_name, 'Total Dependencies']
                        change = future_deps - current_deps
                        
                        #checking change threshold
                        if abs(change) >= self.min_change_threshold:
                            next_change = change
                            releases_ahead = j - i
                            break  
                    else:
                        #represents classes being removed
                        next_change = -current_deps
                        releases_ahead = j - i
                        break
                
                if next_change is None:
                    next_change = 0 
                    releases_ahead = len(self.ordered_releases) - i - 1
                
                next_actual_change[class_name] = next_change
                releases_until_change[class_name] = releases_ahead
            
            self.releases[release]['Next Actual Change'] = pd.Series(next_actual_change)
            self.releases[release]['Releases Until Change'] = pd.Series(releases_until_change)
    
    def get_training_data_next_actual(self):
        training_data = []
        end_idx = len(self.ordered_releases) - 2  #keeping releaqses for future
        if end_idx <= 0:
            end_idx = len(self.ordered_releases) - 1
        
        for i, release in enumerate(self.ordered_releases[:end_idx]):
            release_metrics = self.releases[release]
            
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
                
                for col in release_metrics.columns:
                    if col not in ['Total Dependencies', 'Next Actual Change', 'Releases Until Change']:
                        row_data[col] = release_metrics.loc[class_name, col]
                
                training_data.append(row_data)
        
        return pd.DataFrame(training_data)
    
    
    def create_training_data(self):
        from src.ml.ClassChangeModel import ClassChangeModel
        model = ClassChangeModel()
        
        raw_data = self.get_training_data_next_actual()
        
        features = []
        labels = []
        
        for row in raw_data:
            #features are dep %
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
            
            change = row['Next Actual Change']
            label = model.change_buckets(change)
            
            features.append(feature_row)
            labels.append(label)
        
        return features, labels

