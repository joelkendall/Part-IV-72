from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

class ClassChangeModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.data_columns = []
        self.cat_cols = []
        
    def change_buckets(change):
        match change:
            case change if change < 9 or change > -9:
                return 'Little to No Change'
            case change if  -50 <= change < -9:
                return 'Small Decrease'
            case change if 9 <= change < 50:
                return 'Small Increase'
            case change if change < -50:
                return 'Large Decrease'
            case change if change >= 50:
                return 'Large Increase'
    
    def process_data(self, df):
        data_columns = [col for col in df.columns if col not in ['Class', 'Incoming Change']]
        category_cols = [col for col in data_columns if col not in ['Total Dependencies', 'Class', 'Incoming Change']]
        X = df[data_columns]
        y = df['Incoming Change'].apply(self.change_buckets)

        self.data_columns = data_columns
        self.cat_cols = category_cols

        return X, y
    
    def train(self, tracker_df):

        X_train, X_test, y_train, y_test = self.process_data(tracker_df)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        importance = pd.DataFrame({
                'Characteristic': self.data_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
        print("\nCharacteristic Importance:")
        print(importance)
        
        
        category_importance = importance[
            importance['Characteristic'].isin(self.cat_cols)
        ]
        print("\nCategory Influence on Changes:")
        print(category_importance)
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=self.model.classes_),
            'Characteristic_importance': importance
        } 
    
