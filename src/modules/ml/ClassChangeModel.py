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
        X = df[data_columns]
        y = df['Incoming Change'].apply(self.change_buckets)

        return X, y
    
