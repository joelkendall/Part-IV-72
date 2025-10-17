from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

class ClassChangeModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.data_columns = []
        self.cat_cols = []
        self.is_trained = False
        self.training_history = []
        
    def change_buckets(self, change):
        if pd.isna(change):
            return 'Little to No Change'
        elif -2 <= change <= 2:
            return 'Little to No Change'
        elif -9 <= change < -2:
            return 'Small Decrease'
        elif 2 < change < 9:
            return 'Small Increase'
        elif change < -10:
            return 'Large Decrease'
        elif change >= 10:
            return 'Large Increase'
        else:
            return 'Little to No Change'
    
    def process_data(self, df):
        df_clean = df.copy()
        
        data_columns = [col for col in df_clean.columns if col not in ['Class', 'Release', 'Incoming Change']]
        category_cols = [col for col in data_columns if col not in ['Total Dependencies', 'Class', 'Incoming Change', 'Release']]
        
        for col in category_cols:
            df_clean[col] = df_clean[col].fillna(0)
        
        X = df_clean[data_columns]
        y = df_clean['Incoming Change'].apply(self.change_buckets)

        self.data_columns = data_columns
        self.cat_cols = category_cols

        print(f"Data shape after cleaning: {X.shape}")
        print(f"Target distribution: {y.value_counts()}")
        print(f"Features with NaN: {X.isnull().sum().sum()}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale only Total Dependencies column
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled['Total Dependencies'] = self.scaler.fit_transform(X_train[['Total Dependencies']])
        X_test_scaled['Total Dependencies'] = self.scaler.transform(X_test[['Total Dependencies']])
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, tracker_df):
        # Process and train
        X_train, X_test, y_train, y_test = self.process_data(tracker_df)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Mark as trained
        self.is_trained = True
        self.training_history.append({
            'data_shape': tracker_df.shape,
            'timestamp': pd.Timestamp.now(),
            'test_accuracy': (y_pred == y_test).mean()
        })

        # Create importance DataFrame
        importance = pd.DataFrame({
            'Feature': self.data_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance)
        
        # Calculate category importance
        category_importance = importance[
            importance['Feature'].isin(self.cat_cols)
        ]
        print("\nCategory Influence on Changes:")
        print(category_importance)
        
        # Define all possible labels for confusion matrix
        all_labels = ['Little to No Change', 'Small Decrease', 'Small Increase', 'Large Decrease', 'Large Increase']
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=all_labels),
            'feature_importance': importance
        }
    
    def save_model(self, filepath: str = "trained_model.pkl"):
        """
        Save the trained model, scaler, and metadata to disk
        """
        if not self.is_trained:
            print("Warning: Model hasn't been trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'data_columns': self.data_columns,
            'cat_cols': self.cat_cols,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """
        Load a previously trained model from disk
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.data_columns = model_data['data_columns']
        self.cat_cols = model_data['cat_cols']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Training history: {len(self.training_history)} sessions")
        return self
    
    def retrain_with_new_data(self, new_tracker_df):
        """
        Retrain the model with additional data
        Note: Random Forest doesn't support true incremental learning,
        so this combines old and new data for retraining
        """
        if not self.is_trained:
            print("No previous model found. Training from scratch...")
            return self.train(new_tracker_df)
        
        print("Retraining model with additional data...")
        
        # For now, we need to retrain from scratch with combined data
        # In a real scenario, you'd combine this with previously saved training data
        results = self.train(new_tracker_df)
        
        self.training_history.append({
            'data_shape': new_tracker_df.shape,
            'timestamp': pd.Timestamp.now()
        })
        
        return results
    
    def predict_changes(self, class_data_df):
        """
        Predict dependency changes for new class data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Clean and prepare the data the same way as training
        df_clean = class_data_df.copy()
        
        # Ensure we have the same columns as training
        missing_cols = set(self.data_columns) - set(df_clean.columns)
        if missing_cols:
            for col in missing_cols:
                df_clean[col] = 0  # Fill missing columns with 0
        
        # Fill NaN values
        for col in self.cat_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        X = df_clean[self.data_columns]
        
        # Scale Total Dependencies
        X_scaled = X.copy()
        if 'Total Dependencies' in X_scaled.columns:
            X_scaled['Total Dependencies'] = self.scaler.transform(X[['Total Dependencies']])
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Class': class_data_df.index if hasattr(class_data_df, 'index') else range(len(predictions)),
            'Predicted_Change': predictions,
            'Confidence': probabilities.max(axis=1)
        })
        
        # Set index to Class names for easier lookup
        results.set_index('Class', inplace=True)
        
        # Add probability columns for each class
        class_names = self.model.classes_
        for i, class_name in enumerate(class_names):
            results[f'Prob_{class_name}'] = probabilities[:, i]
        
        return results
