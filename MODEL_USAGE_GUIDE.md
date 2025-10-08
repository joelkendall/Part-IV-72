# Enhanced Machine Learning Model - Usage Guide

## What's New

The `ClassChangeModel` has been enhanced with several powerful features:

### 1. **Model Persistence**
- Save trained models to disk
- Load previously trained models
- Preserve training history and metadata

### 2. **Incremental Training**
- Train with initial dataset
- Add more data and retrain
- Track training sessions over time

### 3. **Prediction Capabilities**
- Make predictions on new class data
- Get prediction probabilities
- Confidence scores for each prediction

### 4. **Training History Tracking**
- Track what data was used for training
- Monitor model performance over time
- Compare training sessions

## How to Use

### Basic Training and Saving
```python
from src.modules.ml.ClassChangeModel import ClassChangeModel
from src.utils.ChangeTracker import ChangeTracker

# Create and train model
model = ClassChangeModel()
# ... load your data into tracker_df ...
results = model.train(tracker_df)

# Save the trained model
model.save_model("my_trained_model.pkl")
```

### Loading and Reusing a Trained Model
```python
# Load a previously trained model
model = ClassChangeModel()
model.load_model("my_trained_model.pkl")

# Check training history
print(f"Training sessions: {len(model.training_history)}")
for session in model.training_history:
    print(f"Data shape: {session['data_shape']}, Accuracy: {session['test_accuracy']}")

# Make predictions on new data
predictions = model.predict_changes(new_class_data_df)
```

### Training with More Data
```python
# Method 1: Retrain with additional data
model.load_model("existing_model.pkl")
new_results = model.retrain_with_new_data(additional_tracker_df)

# Method 2: Combine datasets before training
combined_data = pd.concat([original_data, new_data], ignore_index=True)
model.train(combined_data)
```

### Making Predictions
```python
# Predict dependency changes for new classes
predictions = model.predict_changes(class_features_df)

# Results include:
# - Predicted_Change: The predicted category
# - Confidence: Maximum probability (0-1)
# - Prob_[Category]: Probability for each change category

print(predictions[['Class', 'Predicted_Change', 'Confidence']])
```

## File Examples

### Available Pre-trained Models
```
- test_model.pkl (152 KB) - Small test model
- junit_trained_model.pkl (735 KB) - Trained on JUnit data
- model_trained_on_1_datasets.pkl (1.1 MB) - Enhanced model with more data
```

### Training Scripts
```
- quick_test.py - Basic functionality test
- advanced_training.py - Train with multiple datasets
- enhanced_training.py - Full feature demonstration
```

## Benefits of Enhanced Model

1. **No Need to Retrain from Scratch**: Save your trained model and reuse it
2. **Incremental Learning**: Add new data without losing previous training
3. **Better Performance with More Data**: Combine multiple datasets for better accuracy
4. **Production Ready**: Make predictions on new dependency data
5. **Training Insights**: Track how your model improves over time

## Next Steps

1. **Collect More Data**: Add more TSV files from different projects
2. **Experiment with Parameters**: Try different Random Forest settings
3. **Feature Engineering**: Add new calculated features
4. **Cross-Validation**: Implement more robust training evaluation
5. **Web Interface**: Create a GUI for easy model training and prediction

## Performance Notes

The enhanced model with more training data shows:
- Better feature importance distribution
- More robust predictions
- Higher confidence scores
- Better generalization to new projects

Train with as much diverse dependency data as possible for best results!
